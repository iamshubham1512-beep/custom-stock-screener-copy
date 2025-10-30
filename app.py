import streamlit as st
import polars as pl
import os
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener (NSE)", layout="wide")

# Hide Streamlit default number input spinners
st.markdown("""
    <style>
    div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
    div[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    div[data-testid="stNumberInput"] input[type=number] {
        -moz-appearance: textfield;
    }
    </style>
""", unsafe_allow_html=True)


# ======================
# 📂 CONSTANTS
# ======================
CACHE_DIR = Path(".hf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Updated data sources (separated by year ranges)
DATA_SOURCES = {
    "2016_2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021_2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet",
}

# ======================
# ⚙️ HELPER FUNCTIONS
# ======================
def _select_data_source(year: int) -> str:
    """Select correct data source based on year range."""
    if 2016 <= year <= 2020:
        return DATA_SOURCES["2016_2020"]
    elif 2021 <= year <= 2024:
        return DATA_SOURCES["2021_2024"]
    else:
        return None


def _download_with_cache(url: str) -> str:
    """Download file if not already cached."""
    file_hash = hashlib.sha1(url.encode()).hexdigest()
    cached_path = CACHE_DIR / f"{file_hash}.parquet"

    if cached_path.exists():
        return str(cached_path)

    st.info(f"Downloading dataset from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(cached_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(cached_path)


@st.cache_data(show_spinner=False)
def load_master_table_parquet(year: int) -> pl.DataFrame:
    """Load and cache master stock data based on the selected year."""
    source_url = _select_data_source(year)
    if not source_url:
        st.warning(f"⚠️ Data for the year {year} is currently unavailable.")
        return pl.DataFrame()  # Return empty DataFrame instead of raising error

    local_file = _download_with_cache(source_url)

    try:
        df = pl.read_parquet(local_file)
    except Exception as e:
        st.error(f"❌ Failed to read data file: {e}")
        return pl.DataFrame()

    # Normalize columns
    expected_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df_cols = [c.lower().strip() for c in df.columns]
    rename_map = dict(zip(df.columns, df_cols))
    df = df.rename(rename_map)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"⚠️ Missing columns: {missing}")
        return pl.DataFrame()

    # Cleanup & normalization
    df = df.with_columns([
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).dt.date().alias("date"),
        pl.col("symbol").str.strip_chars().alias("symbol")
    ])
    return df.drop_nulls()


@st.cache_data
def list_all_symbols(year: int):
    """List all available stock symbols."""
    df = load_master_table_parquet(year)
    if df.is_empty():
        return []
    return sorted(df["symbol"].unique().to_list())


def build_first_trade_map(df: pl.DataFrame) -> dict:
    """Map each stock to its first-ever trade date."""
    if df.is_empty():
        return {}
    first_trade = (
        df.group_by("symbol")
        .agg(pl.col("date").min().alias("first_trade"))
        .to_dict(as_series=False)
    )
    return dict(zip(first_trade["symbol"], first_trade["first_trade"]))


@st.cache_data
def fetch_yearly_data(year: int):
    """Compute yearly performance data."""
    df = load_master_table_parquet(year)
    if df.is_empty():
        return pl.DataFrame()

    df = df.filter(pl.col("date").dt.year() == year)

    first_day = df.group_by("symbol").agg(pl.col("date").min().alias("first_date"))
    last_day = df.group_by("symbol").agg(pl.col("date").max().alias("last_date"))

    merged = first_day.join(last_day, on="symbol", how="inner")
    merged = merged.join(df, left_on=["symbol", "first_date"], right_on=["symbol", "date"]).rename({
        "open": "open_price"
    })
    merged = merged.join(df, left_on=["symbol", "last_date"], right_on=["symbol", "date"]).rename({
        "close": "close_price",
        "volume": "last_volume"
    })

    merged = merged.with_columns([
        ((pl.col("close_price") - pl.col("open_price")) / pl.col("open_price") * 100).round(2).alias("pct_change"),
        pl.col("last_volume").cast(pl.Float64).alias("volume")
    ])

    merged = merged.select(["symbol", "open_price", "close_price", "pct_change", "volume"])
    return merged.drop_nulls()


# ======================
# 🧮 UI ELEMENTS
# ======================
st.title("📈 Yearly Stock Screener (NSE)")
st.caption("Analyze top gainers by year, open/close performance, and volume trends.")

col1, col2 = st.columns([1, 2])
with col1:
    year = st.number_input("Select Year", min_value=2016, max_value=2024, value=2024, step=1)
with col2:
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")

# ======================
# 🧩 MAIN LOGIC
# ======================
if st.button("📊 Fetch Yearly Data"):
    with st.spinner("Processing data..."):
        df_year = fetch_yearly_data(year)

        if df_year.is_empty():
            st.warning("No data available for the selected year.")
        else:
            st.success(f"Loaded {len(df_year)} records for {year}.")

            # Filters
            min_open = float(df_year["open_price"].min())
            max_open = float(df_year["open_price"].max())
            open_range = st.slider("Filter by Open Price", min_open, max_open, (min_open, max_open))

            min_change = float(df_year["pct_change"].min())
            max_change = float(df_year["pct_change"].max())
            pct_range = st.slider("Filter by % Change", min_change, max_change, (min_change, max_change))

            df_filtered = df_year.filter(
                (pl.col("open_price") >= open_range[0]) &
                (pl.col("open_price") <= open_range[1]) &
                (pl.col("pct_change") >= pct_range[0]) &
                (pl.col("pct_change") <= pct_range[1])
            )

            st.write(f"📊 Filtered Stocks: {len(df_filtered)}")
            st.dataframe(df_filtered.to_pandas(), use_container_width=True)

            csv = df_filtered.write_csv()
            st.download_button("📥 Download CSV", csv, "filtered_stocks.csv", "text/csv")

st.markdown("---")
st.caption("⚙️ Data Source: Hugging Face Datasets | Engine: Polars | UI: Streamlit")
