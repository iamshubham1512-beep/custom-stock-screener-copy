import streamlit as st
import polars as pl
import requests, io
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple

# ======================================================
# 🎯 APP CONFIGURATION
# ======================================================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")
st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year to view top-performing NSE stocks based on yearly price change and average volume.")

# Hide number_input spinners
st.markdown("""
<style>
button.step-up, button.step-down { display: none !important; }
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button {
  -webkit-appearance: none; margin: 0;
}
input[type=number] { -moz-appearance: textfield; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 📂 DATA SOURCES (Hugging Face)
# ======================================================
DATA_URLS = {
    "2016-2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021-2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet"
}

# ======================================================
# ⚡ DATA LOADING WITH POLARS + CACHE
# ======================================================
@st.cache_data(ttl=3600, show_spinner=True)
def load_parquet_from_hf(url: str) -> pl.DataFrame:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        data = io.BytesIO(response.content)
        df = pl.read_parquet(data)

        # Normalize columns
        expected_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        df.columns = [c.lower() for c in df.columns]

        # Fix date if needed
        if df['date'].dtype != pl.Datetime:
            df = df.with_columns([
                pl.when(pl.col('date').str.contains('-'))
                .then(pl.col('date').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
                .otherwise(pl.col('date').str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False))
                .alias('date')
            ])

        # Clean and add year
        df = df.drop_nulls(['date', 'open', 'close']).with_columns(
            pl.col('date').dt.year().alias('year')
        )

        return df
    except Exception as e:
        st.error(f"❌ Error loading {url}: {e}")
        return pl.DataFrame()

@st.cache_data(ttl=3600)
def load_master_table_parquet() -> pl.DataFrame:
    df1 = load_parquet_from_hf(DATA_URLS["2016-2020"])
    df2 = load_parquet_from_hf(DATA_URLS["2021-2024"])
    if not df1.is_empty() and not df2.is_empty():
        df = pl.concat([df1, df2], how="vertical_relaxed")
    else:
        df = df1 if not df1.is_empty() else df2
    return df

@st.cache_data(ttl=3600)
def list_all_symbols() -> List[str]:
    master = load_master_table_parquet()
    return sorted(master.select(pl.col("symbol").unique()).to_series().to_list())

# ======================================================
# 📅 YEAR SELECTION
# ======================================================
year = st.selectbox("Select Year", options=list(range(2016, datetime.now().year + 1))[::-1])

# ======================================================
# 🧠 AGE MAP + FILTERS
# ======================================================
def strip_indices(symbols: List[str]) -> List[str]:
    return [s for s in symbols if not str(s).startswith("^")]

def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(s).strip().upper() for s in symbols if s is not None))

@st.cache_data(ttl=86400)
def build_first_trade_map(symbols: List[str], selected_year: int, max_threshold_years: int = 5) -> Dict[str, datetime]:
    master = load_master_table_parquet()
    if not symbols:
        return {}

    cutoff_year = selected_year - max_threshold_years
    start = datetime(max(cutoff_year - 1, 1990), 1, 1).date()
    end = datetime(selected_year, 12, 31).date()

    df = (
        master
        .filter(
            pl.col("symbol").is_in(symbols) &
            (pl.col("date") >= start) & (pl.col("date") <= end) &
            pl.col("open").is_not_null() & pl.col("close").is_not_null()
        )
        .select(["symbol", "date"])
        .sort(["symbol", "date"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("date").first().alias("FIRST_DATE"))
    )

    return {r["symbol"]: r["FIRST_DATE"] for r in df.iter_rows(named=True)}

def filter_by_age_pl(df_pl: pl.DataFrame, year: int, age_option: str) -> pl.DataFrame:
    if df_pl.is_empty() or age_option == "All":
        return df_pl

    threshold_map = {"Older than 1 year": 1, "Older than 2 years": 2, "Older than 3 years": 3}
    if age_option not in threshold_map:
        return df_pl

    N = threshold_map[age_option]
    symbols_list = df_pl.select(pl.col("symbol").unique()).to_series().to_list()
    first_trade_map = build_first_trade_map(symbols_list, year, max(5, N + 1))
    cutoff = datetime(year - N, 12, 31).date()
    valid = {s for s, d in first_trade_map.items() if d <= cutoff}
    return df_pl.filter(pl.col("symbol").is_in(list(valid)))

# ======================================================
# 🧹 CACHE CLEAR BUTTON
# ======================================================
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared all caches. App will refresh.")

# ======================================================
# 🧮 YEARLY CALCULATION FUNCTION
# ======================================================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    master = load_master_table_parquet()
    df_year = master.filter(pl.col("year") == year)

    if df_year.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    yearly_data = (
        df_year.group_by("symbol")
        .agg([
            pl.col("open").first().alias("Open Price"),
            pl.col("close").last().alias("Close Price"),
            pl.col("volume").mean().alias("Avg. Volume"),
        ])
        .with_columns([
            ((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100)
            .round(2)
            .alias("% Change")
        ])
        .sort("% Change", descending=True)
    )

    gainers = yearly_data.filter(pl.col("% Change") > 0)
    return gainers, yearly_data

# ======================================================
# 🔍 FETCH YEARLY DATA
# ======================================================
all_symbols = list_all_symbols()

if all_symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_final_pl, df_all_pl = fetch_yearly_data(all_symbols, year)
            if not df_final_pl.is_empty():
                st.session_state["fetched_data_pl"] = df_final_pl
                st.session_state["fetched_all_pl"] = df_all_pl
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {df_final_pl.height} positive gainers out of {df_all_pl.height} stocks.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data_pl"] = None
else:
    st.stop()

# ======================================================
# 🎛️ FILTERS + DISPLAY
# ======================================================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result_pl = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Range bounds
    try:
        open_min_bound = int(np.floor(df_result_pl.select(pl.min("Open Price")).item() / 10) * 10)
        open_max_bound = int(np.ceil(df_result_pl.select(pl.max("Open Price")).item() / 10) * 10)
        pct_min_bound = int(np.floor(df_result_pl.select(pl.min("% Change")).item() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result_pl.select(pl.max("% Change")).item() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Open & % Change filters
    cmin, cmax = st.columns(2)
    st.session_state.open_min_val = cmin.number_input("Min Open", min_value=open_min_bound, max_value=open_max_bound, value=open_min_bound)
    st.session_state.open_max_val = cmax.number_input("Max Open", min_value=open_min_bound, max_value=open_max_bound, value=open_max_bound)

    cpmin, cpmax = st.columns(2)
    st.session_state.pct_min_val = cpmin.number_input("Min % Change", min_value=pct_min_bound, max_value=pct_max_bound, value=pct_min_bound)
    st.session_state.pct_max_val = cpmax.number_input("Max % Change", min_value=pct_min_bound, max_value=pct_max_bound, value=pct_max_bound)

    # Apply filters
    filtered_pl = df_result_pl.filter(
        (pl.col("Open Price") >= st.session_state.open_min_val) &
        (pl.col("Open Price") <= st.session_state.open_max_val) &
        (pl.col("% Change") >= st.session_state.pct_min_val) &
        (pl.col("% Change") <= st.session_state.pct_max_val)
    )

    # Volume & age filters
    vol_filter = st.selectbox("Filter by Avg. Volume", ["All", "More than 100K", "More than 200K", "More than 300K"])
    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_pl = filtered_pl.filter(pl.col("Avg. Volume") > vol_threshold)

    age_filter = st.selectbox("Company Older Than", ["All", "Older than 1 year", "Older than 2 years", "Older than 3 years"])
    if age_filter != "All":
        with st.spinner(f"Filtering companies {age_filter.lower()}..."):
            filtered_pl = filter_by_age_pl(filtered_pl, st.session_state["fetched_year"], age_filter)

    # Display & download
    filtered_pd = filtered_pl.to_pandas().reset_index(drop=True)
    filtered_pd.index = range(1, len(filtered_pd) + 1)
    filtered_pd.index.name = "Sl. No."

    st.write(f"📈 Showing {len(filtered_pd)} results after filters:")
    st.dataframe(filtered_pd, use_container_width=True)

    csv = filtered_pd.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv,
                       file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv",
                       mime="text/csv")

# ======================================================
# 🧾 FOOTNOTE
# ======================================================
st.caption("Data Source: Hugging Face Parquet (2016–2020 & 2021–2024) | Built by Shubham Kishor | Cached for 1 hour.")
