import streamlit as st
import polars as pl
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import requests
import hashlib
import time
from pathlib import Path

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")

st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year to view top-performing NSE stocks based on yearly price change and average volume.")

# Hide +/- steppers
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

# ======================
# 📂 HELPERS
# ======================
def strip_indices(symbols: List[str]) -> List[str]:
    return [s for s in symbols if not str(s).startswith("^")]

def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(s).strip().upper() for s in symbols if s is not None))

# ======================
# 📦 DATA SOURCE (Dynamic based on year)
# ======================
HF_PARQUET_MAP = {
    "2016_2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/blob/main/NSE_Stocks_2016_2020.parquet",
    "2021_2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/blob/main/NSE_Stocks_2021_2024.parquet",
}

CACHE_DIR = Path(".hf_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _download_with_cache(url: str, cache_dir: Path = CACHE_DIR, max_retries: int = 3, timeout: int = 60) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    data_path = cache_dir / f"{digest}.parquet"
    etag_path = cache_dir / f"{digest}.etag"

    headers = {}
    if etag_path.exists():
        etag = etag_path.read_text().strip()
        if etag:
            headers["If-None-Match"] = etag

    session = requests.Session()
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 304 and data_path.exists():
                return data_path
            resp.raise_for_status()
            etag = resp.headers.get("ETag")
            if etag:
                etag_path.write_text(etag)
            tmp_path = data_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(data_path)
            return data_path
        except Exception:
            if attempt == max_retries:
                if data_path.exists():
                    return data_path
                raise
            time.sleep(1.5 * attempt)

    if data_path.exists():
        return data_path
    raise OSError("Failed to download Parquet after retries.")

def _select_data_source(year: int) -> str:
    """Choose which Parquet file to load based on the year."""
    if 2016 <= year <= 2020:
        return HF_PARQUET_MAP["2016_2020"]
    elif 2021 <= year <= 2024:
        return HF_PARQUET_MAP["2021_2024"]
    else:
        raise ValueError(f"No data available for year {year}")

# ======================
# 🧠 LOAD MASTER PARQUET (Per Year)
# ======================
@st.cache_data(ttl=3600)
def load_master_table_parquet(year: int) -> pl.DataFrame:
    source_url = _select_data_source(year)
    local_path = _download_with_cache(source_url)
    df = pl.read_parquet(str(local_path))

    # Normalize columns
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("symbol", "stock"):
            rename_map[c] = "SYMBOL"
        elif cl in ("date", "datetime", "timestamp"):
            rename_map[c] = "DATE"
        elif cl == "open":
            rename_map[c] = "Open"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "volume":
            rename_map[c] = "Volume"
    df = df.rename(rename_map)

    required = {"SYMBOL", "DATE", "Open", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    # Cast & clean data
    df = df.with_columns([
        pl.col("SYMBOL").cast(pl.Utf8).str.strip_chars().alias("SYMBOL"),
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, strict=False, fmt=None).alias("DATE"),
        pl.col("Open").cast(pl.Float64).alias("Open"),
        pl.col("Close").cast(pl.Float64).alias("Close"),
        pl.col("Volume").cast(pl.Float64).fill_null(0).alias("Volume")
    ])

    df = df.filter(pl.col("SYMBOL").is_not_null() & (pl.col("SYMBOL") != ""))
    return df

@st.cache_data(ttl=3600)
def list_all_symbols(year: int) -> List[str]:
    master = load_master_table_parquet(year)
    syms = master.select(pl.col("SYMBOL").unique()).to_series().to_list()
    return strip_indices(syms)

# ======================
# 📅 YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2016, datetime.now().year + 1))[::-1])

# ======================
# ⚡ YEARLY METRICS
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    master = load_master_table_parquet(year)
    start_date, end_date = datetime(year, 1, 1).date(), datetime(year, 12, 31).date()

    df_year = (
        master.filter(
            pl.col("SYMBOL").is_in(symbols) &
            (pl.col("DATE") >= start_date) &
            (pl.col("DATE") <= end_date)
        ).select(["SYMBOL", "DATE", "Open", "Close", "Volume"])
    )

    if df_year.is_empty():
        empty = pl.DataFrame(schema={
            "SYMBOL": pl.Utf8,
            "Open Price": pl.Float64,
            "Close Price": pl.Float64,
            "% Change": pl.Float64,
            "Avg. Volume": pl.Float64,
            "First Traded Date": pl.Date
        })
        return empty, empty

    per_symbol = (
        df_year.sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg([
            pl.col("Open").first().alias("Open Price"),
            pl.col("Close").last().alias("Close Price"),
            pl.col("Volume").mean().alias("Avg. Volume"),
        ])
        .with_columns(((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100.0)
                      .round(2)
                      .alias("% Change"))
        .select(["SYMBOL", "Open Price", "Close Price", "% Change", "Avg. Volume"])
    )

    first_trade = (
        master.filter(pl.col("SYMBOL").is_in(symbols))
        .select(["SYMBOL", "DATE"])
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").first().alias("First Traded Date"))
    )

    per_symbol = per_symbol.join(first_trade, on="SYMBOL", how="left").sort(by="% Change", descending=True)

    df_all = per_symbol
    df_final = df_all.filter(pl.col("% Change") > 0)

    return df_final, df_all

# ======================
# 🧮 AGE MAP
# ======================
@st.cache_data(ttl=86400)
def build_first_trade_map(symbols: List[str], year: int, max_threshold_years: int = 5) -> Dict[str, datetime]:
    master = load_master_table_parquet(year)
    if not symbols:
        return {}

    cutoff_year = year - max_threshold_years
    start = datetime(max(cutoff_year - 1, 1990), 1, 1).date()
    end = datetime(year, 12, 31).date()

    df = (
        master.filter(
            pl.col("SYMBOL").is_in(symbols) &
            (pl.col("DATE") >= start) &
            (pl.col("DATE") <= end)
        )
        .select(["SYMBOL", "DATE"])
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").first().alias("FIRST_DATE"))
    )

    return {row["SYMBOL"]: row["FIRST_DATE"] for row in df.iter_rows(named=True)}

def filter_by_age_pl(df_pl: pl.DataFrame, year: int, age_option: str) -> pl.DataFrame:
    if df_pl.is_empty() or age_option == "All":
        return df_pl

    threshold_map = {"Older than 1 year": 1, "Older than 2 years": 2, "Older than 3 years": 3}
    N = threshold_map.get(age_option, 0)

    symbols_list = df_pl.select(pl.col("SYMBOL").unique()).to_series().to_list()
    first_trade_map = build_first_trade_map(symbols_list, year, max_threshold_years=max(5, N + 1))

    cutoff = datetime(year - N, 12, 31).date()
    valid = {s for s, d in first_trade_map.items() if d <= cutoff}

    return df_pl.filter(pl.col("SYMBOL").is_in(list(valid)))

# ======================
# 🧹 CACHE CONTROL UI
# ======================
c1, c2 = st.columns(2)
with c1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared! App will refresh.")

# ======================
# 🔍 FETCH BUTTON
# ======================
all_symbols = list_all_symbols(year)

if all_symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}..."):
            df_final_pl, df_all_pl = fetch_yearly_data(all_symbols, year)
            if not df_final_pl.is_empty():
                st.session_state["fetched_data_pl"] = df_final_pl
                st.session_state["fetched_all_pl"] = df_all_pl
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {df_final_pl.height} gainers out of {df_all_pl.height} symbols.")
            else:
                st.warning("⚠️ No positive gainers found.")
else:
    st.stop()

# ======================
# 🎛️ FILTER UI + TABLE
# ======================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result_pl = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Numeric bounds
    try:
        open_min_bound = int(np.floor(df_result_pl.select(pl.min("Open Price")).item() / 10) * 10)
        open_max_bound = int(np.ceil(df_result_pl.select(pl.max("Open Price")).item() / 10) * 10)
        pct_min_bound = int(np.floor(df_result_pl.select(pl.min("% Change")).item() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result_pl.select(pl.max("% Change")).item() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Inputs
    st.markdown("#### Open Price Range (₹)")
    cmin, cmax = st.columns(2)
    with cmin:
        open_min = st.number_input("Min", value=open_min_bound)
    with cmax:
        open_max = st.number_input("Max", value=open_max_bound)

    st.markdown("#### % Change Range")
    cpmin, cpmax = st.columns(2)
    with cpmin:
        pct_min = st.number_input("Min %", value=pct_min_bound)
    with cpmax:
        pct_max = st.number_input("Max %", value=pct_max_bound)

    filtered_pl = df_result_pl.filter(
        (pl.col("Open Price") >= open_min) &
        (pl.col("Open Price") <= open_max) &
        (pl.col("% Change") >= pct_min) &
        (pl.col("% Change") <= pct_max)
    )

    vol_filter = st.selectbox("Filter by Avg. Volume", [
        "All", "More than 100K", "More than 200K", "More than 300K", "More than 400K", "More than 500K"
    ])
    if vol_filter != "All":
        threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_pl = filtered_pl.filter(pl.col("Avg. Volume") > threshold)

    age_filter = st.selectbox("Company Older Than", [
        "All", "Older than 1 year", "Older than 2 years", "Older than 3 years"
    ])
    if age_filter != "All":
        with st.spinner("Filtering by age..."):
            filtered_pl = filter_by_age_pl(filtered_pl, year, age_filter)

    # Display table
    filtered_pd = filtered_pl.to_pandas().reset_index(drop=True)
    filtered_pd.index = range(1, len(filtered_pd) + 1)
    filtered_pd.index.name = "Sl. No."

    st.write(f"📈 Showing {len(filtered_pd)} results:")
    st.dataframe(filtered_pd, use_container_width=True)

    csv = filtered_pd.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv,
                       file_name=f"Filtered_Gainers_{year}.csv", mime="text/csv")

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Hugging Face (Chiron-S/NSE_Stocks_Data) | Built by Shubham Kishor | Cached for 1 hour.")
