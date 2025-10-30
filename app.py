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

# ======================
# 💄 UI TWEAKS
# ======================
st.markdown("""
<style>
button.step-up { display: none !important; }
button.step-down { display: none !important; }
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
input[type=number] {
  -moz-appearance: textfield;
}
</style>
""", unsafe_allow_html=True)

# ======================
# ⚙️ HELPERS
# ======================
def strip_indices(symbols: List[str]) -> List[str]:
    return [s for s in symbols if not str(s).startswith("^")]

def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(s).strip().upper() for s in symbols if s is not None))

# ======================
# 📦 DATA SOURCES (Hugging Face + Local Cache)
# ======================
PARQUET_URLS = {
    "2016_2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021_2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet"
}
CACHE_DIR = Path(".hf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ======================
# ⚡ SMART DOWNLOAD + LOCAL CACHE
# ======================
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
        except Exception as e:
            if attempt == max_retries:
                if data_path.exists():
                    return data_path
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")
            time.sleep(1.5 * attempt)

# ======================
# 🧠 DATA LOADER
# ======================
@st.cache_data(ttl=3600)
def load_master_table_parquet() -> pl.DataFrame:
    dfs = []
    for key, url in PARQUET_URLS.items():
        try:
            local_path = _download_with_cache(url)
            df = pl.read_parquet(str(local_path))
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {key}: {e}")

    if not dfs:
        st.error("No data loaded from Hugging Face. Please check URLs or internet connection.")
        st.stop()

    df = pl.concat(dfs, how="vertical_relaxed")

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
        elif cl == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(rename_map)

    # Ensure mandatory columns
    required = {"SYMBOL", "DATE", "Open", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean columns
    df = df.with_columns([
        pl.col("SYMBOL").cast(pl.Utf8).str.strip_chars().alias("SYMBOL"),
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, strict=False, exact=False).alias("DATE"),
        pl.col("Open").cast(pl.Float64),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Int64).fill_null(0)
    ])

    # Drop blanks
    df = df.filter(pl.col("SYMBOL").is_not_null() & (pl.col("SYMBOL") != ""))
    return df

@st.cache_data(ttl=3600)
def list_all_symbols() -> List[str]:
    master = load_master_table_parquet()
    syms = master.select(pl.col("SYMBOL").unique()).to_series().to_list()
    return strip_indices(syms)

# ======================
# 📅 YEAR SELECT
# ======================
year = st.selectbox("Select Year", options=list(range(2016, datetime.now().year + 1))[::-1])

# ======================
# 📈 YEARLY ANALYSIS
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    master = load_master_table_parquet()

    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()

    df_year = (
        master
        .filter(pl.col("SYMBOL").is_in(symbols) & (pl.col("DATE") >= start_date) & (pl.col("DATE") <= end_date))
        .select(["SYMBOL", "DATE", "Open", "Close", "Volume"])
    )

    if df_year.is_empty():
        empty = pl.DataFrame(schema={
            "SYMBOL": pl.Utf8,
            "Open Price": pl.Float64,
            "Close Price": pl.Float64,
            "% Change": pl.Float64,
            "Avg. Volume": pl.Int64,
            "First Traded Date": pl.Date
        })
        return empty, empty

    per_symbol = (
        df_year
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg([
            pl.col("Open").first().alias("Open Price"),
            pl.col("Close").last().alias("Close Price"),
            pl.col("Volume").mean().cast(pl.Int64).fill_null(0).alias("Avg. Volume")
        ])
        .with_columns([
            ((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100.0).round(2).alias("% Change")
        ])
    )

    first_trade = (
        master
        .filter(pl.col("SYMBOL").is_in(symbols))
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").min().alias("First Traded Date"))
    )

    per_symbol = per_symbol.join(first_trade, on="SYMBOL", how="left").sort(by="% Change", descending=True)
    df_all = per_symbol
    df_final = df_all.filter(pl.col("% Change") > 0)

    return df_final, df_all

# ======================
# 🧮 AGE MAP
# ======================
@st.cache_data(ttl=86400)
def build_first_trade_map(symbols: List[str], selected_year: int, max_threshold_years: int = 5) -> Dict[str, datetime]:
    master = load_master_table_parquet()
    cutoff_year = selected_year - max_threshold_years
    start = datetime(max(cutoff_year - 1, 1990), 1, 1).date()
    end = datetime(selected_year, 12, 31).date()

    df = (
        master
        .filter(pl.col("SYMBOL").is_in(symbols) & (pl.col("DATE") >= start) & (pl.col("DATE") <= end))
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").min().alias("FIRST_DATE"))
    )

    return {row["SYMBOL"]: row["FIRST_DATE"] for row in df.iter_rows(named=True)}

def filter_by_age_pl(df_pl: pl.DataFrame, year: int, age_option: str) -> pl.DataFrame:
    if df_pl.is_empty() or age_option == "All":
        return df_pl

    threshold_map = {"Older than 1 year": 1, "Older than 2 years": 2, "Older than 3 years": 3}
    if age_option not in threshold_map:
        return df_pl

    N = threshold_map[age_option]
    symbols_list = df_pl.select(pl.col("SYMBOL").unique()).to_series().to_list()
    first_trade_map = build_first_trade_map(symbols_list, year, max_threshold_years=max(5, N + 1))
    cutoff = datetime(year - N, 12, 31).date()
    valid = [s for s, d in first_trade_map.items() if d <= cutoff]
    return df_pl.filter(pl.col("SYMBOL").is_in(valid))

# ======================
# 🧹 CACHE CONTROL
# ======================
if st.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cleared all caches. Please reload the app.")

# ======================
# 🔍 FETCH + FILTER
# ======================
all_symbols = list_all_symbols()

if all_symbols and st.button("🔎 Fetch Yearly Data"):
    with st.spinner(f"Fetching data for {year}..."):
        df_final, df_all = fetch_yearly_data(all_symbols, year)
        if df_final.is_empty():
            st.warning("No positive gainers found.")
        else:
            st.session_state["fetched_data"] = df_final
            st.session_state["fetched_year"] = year
            st.success(f"✅ Found {df_final.height} positive gainers.")

# ======================
# 🎛️ FILTERS
# ======================
if "fetched_data" in st.session_state:
    df_result = st.session_state["fetched_data"]
    st.subheader(f"📊 Filtered Results for {st.session_state['fetched_year']}")

    open_min = int(np.floor(df_result["Open Price"].min() / 10) * 10)
    open_max = int(np.ceil(df_result["Open Price"].max() / 10) * 10)
    pct_min = int(np.floor(df_result["% Change"].min() / 10) * 10)
    pct_max = int(np.ceil(df_result["% Change"].max() / 10) * 10)

    c1, c2 = st.columns(2)
    with c1:
        min_open = st.number_input("Min Open", open_min, open_max, open_min)
    with c2:
        max_open = st.number_input("Max Open", open_min, open_max, open_max)

    c3, c4 = st.columns(2)
    with c3:
        min_pct = st.number_input("Min % Change", pct_min, pct_max, pct_min)
    with c4:
        max_pct = st.number_input("Max % Change", pct_min, pct_max, pct_max)

    filtered = df_result.filter(
        (pl.col("Open Price") >= min_open) &
        (pl.col("Open Price") <= max_open) &
        (pl.col("% Change") >= min_pct) &
        (pl.col("% Change") <= max_pct)
    )

    vol_filter = st.selectbox("Avg. Volume Filter", ["All", "More than 100K", "More than 200K", "More than 300K"])
    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered = filtered.filter(pl.col("Avg. Volume") > vol_threshold)

    age_filter = st.selectbox("Company Age", ["All", "Older than 1 year", "Older than 2 years", "Older than 3 years"])
    if age_filter != "All":
        filtered = filter_by_age_pl(filtered, st.session_state["fetched_year"], age_filter)

    df_final_pd = filtered.to_pandas().reset_index(drop=True)
    df_final_pd.index += 1
    df_final_pd.index.name = "Sl. No."

    st.dataframe(df_final_pd, use_container_width=True)

    csv = df_final_pd.to_csv().encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv", mime="text/csv")

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Hugging Face Parquets (2016–2020, 2021–2024) | Optimized by Shubham Kishor | Cached for 1 hour.")
