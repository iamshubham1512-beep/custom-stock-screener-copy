import streamlit as st
import polars as pl
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
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

# Hide number steppers for better UI
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
# 📂 CONFIG & CONSTANTS
# ======================
DATASETS = {
    "2016-2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021-2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet",
}

CACHE_DIR = Path(".hf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ======================
# 🧠 HELPERS
# ======================
def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def _download_with_cache(url: str, cache_dir: Path = CACHE_DIR, max_retries: int = 3, timeout: int = 60) -> Path:
    """Download Parquet file with local caching and ETag validation."""
    digest = _hash_url(url)
    data_path = cache_dir / f"{digest}.parquet"
    etag_path = cache_dir / f"{digest}.etag"
    headers = {}

    if etag_path.exists():
        headers["If-None-Match"] = etag_path.read_text().strip()

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 304 and data_path.exists():
                return data_path
            resp.raise_for_status()

            # Save ETag and data
            etag = resp.headers.get("ETag")
            if etag:
                etag_path.write_text(etag)

            tmp_path = data_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(data_path)
            return data_path
        except Exception as e:
            if data_path.exists():
                st.warning(f"⚠️ Using cached data ({url.split('/')[-1]}): {e}")
                return data_path
            if attempt == max_retries:
                raise OSError(f"Failed to fetch {url}: {e}")
            time.sleep(1.5 * attempt)

    return data_path

# ======================
# 🧩 LOAD DATA (Lazy + Fast)
# ======================
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data() -> pl.LazyFrame:
    """Efficiently loads both Parquet files as LazyFrames and combines."""
    lazy_frames = []
    for name, url in DATASETS.items():
        local_path = _download_with_cache(url)
        try:
            lf = pl.scan_parquet(str(local_path))
            lazy_frames.append(lf)
        except Exception as e:
            st.error(f"❌ Failed to read {name}: {e}")
    if not lazy_frames:
        raise RuntimeError("No valid datasets loaded.")
    return pl.concat(lazy_frames, how="vertical")

@st.cache_data(ttl=3600)
def load_master_table() -> pl.DataFrame:
    """Loads and normalizes combined dataset into memory."""
    df = load_all_data().collect(streaming=True)

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

    required = {"SYMBOL", "DATE", "Open", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize data types
    df = df.with_columns([
        pl.col("SYMBOL").cast(pl.Utf8).str.strip_chars().alias("SYMBOL"),
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("DATE"),
        pl.col("Open").cast(pl.Float64).alias("Open"),
        pl.col("Close").cast(pl.Float64).alias("Close"),
        pl.col("Volume").fill_null(0).cast(pl.Int64).alias("Volume"),
    ])

    return df.filter(pl.col("SYMBOL").is_not_null() & (pl.col("SYMBOL") != ""))

# ======================
# 🔤 SYMBOL LIST
# ======================
@st.cache_data(ttl=3600)
def list_all_symbols() -> List[str]:
    df = load_master_table()
    return sorted(df.select(pl.col("SYMBOL").unique()).to_series().to_list())

# ======================
# 📅 YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2016, datetime.now().year + 1))[::-1])

# ======================
# 📊 YEARLY DATA FETCH
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    master = load_master_table()

    start, end = datetime(year, 1, 1).date(), datetime(year, 12, 31).date()
    df_year = master.filter(
        pl.col("SYMBOL").is_in(symbols) &
        (pl.col("DATE") >= start) & (pl.col("DATE") <= end)
    ).select(["SYMBOL", "DATE", "Open", "Close", "Volume"])

    if df_year.is_empty():
        empty_schema = {
            "SYMBOL": pl.Utf8, "Open Price": pl.Float64, "Close Price": pl.Float64,
            "% Change": pl.Float64, "Avg. Volume": pl.Int64, "First Traded Date": pl.Date
        }
        empty_df = pl.DataFrame(schema=empty_schema)
        return empty_df, empty_df

    per_symbol = (
        df_year
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg([
            pl.col("Open").first().alias("Open Price"),
            pl.col("Close").last().alias("Close Price"),
            pl.col("Volume").mean().cast(pl.Int64).fill_null(0).alias("Avg. Volume")
        ])
        .with_columns(((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100).alias("% Change"))
        .with_columns(pl.col("% Change").round(2))
        .sort("% Change", descending=True)
    )

    # Add first traded date
    first_trade = (
        master
        .filter(pl.col("SYMBOL").is_in(symbols))
        .select(["SYMBOL", "DATE"])
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").first().alias("First Traded Date"))
    )

    per_symbol = per_symbol.join(first_trade, on="SYMBOL", how="left")
    df_final = per_symbol.filter(pl.col("% Change") > 0)
    return df_final, per_symbol

# ======================
# 🧹 CACHE CONTROL
# ======================
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("✅ Cache cleared successfully.")

# ======================
# 🔍 FETCH BUTTON
# ======================
all_symbols = list_all_symbols()
if all_symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}..."):
            df_final, df_all = fetch_yearly_data(all_symbols, year)
            if not df_final.is_empty():
                st.session_state["fetched_data_pl"] = df_final
                st.session_state["fetched_all_pl"] = df_all
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {df_final.height} positive gainers out of {df_all.height} stocks.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data_pl"] = None
else:
    st.stop()

# ======================
# 🧮 FILTERS & RESULTS
# ======================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    open_min = int(np.floor(df_result.select(pl.min("Open Price")).item() / 10) * 10)
    open_max = int(np.ceil(df_result.select(pl.max("Open Price")).item() / 10) * 10)
    pct_min = int(np.floor(df_result.select(pl.min("% Change")).item() / 10) * 10)
    pct_max = int(np.ceil(df_result.select(pl.max("% Change")).item() / 10) * 10)

    st.markdown(f"#### Open Price Range (₹) — Min ({open_min}) · Max ({open_max})")
    c1, c2 = st.columns(2)
    with c1:
        open_low = st.number_input("Min", min_value=open_min, max_value=open_max, value=open_min)
    with c2:
        open_high = st.number_input("Max", min_value=open_min, max_value=open_max, value=open_max)

    st.markdown(f"#### % Change Range — Min ({pct_min}) · Max ({pct_max})")
    c3, c4 = st.columns(2)
    with c3:
        pct_low = st.number_input("Min %", min_value=pct_min, max_value=pct_max, value=pct_min)
    with c4:
        pct_high = st.number_input("Max %", min_value=pct_min, max_value=pct_max, value=pct_max)

    filtered = df_result.filter(
        (pl.col("Open Price") >= open_low) & (pl.col("Open Price") <= open_high) &
        (pl.col("% Change") >= pct_low) & (pl.col("% Change") <= pct_high)
    )

    vol_filter = st.selectbox("Filter by Avg. Volume", [
        "All", "More than 100K", "More than 200K", "More than 300K", "More than 400K", "More than 500K"
    ])
    if vol_filter != "All":
        threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered = filtered.filter(pl.col("Avg. Volume") > threshold)

    filtered_df = filtered.to_pandas()
    filtered_df.index = range(1, len(filtered_df) + 1)
    filtered_df.index.name = "Sl. No."

    st.write(f"📈 Showing {len(filtered_df)} results after filters:")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv,
                       file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv",
                       mime="text/csv")

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Hugging Face Parquet (Chiron-S/NSE_Stocks_Data) | Optimized for multi-file caching & Polars lazy loading | Built by Shubham Kishor")
