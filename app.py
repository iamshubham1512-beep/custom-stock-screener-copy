import os
import time
import requests
import tempfile
from pathlib import Path
from datetime import datetime
import polars as pl
import streamlit as st
import numpy as np

# ======================================================
# ⚙️ Streamlit Config
# ======================================================
st.set_page_config(page_title="📊 NSE Yearly Stock Screener", layout="wide")

st.title("📈 NSE Stocks Yearly Screener (2016–2024)")
st.caption("Data powered by Hugging Face | Cached locally for speed 🚀")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ======================================================
# 🧱 Hugging Face Parquet File URLs
# ======================================================
HF_FILES = {
    "2016_2020": "https://huggingface.co/datasets/your-hf-username/your-repo/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021_2024": "https://huggingface.co/datasets/your-hf-username/your-repo/resolve/main/NSE_Stocks_2021_2024.parquet",
}

# ======================================================
# 🧩 Download with Cache + ETag Validation
# ======================================================
def _download_with_cache(url: str, max_retries: int = 3, timeout: int = 15) -> Path:
    filename = Path(url.split("/")[-1])
    local_path = CACHE_DIR / filename
    etag_path = local_path.with_suffix(".etag")

    try:
        head = requests.head(url, timeout=timeout)
        etag = head.headers.get("ETag")
    except Exception:
        etag = None

    if local_path.exists() and etag_path.exists():
        cached = etag_path.read_text().strip()
        if etag == cached:
            return local_path

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)
            tmp_path.replace(local_path)
            if etag:
                etag_path.write_text(etag)
            return local_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"❌ Failed to download {url}: {e}")

# ======================================================
# 🧠 Load & Merge Both Parquet Datasets
# ======================================================
@st.cache_data(show_spinner=True)
def load_master_table() -> pl.DataFrame:
    local_files = [_download_with_cache(u) for u in HF_FILES.values()]
    dfs = []

    for file in local_files:
        try:
            lf = pl.scan_parquet(str(file))
            cols_present = lf.columns
            required_cols = ["SYMBOL", "DATE", "Open", "High", "Low", "Close", "Volume"]

            for col in required_cols:
                if col not in cols_present:
                    lf = lf.with_columns(pl.lit(None).alias(col))

            df = (
                lf.with_columns([
                    pl.col("SYMBOL").cast(pl.Utf8, strict=False).str.strip_chars().alias("SYMBOL"),
                    pl.col("DATE").str.strip_chars().str.strptime(pl.Date, fmt=None, strict=False).alias("DATE"),
                    pl.col("Open").cast(pl.Float64, strict=False),
                    pl.col("High").cast(pl.Float64, strict=False),
                    pl.col("Low").cast(pl.Float64, strict=False),
                    pl.col("Close").cast(pl.Float64, strict=False),
                    pl.col("Volume").fill_null(0).cast(pl.Int64, strict=False),
                ])
                .filter(pl.col("SYMBOL").is_not_null())
                .collect(streaming=True)
            )
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Skipped {file.name} due to: {e}")

    if not dfs:
        st.error("❌ No valid data files found.")
        return pl.DataFrame()

    combined = pl.concat(dfs, how="vertical_relaxed").unique(subset=["SYMBOL", "DATE"])
    return combined

# ======================================================
# 🔤 Helper — List All Symbols
# ======================================================
@st.cache_data(show_spinner=False)
def list_all_symbols() -> list[str]:
    df = load_master_table()
    return sorted(df["SYMBOL"].unique().to_list())

# ======================================================
# 🧮 Compute Yearly Stats
# ======================================================
@st.cache_data(show_spinner=True)
def fetch_yearly_data(year: int) -> pl.DataFrame:
    df = load_master_table()
    df = df.filter(pl.col("DATE").dt.year() == year)

    if df.is_empty():
        st.warning(f"No data found for year {year}.")
        return pl.DataFrame()

    yearly_stats = (
        df.group_by("SYMBOL")
        .agg([
            pl.first("Open").alias("Open"),
            pl.last("Close").alias("Close"),
            (pl.last("Close") - pl.first("Open")).alias("Change"),
            ((pl.last("Close") - pl.first("Open")) / pl.first("Open") * 100).alias("PctChange"),
            pl.mean("Volume").alias("AvgVolume"),
            (pl.col("DATE").max() - pl.col("DATE").min()).alias("DaysActive"),
        ])
        .sort("PctChange", descending=True)
    )

    return yearly_stats

# ======================================================
# 🎛️ Sidebar Controls
# ======================================================
st.sidebar.header("⚙️ Filters")

selected_year = st.sidebar.selectbox("Select Year", list(range(2016, 2025))[::-1])
min_price, max_price = st.sidebar.slider("Price Range", 0.0, 5000.0, (0.0, 5000.0))
min_pct, max_pct = st.sidebar.slider("Change % Range", -100.0, 100.0, (-100.0, 100.0))
min_vol = st.sidebar.number_input("Min Avg Volume", 0, 10_000_000, 0)
min_days = st.sidebar.number_input("Min Days Active", 0, 365, 30)

if st.sidebar.button("🔁 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared successfully.")

# ======================================================
# 📊 Display Yearly Data
# ======================================================
with st.spinner(f"Fetching {selected_year} data..."):
    df = fetch_yearly_data(selected_year)

if not df.is_empty():
    filtered = df.filter(
        (pl.col("Open") >= min_price)
        & (pl.col("Close") <= max_price)
        & (pl.col("PctChange") >= min_pct)
        & (pl.col("PctChange") <= max_pct)
        & (pl.col("AvgVolume") >= min_vol)
        & (pl.col("DaysActive") >= min_days)
    )

    pd_df = filtered.to_pandas()
    st.subheader(f"📈 Filtered Stocks — {selected_year}")
    st.dataframe(pd_df, use_container_width=True)

    # Top 10 chart
    top10 = pd_df.nlargest(10, "PctChange")
    if not top10.empty:
        import plotly.express as px
        fig = px.bar(top10, x="SYMBOL", y="PctChange", title="Top 10 Gainers", text="PctChange")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for selected filters.")
