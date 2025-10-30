import polars as pl
import streamlit as st
import os
import tempfile
import requests
from functools import lru_cache

# ======================================================
# ⚙️ CONFIGURATION
# ======================================================
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET_NAME/resolve/main"
PARQUET_FILES = [
    "NSE_Stocks_2016_2020.parquet",
    "NSE_Stocks_2021_2024.parquet"
]

CACHE_DIR = os.path.join(tempfile.gettempdir(), "nse_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ======================================================
# 🧩 UTILITY: Download with caching
# ======================================================
def download_with_cache(file_name: str) -> str:
    """Download parquet from Hugging Face with local caching."""
    cached_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(cached_path):
        return cached_path

    url = f"{HUGGINGFACE_BASE_URL}/{file_name}"
    try:
        st.write(f"🔽 Downloading {file_name} ...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(cached_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write(f"✅ Cached: {cached_path}")
        return cached_path
    except Exception as e:
        st.error(f"❌ Error downloading {file_name}: {e}")
        raise e


# ======================================================
# 🧠 LOAD DATA (cached in memory)
# ======================================================
@st.cache_data(show_spinner=False, persist="disk")
def load_master_table() -> pl.DataFrame:
    """Load both parquet files, clean & merge them."""
    try:
        dfs = []
        for file_name in PARQUET_FILES:
            file_path = download_with_cache(file_name)
            df = pl.read_parquet(file_path)

            # Basic cleaning & type casting
            df = df.with_columns([
                pl.col("SYMBOL").cast(pl.Utf8).str.strip_chars().alias("SYMBOL"),
                pl.col("DATE").str.strip_chars().alias("DATE"),
                pl.col("OPEN").cast(pl.Float64).alias("OPEN"),
                pl.col("HIGH").cast(pl.Float64).alias("HIGH"),
                pl.col("LOW").cast(pl.Float64).alias("LOW"),
                pl.col("CLOSE").cast(pl.Float64).alias("CLOSE"),
                pl.col("Volume").fill_null(0).cast(pl.Int64).alias("Volume")
            ])

            # Filter invalid rows
            df = df.filter(
                pl.col("SYMBOL").is_not_null() &
                pl.col("DATE").is_not_null() &
                pl.col("CLOSE").is_not_null()
            )

            dfs.append(df)

        # Merge all years & remove duplicates
        full_df = pl.concat(dfs, how="vertical").unique(subset=["SYMBOL", "DATE"])
        st.success(f"✅ Loaded {len(full_df):,} rows from {len(PARQUET_FILES)} files.")
        return full_df

    except Exception as e:
        st.error(f"❌ Error loading master table: {e}")
        return pl.DataFrame()


# ======================================================
# 🔍 LIST ALL SYMBOLS
# ======================================================
@st.cache_data(show_spinner=False)
def list_all_symbols() -> list[str]:
    """Return list of all stock symbols."""
    df = load_master_table()
    if df.is_empty():
        return []
    return sorted(df["SYMBOL"].unique().to_list())


# ======================================================
# 🎯 FILTER DATA BY SYMBOL AND DATE RANGE
# ======================================================
def fetch_stock_data(symbol: str, start_date: str = None, end_date: str = None) -> pl.DataFrame:
    """Fetch filtered stock data efficiently."""
    df = load_master_table()
    if df.is_empty():
        return df

    data = df.filter(pl.col("SYMBOL") == symbol)

    if start_date:
        data = data.filter(pl.col("
