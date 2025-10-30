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
    df = df.filter(pl.col("SYMBOL").is_not_null() &
