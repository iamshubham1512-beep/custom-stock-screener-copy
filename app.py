import polars as pl
import streamlit as st
import os
import tempfile
import requests

# ======================================================
# ⚙️ CONFIGURATION
# ======================================================
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/tree/main"
PARQUET_FILES = [
    "NSE_Stocks_2016_2020.parquet",
    "NSE_Stocks_2021_2024.parquet"
]
CACHE_DIR = os.path.join(tempfile.gettempdir(), "nse_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ======================================================
# 🧩 DOWNLOAD WITH LOCAL CACHING
# ======================================================
def download_with_cache(file_name: str) -> str:
    """Download parquet file from Hugging Face with local caching."""
    cached_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(cached_path):
        return cached_path

    url = f"{HUGGINGFACE_BASE_URL}/{file_name}"
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(cached_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return cached_path
    except Exception as e:
        st.error(f"❌ Failed to download {file_name}: {e}")
        raise


# ======================================================
# 🧠 LOAD & CLEAN PARQUET FILES
# ======================================================
@st.cache_data(show_spinner=False, persist="disk")
def load_master_table_parquet() -> pl.DataFrame:
    """Load NSE parquet files, clean, and merge them."""
    dfs = []
    for file_name in PARQUET_FILES:
        try:
            file_path = download_with_cache(file_name)
            df = pl.read_parquet(file_path)

            # 🔍 Normalize column names (case-insensitive)
            df = df.rename({c: c.upper().strip() for c in df.columns})

            # 🧹 Only process known columns that exist
            expected_cols = {
                "SYMBOL": pl.Utf8,
                "DATE": pl.Utf8,
                "OPEN": pl.Float64,
                "HIGH": pl.Float64,
                "LOW": pl.Float64,
                "CLOSE": pl.Float64,
                "VOLUME": pl.Int64
            }

            # keep only columns present in the DataFrame
            cols_to_keep = [c for c in expected_cols if c in df.columns]

            df = df.select(cols_to_keep)

            # 🧩 Apply conversions safely
            for col, dtype in expected_cols.items():
                if col in df.columns:
                    try:
                        if dtype == pl.Utf8:
                            df = df.with_columns(pl.col(col).cast(pl.Utf8).str.strip_chars().alias(col))
                        elif dtype == pl.Float64:
                            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                        elif dtype == pl.Int64:
                            df = df.with_columns(pl.col(col).cast(pl.Int64).fill_null(0).alias(col))
                    except Exception:
                        pass  # continue silently if one column fails

            # 🚫 Filter invalid data
            if "SYMBOL" in df.columns and "DATE" in df.columns:
                df = df.filter(
                    pl.col("SYMBOL").is_not_null() &
                    pl.col("DATE").is_not_null()
                )

            dfs.append(df)

        except Exception as e:
            st.warning(f"⚠️ Skipping {file_name}: {e}")

    # 🔗 Merge all data
    if dfs:
        final_df = pl.concat(dfs, how="vertical", rechunk=True).unique(subset=["SYMBOL", "DATE"])
        return final_df.sort("DATE")
    else:
        st.error("❌ No valid data loaded from parquet files.")
        return pl.DataFrame()


# ======================================================
# 🎯 FETCH YEARLY DATA
# ======================================================
@st.cache_data(show_spinner=False)
def fetch_yearly_data(symbols: list[str], year: int):
    """Fetch stock data filtered by year and symbols."""
    master = load_master_table_parquet()
    if master.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    if "DATE" not in master.columns or "SYMBOL" not in master.columns:
        st.error("❌ Missing essential columns in dataset.")
        return pl.DataFrame(), pl.DataFrame()

    # Convert to datetime for filtering
    master = master.with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d").alias("DATE"))

    df_year = master.filter(pl.col("DATE").dt.year() == year)

    if symbols:
        df_year = df_year.filter(pl.col("SYMBOL").is_in(symbols))

    return df_year, master


# ======================================================
# 🧪 TEST BLOCK
# ======================================================
if __name__ == "__main__":
    st.write("🚀 Testing Hugging Face Data Loader")

    df = load_master_table_parquet()
    st.write(f"Loaded {len(df):,} rows, {len(df['SYMBOL'].unique()) if 'SYMBOL' in df.columns else 0} symbols")

    df_2023, df_all = fetch_yearly_data(["TCS", "INFY"], 2023)
    st.write(df_2023.head())
