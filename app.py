import streamlit as st
# Switch to Polars for speed
import polars as pl
from datetime import datetime
import os
import numpy as np
from typing import Dict, List, Tuple

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")

st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year to view top-performing NSE stocks based on yearly price change and average volume.")

# Hide +/- steppers on all number_input widgets (Streamlit lacks a native switch)
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
# 📂 HELPERS
# ======================
def file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def ensure_ns(sym: str) -> str:
    # Keeping for compatibility with earlier code; dataset already uses NSE symbols
    return sym

def strip_indices(symbols):
    return [s for s in symbols if not str(s).startswith("^")]

# ======================
# 📂 LOAD STOCK SYMBOLS (from CSV list, unchanged)
# ======================
@st.cache_data
def load_stock_list(file_path: str, file_mtime_key: float):
    # Read symbol list using Polars for consistency, return python list
    df = pl.read_csv(file_path)
    if "SYMBOL" not in df.columns:
        raise ValueError("The file must contain a column named 'SYMBOL' (all caps).")
    syms = (
        df.select(pl.col("SYMBOL").cast(pl.Utf8).str.strip())
          .drop_nulls()
          .unique()
          .to_series()
          .to_list()
    )
    return strip_indices(syms)

csv_path = "NSE Stocks List.csv"
try:
    symbols = load_stock_list(csv_path, file_mtime(csv_path))
except FileNotFoundError:
    st.error("⚠️ File 'NSE Stocks List.csv' not found in repository root.")
    symbols = []
except Exception as e:
    st.error(f"⚠️ Error loading stock list: {e}")
    symbols = []

# ======================
# 📅 YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2019, datetime.now().year + 1))[::-1])

# ======================
# ⚡ DATA SOURCE: Hugging Face Parquet + Polars
# ======================
HF_PARQUET_URL = "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/All_Stocks_Master.parquet?download=true"

@st.cache_data(ttl=3600)
def load_master_table_parquet():
    """
    Load the full master table from the Parquet on Hugging Face using Polars.
    Expected columns (typical for OHLCV): SYMBOL, DATE, Open, Close, Volume, etc.
    DATE will be converted to date dtype; SYMBOL normalized to string.
    """
    # pl.read_parquet can stream over HTTP. If the hosting requires Range, Polars handles it for simple files.
    df = pl.read_parquet(HF_PARQUET_URL)
    # Normalize columns we use
    cols = df.columns
    # Try to standardize column names (case-insensitive handling)
    rename_map = {}
    for c in cols:
        cl = c.lower()
        if cl == "symbol":
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

    # Ensure minimal required columns exist
    required = {"SYMBOL", "DATE", "Open", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    # Types
    df = df.with_columns([
        pl.col("SYMBOL").cast(pl.Utf8).str.strip(),
        pl.col("DATE").str.strptime(pl.Datetime, strict=False, utc=False).cast(pl.Date)
    ])

    return df

# ======================
# ⚡ FETCH YEARLY DATA (from Parquet via Polars)
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int):
    """
    Compute yearly per-symbol metrics:
    - Open Price: first available 'Open' of the year by DATE ascending
    - Close Price: last available 'Close' of the year by DATE ascending
    - % Change: ((Close_last - Open_first) / Open_first) * 100
    - Avg. Volume: mean Volume over the year (0 if missing)
    Keep only symbols that appear in the selected year; filter later by % Change > 0 like before.
    """
    master = load_master_table_parquet()

    # Filter to selected symbols and year range
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()

    df_year = (
        master
        .filter(pl.col("SYMBOL").is_in(symbols) & (pl.col("DATE") >= pl.lit(start_date)) & (pl.col("DATE") <= pl.lit(end_date)))
        .select(["SYMBOL", "DATE", "Open", "Close", pl.col("Volume").fill_null(0).alias("Volume")])
    )

    if df_year.is_empty():
        # Return empty like the original
        import pandas as pd
        empty_df = pd.DataFrame(columns=["SYMBOL", "Open Price", "Close Price", "% Change", "Avg. Volume"])
        return empty_df, empty_df

    # Compute metrics per symbol using Polars expressions
    # First open of the year and last close of the year
    per_symbol = (
        df_year
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg([
            pl.col("Open").first().alias("Open Price"),
            pl.col("Close").last().alias("Close Price"),
            pl.when(pl.col("Volume").len() > 0).then(pl.col("Volume").mean().cast(pl.Int64)).otherwise(pl.lit(0)).alias("Avg. Volume"),
        ])
        .with_columns([
            ((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100.0).round(2).alias("% Change")
        ])
        .select(["SYMBOL", "Open Price", "Close Price", "% Change", "Avg. Volume"])
        .sort(by="% Change", descending=True)
    )

    # Convert to pandas for compatibility with st.dataframe and CSV download
    df_all_pd = per_symbol.to_pandas()
    if not df_all_pd.empty:
        df_all_pd.index = range(1, len(df_all_pd) + 1)
        df_all_pd.index.name = "Sl. No."

    df_final_pd = df_all_pd[df_all_pd["% Change"] > 0].copy()
    if not df_final_pd.empty:
        df_final_pd.index = range(1, len(df_final_pd) + 1)

    return df_final_pd, df_all_pd

# ======================
# 🧮 AGE MAP (DAILY, calendar-safe) USING PARQUET
# ======================
def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(s.strip().upper() for s in symbols))

@st.cache_data(ttl=86400)
def build_first_trade_map(
    symbols: List[str],
    selected_year: int,
    max_threshold_years: int = 5,
    force_refresh: bool = False
) -> Dict[str, datetime]:
    """
    SYMBOL -> first trading calendar date present in the Parquet (daily rows).
    Uses daily DATE with valid Open/Close.
    Cache key includes selected_year, symbol set, and horizon for correctness.
    """
    _ = (selected_year, max_threshold_years, _normalize_symbol_list(symbols), force_refresh)

    master = load_master_table_parquet()
    if not symbols:
        return {}

    cutoff_year = selected_year - max_threshold_years
    start = datetime(max(cutoff_year - 1, 1990), 1, 1).date()
    end = datetime(selected_year, 12, 31).date()

    df = (
        master
        .filter(
            pl.col("SYMBOL").is_in(symbols) &
            (pl.col("DATE") >= pl.lit(start)) &
            (pl.col("DATE") <= pl.lit(end)) &
            pl.col("Open").is_not_null() &
            pl.col("Close").is_not_null()
        )
        .select(["SYMBOL", "DATE"])
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").first().alias("FIRST_DATE"))
    )

    # Convert to dict
    first_trade = {}
    if not df.is_empty():
        for row in df.iter_rows(named=True):
            first_trade[row["SYMBOL"]] = row["FIRST_DATE"]
    return first_trade

def filter_by_age(df, year: int, age_option: str):
    if age_option == "All" or df.empty:
        return df.copy()

    threshold_map = {
        "Older than 1 year": 1,
        "Older than 2 years": 2,
        "Older than 3 years": 3,
    }
    if age_option not in threshold_map:
        return df.copy()

    N = threshold_map[age_option]
    symbols_list = df["SYMBOL"].unique().tolist()

    first_trade_map = build_first_trade_map(
        symbols=symbols_list,
        selected_year=year,
        max_threshold_years=max(5, N + 1),
        force_refresh=False
    )

    cutoff = datetime(year - N, 12, 31).date()

    valid = [s for s in symbols_list if (s in first_trade_map and first_trade_map[s] <= cutoff)]
    return df[df["SYMBOL"].isin(valid)].copy()

# ======================
# 🧹 CACHE CONTROL UI
# ======================
cc1, cc2 = st.columns([1, 1])
with cc1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared all caches. App will refresh.")

# ======================
# 🔍 FETCH BUTTON
# ======================
if symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_result, df_all = fetch_yearly_data(symbols, year)

            if not df_result.empty:
                st.session_state["fetched_data"] = df_result
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {len(df_result)} positive gainers out of {len(df_all)} fetched symbols.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data"] = None
else:
    st.stop()

# ======================
# 🎛️ REAL-TIME FILTERS
# ======================
if "fetched_data" in st.session_state and st.session_state["fetched_data"] is not None:
    df_result = st.session_state["fetched_data"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Compute bounds from the first list (use pandas DataFrame already)
    try:
        open_min_bound = int(np.floor(df_result["Open Price"].min() / 10) * 10)
        open_max_bound = int(np.ceil(df_result["Open Price"].max() / 10) * 10)
        pct_min_bound = int(np.floor(df_result["% Change"].min() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result["% Change"].max() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Initialize immutable display values once per fetched year (Open + %)
    if "open_min_init" not in st.session_state or st.session_state.get("init_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_init = open_min_bound
        st.session_state.open_max_init = open_max_bound
        st.session_state.pct_min_init = pct_min_bound
        st.session_state.pct_max_init = pct_max_bound
        st.session_state.init_year = st.session_state["fetched_year"]

    # Initialize editable inputs (used for filtering) for current year
    if "open_min_val" not in st.session_state or st.session_state.get("inputs_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_val = open_min_bound
        st.session_state.open_max_val = open_max_bound
        st.session_state.pct_min_val = pct_min_bound
        st.session_state.pct_max_val = pct_max_bound
        st.session_state.inputs_year = st.session_state["fetched_year"]

    # ---- Open Price Range (fixed label, editable inputs) ----
    st.markdown(
        f"#### Open Price Range (₹) — Min ({st.session_state.open_min_init}) · Max ({st.session_state.open_max_init})"
    )
    cmin, cmax = st.columns(2)
    with cmin:
        st.session_state.open_min_val = st.number_input(
            "Min", min_value=open_min_bound, max_value=open_max_bound,
            value=max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val)),
            step=1, key="open_min_input"
        )
    with cmax:
        st.session_state.open_max_val = st.number_input(
            "Max", min_value=open_min_bound, max_value=open_max_bound,
            value=min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val)),
            step=1, key="open_max_input"
        )
    st.session_state.open_min_val = max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val))
    st.session_state.open_max_val = min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val))
    open_range = (st.session_state.open_min_val, st.session_state.open_max_val)

    # ---- % Change Range (fixed label, editable inputs) ----
    st.markdown(
        f"#### % Change Range — Min ({st.session_state.pct_min_init}) · Max ({st.session_state.pct_max_init})"
    )
    cpmin, cpmax = st.columns(2)
    with cpmin:
        st.session_state.pct_min_val = st.number_input(
            "Min", min_value=pct_min_bound, max_value=pct_max_bound,
            value=max(pct_min_bound, min(st.session_state.pct_min_val, st.session_state.pct_max_val)),
            step=1, key="pct_min_input"
        )
    with cpmax:
        st.session_state.pct_max_val = st.number_input(
            "Max", min_value=pct_min_bound, max_value=pct_max_bound,
            value=min(pct_max_bound, max(st.session_state.pct_max_val, st.session_state.pct_min_val)),
            step=1, key="pct_max_input"
        )
    st.session_state.pct_min_val = max(pct_min_bound, min(st.session_state.pct_min_val, st.session_state.pct_max_val))
    st.session_state.pct_max_val = min(pct_max_bound, max(st.session_state.pct_max_val, st.session_state.pct_min_val))
    pct_range = (st.session_state.pct_min_val, st.session_state.pct_max_val)

    # Other filters
    vol_filter = st.selectbox("Filter by Avg. Volume", options=[
        "All", "More than 100K", "More than 150K", "More than 200K",
        "More than 250K", "More than 300K", "More than 350K",
        "More than 400K", "More than 500K"
    ], key="vol_select")

    age_filter = st.selectbox("Company Older Than", options=[
        "All", "Older than 1 year", "Older than 2 years", "Older than 3 years"
    ], key="age_select")

    # Apply filters (df_result is pandas; keep logic unchanged)
    filtered_df = df_result[
        (df_result["Open Price"] >= open_range[0])
        & (df_result["Open Price"] <= open_range[1])
        & (df_result["% Change"] >= pct_range[0])
        & (df_result["% Change"] <= pct_range[1])
    ].copy()

    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_df = filtered_df[filtered_df["Avg. Volume"] > vol_threshold]

    if age_filter != "All":
        with st.spinner(f"Filtering companies {age_filter.lower()}..."):
            filtered_df = filter_by_age(filtered_df, st.session_state["fetched_year"], age_filter)

    # Reindex to keep Sl. No. dynamic
    filtered_df = filtered_df.reset_index(drop=True)
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
st.caption("Data Source: Hugging Face Parquet (Chiron-S/NSE_Stocks_Data) | Built by Shubham Kishor | Results cached for 1 hour.")
