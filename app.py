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

# Hide +/- steppers on all number_input widgets
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
def strip_indices(symbols: List[str]) -> List[str]:
    return [s for s in symbols if not str(s).startswith("^")]

def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(s).strip().upper() for s in symbols if s is not None))

# ======================
# 📦 DATA SOURCE: Hugging Face Parquet (download to local cache) + Polars
# ======================
HF_PARQUET_URL = "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/All_Stocks_Master.parquet"
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

@st.cache_data(ttl=3600)
def load_master_table_parquet() -> pl.DataFrame:
    local_path = _download_with_cache(HF_PARQUET_URL)
    df = pl.read_parquet(str(local_path))

    # Normalize column names
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
        raise ValueError(f"Parquet is missing required columns: {missing}")

    # SYMBOL: cast then strip safely (no chained .str on non-strings)
    df = df.with_columns(pl.col("SYMBOL").cast(pl.Utf8).alias("SYMBOL"))
    df = df.with_columns(
        pl.when(pl.col("SYMBOL").is_not_null())
          .then(pl.col("SYMBOL").str.strip_chars())
          .otherwise(pl.lit(None, dtype=pl.Utf8))
          .alias("SYMBOL")
    )

    # DATE: coerce using concrete dtype from schema (avoid expr dtype comparisons)
    date_dtype = df.schema.get("DATE")
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col("DATE").cast(pl.Date).alias("DATE"))
    elif date_dtype == pl.Datetime:
        df = df.with_columns(pl.col("DATE").cast(pl.Date).alias("DATE"))
    else:
        df = df.with_columns(
            pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, utc=False).cast(pl.Date).alias("DATE")
        )

    # Volume: ensure numeric and fill nulls
    if "Volume" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("Volume").is_null()).then(pl.lit(0)).otherwise(pl.col("Volume")).alias("Volume")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("Volume"))

    # Drop empty symbols
    df = df.filter(pl.col("SYMBOL").is_not_null() & (pl.col("SYMBOL") != ""))

    return df

@st.cache_data(ttl=3600)
def list_all_symbols() -> List[str]:
    master = load_master_table_parquet()
    syms = master.select(pl.col("SYMBOL").unique()).to_series().to_list()
    return strip_indices(syms)

# ======================
# 📅 YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2019, datetime.now().year + 1))[::-1])

# ======================
# ⚡ YEARLY METRICS (Polars)
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
        empty = pl.DataFrame(schema={"SYMBOL": pl.Utf8, "Open Price": pl.Float64, "Close Price": pl.Float64, "% Change": pl.Float64, "Avg. Volume": pl.Int64})
        return empty, empty

    per_symbol = (
        df_year
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg([
            pl.col("Open").first().alias("Open Price"),
            pl.col("Close").last().alias("Close Price"),
            pl.col("Volume").mean().cast(pl.Int64).fill_null(0).alias("Avg. Volume"),
        ])
        .with_columns([
            ((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100.0).alias("% Change")
        ])
        .with_columns(pl.col("% Change").round(2))
        .select(["SYMBOL", "Open Price", "Close Price", "% Change", "Avg. Volume"])
        .sort(by="% Change", descending=True)
    )

    df_all = per_symbol
    df_final = df_all.filter(pl.col("% Change") > 0)

    return df_final, df_all

# ======================
# 🧮 AGE MAP (Polars, daily, calendar-safe)
# ======================
@st.cache_data(ttl=86400)
def build_first_trade_map(
    symbols: List[str],
    selected_year: int,
    max_threshold_years: int = 5,
    force_refresh: bool = False
) -> Dict[str, datetime]:
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
            (pl.col("DATE") >= start) &
            (pl.col("DATE") <= end) &
            pl.col("Open").is_not_null() &
            pl.col("Close").is_not_null()
        )
        .select(["SYMBOL", "DATE"])
        .sort(["SYMBOL", "DATE"])
        .group_by("SYMBOL", maintain_order=True)
        .agg(pl.col("DATE").first().alias("FIRST_DATE"))
    )

    first_trade = {}
    if not df.is_empty():
        for row in df.iter_rows(named=True):
            first_trade[row["SYMBOL"]] = row["FIRST_DATE"]
    return first_trade

def filter_by_age_pl(df_pl: pl.DataFrame, year: int, age_option: str) -> pl.DataFrame:
    if df_pl.is_empty() or age_option == "All":
        return df_pl

    threshold_map = {
        "Older than 1 year": 1,
        "Older than 2 years": 2,
        "Older than 3 years": 3,
    }
    if age_option not in threshold_map:
        return df_pl

    N = threshold_map[age_option]
    symbols_list = df_pl.select(pl.col("SYMBOL").unique()).to_series().to_list()

    first_trade_map = build_first_trade_map(
        symbols=symbols_list,
        selected_year=year,
        max_threshold_years=max(5, N + 1),
        force_refresh=False
    )

    cutoff = datetime(year - N, 12, 31).date()
    valid = {s for s, d in first_trade_map.items() if d <= cutoff}

    return df_pl.filter(pl.col("SYMBOL").is_in(list(valid)))

# ======================
# 🧹 CACHE CONTROL UI
# ======================
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared all caches. App will refresh.")

# ======================
# 🔍 FETCH BUTTON
# ======================
all_symbols = list_all_symbols()

if all_symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_final_pl, df_all_pl = fetch_yearly_data(all_symbols, year)

            if not df_final_pl.is_empty():
                st.session_state["fetched_data_pl"] = df_final_pl
                st.session_state["fetched_all_pl"] = df_all_pl
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {df_final_pl.height} positive gainers out of {df_all_pl.height} fetched symbols.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data_pl"] = None
else:
    st.stop()

# ======================
# 🎛️ REAL-TIME FILTERS
# ======================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result_pl = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Compute bounds (Polars -> Python scalars)
    try:
        open_min_bound = int(np.floor(df_result_pl.select(pl.min("Open Price")).item() / 10) * 10)
        open_max_bound = int(np.ceil(df_result_pl.select(pl.max("Open Price")).item() / 10) * 10)
        pct_min_bound = int(np.floor(df_result_pl.select(pl.min("% Change")).item() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result_pl.select(pl.max("% Change")).item() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Immutable initial labels
    if "open_min_init" not in st.session_state or st.session_state.get("init_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_init = open_min_bound
        st.session_state.open_max_init = open_max_bound
        st.session_state.pct_min_init = pct_min_bound
        st.session_state.pct_max_init = pct_max_bound
        st.session_state.init_year = st.session_state["fetched_year"]

    # Editable inputs
    if "open_min_val" not in st.session_state or st.session_state.get("inputs_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_val = open_min_bound
        st.session_state.open_max_val = open_max_bound
        st.session_state.pct_min_val = pct_min_bound
        st.session_state.pct_max_val = pct_max_bound
        st.session_state.inputs_year = st.session_state["fetched_year"]

    # Open Price inputs
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

    # % Change inputs
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

    # Apply filters in Polars
    filtered_pl = df_result_pl.filter(
        (pl.col("Open Price") >= open_range[0]) &
        (pl.col("Open Price") <= open_range[1]) &
        (pl.col("% Change") >= pct_range[0]) &
        (pl.col("% Change") <= pct_range[1])
    )

    # Vol filter
    vol_filter = st.selectbox("Filter by Avg. Volume", options=[
        "All", "More than 100K", "More than 150K", "More than 200K",
        "More than 250K", "More than 300K", "More than 350K",
        "More than 400K", "More than 500K"
    ], key="vol_select")
    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_pl = filtered_pl.filter(pl.col("Avg. Volume") > vol_threshold)

    # Age filter (Polars)
    age_filter = st.selectbox("Company Older Than", options=[
        "All", "Older than 1 year", "Older than 2 years", "Older than 3 years"
    ], key="age_select")
    if age_filter != "All":
        with st.spinner(f"Filtering companies {age_filter.lower()}..."):
            filtered_pl = filter_by_age_pl(filtered_pl, st.session_state["fetched_year"], age_option=age_filter)

    # Display and download
    filtered_pd = filtered_pl.to_pandas()
    filtered_pd = filtered_pd.reset_index(drop=True)
    filtered_pd.index = range(1, len(filtered_pd) + 1)
    filtered_pd.index.name = "Sl. No."

    st.write(f"📈 Showing {len(filtered_pd)} results after filters:")
    st.dataframe(filtered_pd, use_container_width=True)

    csv = filtered_pd.to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv,
                       file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv",
                       mime="text/csv")

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Hugging Face Parquet (Chiron-S/NSE_Stocks_Data) | Built by Shubham Kishor | Results cached for 1 hour.")
