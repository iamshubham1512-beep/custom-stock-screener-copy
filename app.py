import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import numpy as np

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
    return sym if "." in sym else f"{sym}.NS"

def strip_indices(symbols):
    return [s for s in symbols if not str(s).startswith("^")]

# ======================
# 📂 LOAD STOCK SYMBOLS
# ======================
@st.cache_data
def load_stock_list(file_path: str, file_mtime_key: float):
    df = pd.read_csv(file_path)
    if "SYMBOL" not in df.columns:
        raise ValueError("The file must contain a column named 'SYMBOL' (all caps).")
    syms = (
        df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
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
# ⚡ FETCH YEARLY DATA (BATCHED)
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year: int):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_folder = "cache"
    os.makedirs(cache_folder, exist_ok=True)
    cache_filename = os.path.join(cache_folder, f"Fetched_Symbols_{year}.csv")

    if os.path.exists(cache_filename):
        try:
            cached_df = pd.read_csv(cache_filename, index_col=0)
            df_final = cached_df[cached_df["% Change"] > 0].sort_values(by="% Change", ascending=False)
            df_final.index = range(1, len(df_final) + 1)
            return df_final, cached_df
        except Exception:
            pass

    tickers = [ensure_ns(s) for s in symbols]
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1mo",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    collected = []
    for sym in symbols:
        t = ensure_ns(sym)
        if t not in getattr(data.columns, "levels", [data.columns])[0]:
            continue
        try:
            df_t = data[t].dropna(how="all")
        except Exception:
            try:
                df_t = data.xs(t, axis=1, level=0, drop_level=False).droplevel(0, axis=1).dropna(how="all")
            except Exception:
                continue

        df_year = df_t[(df_t.index >= pd.Timestamp(start_date)) & (df_t.index <= pd.Timestamp(end_date))]
        if df_year.empty or not set(["Open", "Close"]).issubset(df_year.columns):
            continue

        open_series = df_year["Open"].dropna()
        close_series = df_year["Close"].dropna()
        vol_series = df_year["Volume"].dropna() if "Volume" in df_year else pd.Series(dtype="float64")
        if open_series.empty or close_series.empty:
            continue

        open_price = float(open_series.iloc[0])
        close_price = float(close_series.iloc[-1])
        if open_price == 0:
            continue

        pct_change = round(((close_price - open_price) / open_price) * 100, 2)
        avg_volume = int(vol_series.mean()) if not vol_series.empty else 0

        collected.append({
            "SYMBOL": sym,
            "Open Price": round(open_price, 2),
            "Close Price": round(close_price, 2),
            "% Change": pct_change,
            "Avg. Volume": avg_volume,
        })

    df_all = pd.DataFrame(collected).sort_values(by="% Change", ascending=False).reset_index(drop=True)
    df_all.index += 1
    df_all.index.name = "Sl. No."

    df_final = df_all[df_all["% Change"] > 0].copy()
    df_final.index = range(1, len(df_final) + 1)

    if not df_all.empty:
        try:
            df_all.to_csv(cache_filename, index=True)
        except Exception:
            pass

    return df_final, df_all

# ======================
# 🧮 AGE MAP (BATCHED, FAST)
# ======================
from typing import Dict, List, Tuple

def _normalize_symbol_list(symbols: List[str]) -> Tuple[str, ...]:
    # Stable, hashable key for caching; upper-case and strip just in case
    return tuple(sorted(s.strip().upper() for s in symbols))

@st.cache_data(ttl=86400)
def build_first_trade_map(
    symbols: List[str],
    selected_year: int,
    max_threshold_years: int = 5,
    force_refresh: bool = False
) -> Dict[str, pd.Timestamp]:
    """
    Returns a dict: SYMBOL -> first tradable daily bar date (UTC-naive Timestamp).
    - Uses 1d bars for accuracy (first available open/close).
    - Cache key is tied to (selected_year, symbols, max_threshold_years) to prevent stale reuse.
    - Ensures lookback spans enough years to validate 'Older than N years' up to max_threshold_years.
    """
    # Cache key components that Streamlit uses (function args)
    # 'force_refresh' doesn't get used in logic; it only helps bust cache when needed.
    _ = (selected_year, max_threshold_years, _normalize_symbol_list(symbols), force_refresh)

    if not symbols:
        return {}

    # Determine a lookback start so we can validate up to N years older than selected_year
    # Example: selected_year=2022, N=3 -> need data <= 2019-12-31
    cutoff_year = selected_year - max_threshold_years
    end = pd.Timestamp(year=selected_year, month=12, day=31)
    start = pd.Timestamp(year=max(cutoff_year - 1, 1990), month=1, day=1)  # go one extra year to be safe

    tickers = [ensure_ns(s) for s in symbols]

    hist = yf.download(
        tickers=tickers,
        start=start,
        end=end + pd.Timedelta(days=1),  # inclusive end
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    first_trade: Dict[str, pd.Timestamp] = {}
    # Multi-ticker returns MultiIndex columns [ticker -> fields]
    for sym in symbols:
        t = ensure_ns(sym)
        try:
            # try direct selection
            if t in getattr(hist.columns, "levels", [hist.columns])[0]:
                try:
                    df_t = hist[t]
                except Exception:
                    df_t = hist.xs(t, axis=1, level=0, drop_level=False).droplevel(0, axis=1)
            else:
                continue

            # Keep only rows that have valid open & close
            cols = df_t.columns
            if "Open" not in cols or "Close" not in cols:
                continue
            df_t = df_t.loc[df_t["Open"].notna() & df_t["Close"].notna()].copy()
            if df_t.empty:
                continue

            # The earliest valid trading date
            d0 = pd.Timestamp(df_t.index.min().date())
            first_trade[sym] = d0
        except Exception:
            continue

    return first_trade

def filter_by_age(df: pd.DataFrame, year: int, age_option: str) -> pd.DataFrame:
    """
    Robust age filter:
    - For 'Older than N years' in year Y, requires first_trade_date <= Dec 31 of (Y - N).
    - Uses daily history based first trade map.
    - Excludes symbols with first trade inside the selected year when N >= 1.
    """
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
    symbols = df["SYMBOL"].unique().tolist()

    # Build an accurate first-trade map. We pick a max_threshold_years >= N to ensure sufficient lookback.
    first_trade_map = build_first_trade_map(
        symbols=symbols,
        selected_year=year,
        max_threshold_years=max(5, N + 1),
        force_refresh=False  # set True to force recompute after big changes
    )

    # Eligibility cutoff: on or before 31-Dec-(year-N)
    cutoff = pd.Timestamp(year=year - N, month=12, day=31)

    valid = []
    for sym in symbols:
        d0 = first_trade_map.get(sym)
        # If we can't determine a date, be conservative: exclude it from "older than" buckets.
        if d0 is None:
            continue
        # Must be listed/tradable on or before cutoff
        if d0 <= cutoff:
            valid.append(sym)

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

    # Compute bounds from the first list
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

    # Apply filters
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
st.caption("Data Source: Yahoo Finance | Built by Shubham Kishor | Results cached for 1 hour.")
