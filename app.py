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
    # Avoid indices like ^NSEI which behave differently
    return [s for s in symbols if not str(s).startswith("^")]

# ======================
# 📂 LOAD STOCK SYMBOLS
# ======================
@st.cache_data
def load_stock_list(file_path: str, file_mtime_key: float):
    """Load stock symbols from CSV"""
    df = pd.read_csv(file_path)
    if "SYMBOL" not in df.columns:
        raise ValueError("The file must contain a column named 'SYMBOL' (all caps).")
    syms = (
        df["SYMBOL"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
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
    """Fetch yearly OHLCV data using a single batched download for speed and reliability."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_folder = "cache"
    os.makedirs(cache_folder, exist_ok=True)
    cache_filename = os.path.join(cache_folder, f"Fetched_Symbols_{year}.csv")

    # Try local CSV cache first
    if os.path.exists(cache_filename):
        try:
            cached_df = pd.read_csv(cache_filename, index_col=0)
            df_final = cached_df[cached_df["% Change"] > 0].sort_values(by="% Change", ascending=False)
            df_final.index = range(1, len(df_final) + 1)
            return df_final, cached_df
        except Exception:
            pass

    tickers = [ensure_ns(s) for s in symbols]
    # Multi-ticker monthly download
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
    # yfinance multi-ticker download yields a MultiIndex on columns: level 0 = ticker
    for sym in symbols:
        t = ensure_ns(sym)
        # Guard for missing ticker columns
        if t not in getattr(data.columns, "levels", [data.columns])[0]:
            continue

        try:
            df_t = data[t].dropna(how="all")
        except Exception:
            # Some versions require .xs; fallback
            try:
                df_t = data.xs(t, axis=1, level=0, drop_level=False).droplevel(0, axis=1).dropna(how="all")
            except Exception:
                continue

        # Restrict to calendar year window to avoid off-by-one monthly edges
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
@st.cache_data(ttl=86400)
def build_history_start_year(symbols, lookback_years: int = 6):
    """Build a map of symbol -> first available year using one batched daily download."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.DateOffset(years=lookback_years)
    tickers = [ensure_ns(s) for s in symbols]

    hist = yf.download(
        tickers=tickers,
        start=start.tz_localize(None),
        end=end.tz_localize(None),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    first_year_map = {}
    for sym in symbols:
        t = ensure_ns(sym)
        # Check ticker presence in columns
        try:
            if t in getattr(hist.columns, "levels", [hist.columns])[0]:
                try:
                    df_t = hist[t].dropna(how="all")
                except Exception:
                    df_t = hist.xs(t, axis=1, level=0, drop_level=False).droplevel(0, axis=1).dropna(how="all")
                if not df_t.empty:
                    first_year_map[sym] = int(df_t.index.min().year)
        except Exception:
            continue
    return first_year_map

def filter_by_age(df: pd.DataFrame, year: int, age_option: str) -> pd.DataFrame:
    if age_option == "All" or df.empty:
        return df.copy()
    symbols_unique = df["SYMBOL"].unique().tolist()
    first_year_map = build_history_start_year(symbols_unique, lookback_years=6)
    threshold = {"Older than 1 year": 1, "Older than 2 years": 2, "Older than 3 years": 3}[age_option]

    valid = []
    for sym in symbols_unique:
        if sym in first_year_map:
            company_age = year - first_year_map[sym]
            if company_age >= threshold:
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

    # Bounds for Open and % Change
    try:
        open_min_bound = int(np.floor(df_result["Open Price"].min() / 10) * 10)
        open_max_bound = int(np.ceil(df_result["Open Price"].max() / 10) * 10)
        pct_min = int(np.floor(df_result["% Change"].min() / 10) * 10)
        pct_max = int(np.ceil(df_result["% Change"].max() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min, pct_max = 0, 1000, -100, 100

    # ---- New, user-friendly Open Price controls (synced number inputs + slider) ----
    # Initialize session defaults
    if "open_min_val" not in st.session_state:
        st.session_state.open_min_val = open_min_bound
    if "open_max_val" not in st.session_state:
        st.session_state.open_max_val = open_max_bound

    st.markdown("#### Open Price Range (₹)")
    c1, c2 = st.columns([1, 2])

    with c1:
        n1, n2 = st.columns(2)
        st.session_state.open_min_val = n1.number_input(
            "Min", min_value=open_min_bound, max_value=open_max_bound,
            value=max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val)),
            step=1, key="open_min_input"
        )
        st.session_state.open_max_val = n2.number_input(
            "Max", min_value=open_min_bound, max_value=open_max_bound,
            value=min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val)),
            step=1, key="open_max_input"
        )

    with c2:
        slider_min, slider_max = st.slider(
            "Drag to adjust",
            min_value=open_min_bound, max_value=open_max_bound,
            value=(
                max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val)),
                min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val)),
            ),
            step=10, key="open_slider_synced"
        )

    # Two-way sync resolution
    if (slider_min, slider_max) != (st.session_state.open_min_val, st.session_state.open_max_val):
        st.session_state.open_min_val, st.session_state.open_max_val = slider_min, slider_max
    else:
        st.session_state.open_min_val = max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val))
        st.session_state.open_max_val = min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val))

    open_range = (st.session_state.open_min_val, st.session_state.open_max_val)
    # ---- End of new Open Price controls ----

    # Keep existing % Change slider as is
    pct_range = st.slider("% Change Range", min_value=pct_min, max_value=pct_max,
                          value=(pct_min, pct_max), step=10, key="pct_slider")

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
