import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import numpy as np

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")

st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year to view top-performing NSE stocks based on yearly price change and average volume.")

# ======================
# 📂 LOAD STOCK SYMBOLS
# ======================
@st.cache_data
def load_stock_list():
    """Load stock symbols from CSV"""
    try:
        df = pd.read_csv("NSE Stocks List.csv")
        if "SYMBOL" not in df.columns:
            st.error("❌ The file must contain a column named 'SYMBOL' (all caps).")
            return []
        syms = df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
        return syms
    except FileNotFoundError:
        st.error("⚠️ File 'NSE Stocks List.csv' not found in repository root.")
        return []
    except Exception as e:
        st.error(f"⚠️ Error loading stock list: {e}")
        return []

symbols = load_stock_list()

# ======================
# 📅 YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2019, datetime.now().year + 1))[::-1])

# ======================
# ⚡ FETCH YEARLY DATA
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year):
    """Fetch yearly OHLCV data"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_folder = "cache"
    os.makedirs(cache_folder, exist_ok=True)
    cache_filename = os.path.join(cache_folder, f"Fetched_Symbols_{year}.csv")

    if os.path.exists(cache_filename):
        try:
            cached_df = pd.read_csv(cache_filename, index_col=0)
            st.info(f"📦 Loaded cached data from {cache_filename}")
            df_final = cached_df[cached_df["% Change"] > 0].sort_values(by="% Change", ascending=False)
            df_final.index = range(1, len(df_final) + 1)
            return df_final, cached_df
        except Exception as e:
            st.warning(f"⚠️ Failed to read cached file: {e}. Fetching fresh data...")

    collected_all = []
    for sym in symbols:
        ticker_symbol = sym if "." in sym else f"{sym}.NS"
        try:
            df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1mo", progress=False)
            if df.empty:
                continue

            open_series = df["Open"].dropna()
            close_series = df["Close"].dropna()
            vol_series = df["Volume"].dropna()
            if open_series.empty or close_series.empty:
                continue

            open_price = float(open_series.iloc[0])
            close_price = float(close_series.iloc[-1])
            if open_price == 0:
                continue

            pct_change = round(((close_price - open_price) / open_price) * 100, 2)
            avg_volume = int(vol_series.mean()) if not vol_series.empty else 0

            collected_all.append({
                "SYMBOL": sym,
                "Open Price": round(open_price, 2),
                "Close Price": round(close_price, 2),
                "% Change": pct_change,
                "Avg. Volume": avg_volume,
            })
        except Exception:
            continue

    df_all = pd.DataFrame(collected_all).sort_values(by="% Change", ascending=False).reset_index(drop=True)
    df_all.index += 1
    df_all.index.name = "Sl. No."

    df_final = df_all[df_all["% Change"] > 0].copy()
    df_final.index = range(1, len(df_final) + 1)

    if not df_all.empty:
        df_all.to_csv(cache_filename, index=True)

    return df_final, df_all


# ======================
# 🧮 COMPANY AGE CHECK (Simplified & Reliable)
# ======================
@st.cache_data(ttl=86400)
def build_company_age_cache(symbols, year):
    """
    Build a simple cache of company 'listing age' based on Yahoo Finance data availability.
    Checks if data exists for both start and end of the selected year.
    """
    valid_syms = []
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    for sym in symbols:
        ticker_symbol = sym if "." in sym else f"{sym}.NS"
        try:
            df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1mo", progress=False)
            # Simplified age check: company must have data spanning the selected year
            if not df.empty and df.index.min().year <= year and df.index.max().year >= year:
                valid_syms.append(sym)
        except Exception:
            continue

    df_age = pd.DataFrame(valid_syms, columns=["SYMBOL"])
    return df_age


def filter_by_age(df, year, age_option):
    """
    Filters DataFrame by simplified company age logic.
    - 'Older than 1 year': must have data before the selected year
    - 'Older than 2/3 years': must have data going back that many years
    """
    symbols = df["SYMBOL"].unique().tolist()
    df_age = build_company_age_cache(symbols, year)
    valid_syms = []

    for sym in df_age["SYMBOL"]:
        ticker_symbol = sym if "." in sym else f"{sym}.NS"
        try:
            info = yf.Ticker(ticker_symbol).history(period="5y")
            if info.empty:
                continue

            first_year = info.index.min().year
            company_age = year - first_year

            if age_option == "Older than 1 year" and company_age >= 1:
                valid_syms.append(sym)
            elif age_option == "Older than 2 years" and company_age >= 2:
                valid_syms.append(sym)
            elif age_option == "Older than 3 years" and company_age >= 3:
                valid_syms.append(sym)
            elif age_option == "All":
                valid_syms.append(sym)
        except Exception:
            continue

    return df[df["SYMBOL"].isin(valid_syms)].copy()

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

    try:
        open_min = int(np.floor(df_result["Open Price"].min() / 10) * 10)
        open_max = int(np.ceil(df_result["Open Price"].max() / 10) * 10)
        pct_min = int(np.floor(df_result["% Change"].min() / 10) * 10)
        pct_max = int(np.ceil(df_result["% Change"].max() / 10) * 10)
    except Exception:
        open_min, open_max, pct_min, pct_max = 0, 1000, -100, 100

    open_range = st.slider("Open Price Range (₹)", min_value=open_min, max_value=open_max,
                           value=(open_min, open_max), step=10, key="open_slider")

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
