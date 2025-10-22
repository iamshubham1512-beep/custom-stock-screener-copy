import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import os

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
# ⚡ FETCH DATA FUNCTION
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year):
    """Fetch monthly OHLCV data and calculate yearly change"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_folder = "cache"
    os.makedirs(cache_folder, exist_ok=True)
    cache_filename = os.path.join(cache_folder, f"Fetched_Symbols_{year}.csv")

    # 🔁 Check Local Cache
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

            row = {
                "SYMBOL": sym,
                "Open Price": round(open_price, 2),
                "Close Price": round(close_price, 2),
                "% Change": pct_change,
                "Avg. Volume": avg_volume,
            }

            collected_all.append(row)

        except Exception:
            continue

    df_all = pd.DataFrame(collected_all).sort_values(by="% Change", ascending=False).reset_index(drop=True)
    df_all.index += 1
    df_all.index.name = "Sl. No."

    df_final = df_all[df_all["% Change"] > 0].copy()
    df_final.index = range(1, len(df_final) + 1)

    # 💾 Save Cached Copy
    if not df_all.empty:
        df_all.to_csv(cache_filename, index=True)

    return df_final, df_all


# ======================
# 🔍 FETCH BUTTON
# ======================
if symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_result, df_all = fetch_yearly_data(symbols, year)

            if not df_result.empty:
                st.session_state["fetched_data"] = df_result  # ✅ Store in session
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {len(df_result)} positive gainers out of {len(df_all)} fetched symbols.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data"] = None
else:
    st.stop()


# ======================
# 🎛️ REAL-TIME FILTERS (Persistent)
# ======================
if "fetched_data" in st.session_state and st.session_state["fetched_data"] is not None:
    df_result = st.session_state["fetched_data"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Safely calculate slider limits
    try:
        open_min, open_max = int(df_result["Open Price"].min()), int(df_result["Open Price"].max())
        pct_min, pct_max = float(df_result["% Change"].min()), float(df_result["% Change"].max())
    except Exception:
        open_min, open_max, pct_min, pct_max = 0, 1000, -100, 100

    # Open Price Range Filter
    open_range = st.slider(
        "Open Price Range (₹)",
        min_value=open_min,
        max_value=open_max,
        value=(open_min, open_max),
        step=10,
        key="open_slider"
    )

    # % Change Range Filter
    pct_range = st.slider(
        "% Change Range",
        min_value=pct_min,
        max_value=pct_max,
        value=(pct_min, pct_max),
        step=1.0,
        key="pct_slider"
    )

    # Avg. Volume Filter
    vol_filter = st.selectbox(
        "Filter by Avg. Volume",
        options=["All", "More than 100K", "More than 150K", "More than 200K"],
        key="vol_select"
    )

    # Apply filters dynamically
    filtered_df = df_result[
        (df_result["Open Price"] >= open_range[0])
        & (df_result["Open Price"] <= open_range[1])
        & (df_result["% Change"] >= pct_range[0])
        & (df_result["% Change"] <= pct_range[1])
    ].copy()

    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_df = filtered_df[filtered_df["Avg. Volume"] > vol_threshold]

    # Display filtered data
    st.write(f"📈 Showing {len(filtered_df)} results after filters:")
    st.dataframe(filtered_df, use_container_width=True)

    # Download filtered data
    csv = filtered_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv,
        file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv",
        mime="text/csv",
    )

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Yahoo Finance | Built by Shubham Kishor | Results cached for 1 hour.")
