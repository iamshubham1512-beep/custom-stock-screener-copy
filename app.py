import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")

st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year and apply filters to explore top-performing NSE stocks based on price change and volume.")

# ======================
# 📂 CACHING & DATA FETCHING
# ======================
def fetch_stock_data(symbol, year):
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        data = yf.download(symbol + ".NS", start=start_date, end=end_date, progress=False)
        if not data.empty:
            open_price = data.iloc[0]["Open"]
            close_price = data.iloc[-1]["Close"]
            pct_change = ((close_price - open_price) / open_price) * 100
            avg_volume = data["Volume"].mean()
            return symbol, open_price, close_price, pct_change, avg_volume
    except Exception:
        return None

@st.cache_data
def get_cached_data(year):
    cache_filename = f"Fetched_Symbols_{year}.csv"

    if os.path.exists(cache_filename):
        df = pd.read_csv(cache_filename)
    else:
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "HINDUNILVR", "BAJFINANCE", "ITC", "KOTAKBANK"
        ]
        results = []
        for sym in symbols:
            result = fetch_stock_data(sym, year)
            if result:
                results.append(result)
        df = pd.DataFrame(results, columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg Volume"])
        df.to_csv(cache_filename, index=False)
    return df

# ======================
# 🧮 MAIN APP LOGIC
# ======================
year = st.selectbox("Select Year", list(range(2018, datetime.now().year + 1))[::-1])
data = get_cached_data(year)

if not data.empty:
    st.subheader("📊 Apply Filters")

    # ----- FILTERS -----
    open_min = int(data["Open Price"].min() // 10 * 10)
    open_max = int(data["Open Price"].max() // 10 * 10)
    open_range = st.slider(
        "Open Price Range (₹)",
        min_value=open_min,
        max_value=open_max,
        value=(open_min, open_max),
        step=10
    )

    pct_min = int(data["% Change"].min() // 10 * 10)
    pct_max = int(data["% Change"].max() // 10 * 10)
    pct_range = st.slider(
        "% Change Range (%)",
        min_value=pct_min,
        max_value=pct_max,
        value=(pct_min, pct_max),
        step=10
    )

    volume_options = [100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
    selected_volume = st.selectbox("Avg. Volume Filter", [f"More than {int(v/1000)}K" for v in volume_options])
    min_volume = int(selected_volume.split()[2].replace("K", "")) * 1000

    # ----- APPLY FILTERS -----
    filtered_data = data[
        (data["Open Price"].between(open_range[0], open_range[1])) &
        (data["% Change"].between(pct_range[0], pct_range[1])) &
        (data["Avg Volume"] > min_volume)
    ]

    st.subheader("📈 Filtered Results")
    st.dataframe(
        filtered_data.sort_values(by="% Change", ascending=False),
        use_container_width=True
    )
else:
    st.warning("⚠️ No data found for the selected year.")
