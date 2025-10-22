import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yearly Top Gainers", layout="wide")

# --- HEADER ---
st.title("📈 Yearly Top Gainers")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Year Open Price Range
price_min = st.sidebar.slider("Year Open Price Range (₹)", min_value=0, max_value=5000, value=(0, 1000), step=50)

# % Change Filter
change_min, change_max = st.sidebar.slider("% Change Range", min_value=-100, max_value=500, value=(-10, 100), step=5)

# Avg Volume Filter
volume_filter = st.sidebar.selectbox(
    "Avg. Volume (more than)",
    options=["0", "100K", "500K", "1M", "5M", "10M"],
    index=0
)

# Convert selected volume option to numeric
volume_threshold = {
    "0": 0,
    "100K": 1e5,
    "500K": 5e5,
    "1M": 1e6,
    "5M": 5e6,
    "10M": 1e7
}[volume_filter]

# --- YEAR SELECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    year = st.selectbox("Select Year", [2020, 2021, 2022, 2023, 2024, 2025])
with col2:
    st.button("Saved Result")

# --- LOAD SYMBOLS ---
# You can replace this with your CSV of NSE/BSE stocks
symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"]

# --- DATA FETCH FUNCTION ---
@st.cache_data
def fetch_yearly_data(symbols, year):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    data = []
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if not df.empty:
                open_price = df["Open"].iloc[0]
                close_price = df["Close"].iloc[-1]
                pct_change = ((close_price - open_price) / open_price) * 100
                avg_volume = df["Volume"].mean()
                data.append([sym.replace(".NS", ""), open_price, close_price, pct_change, avg_volume])
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    df = pd.DataFrame(data, columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"])
    return df

# --- FETCH DATA BUTTON ---
if st.button("🔍 Fetch Data"):
    with st.spinner("Fetching stock data..."):
        df = fetch_yearly_data(symbols, year)

        # --- APPLY FILTERS ---
        filtered_df = df[
            (df["Open Price"].between(price_min[0], price_min[1])) &
            (df["% Change"].between(change_min, change_max)) &
            (df["Avg. Volume"] > volume_threshold)
        ].reset_index(drop=True)

        # Sort by % Change
        filtered_df = filtered_df.sort_values(by="% Change", ascending=False).reset_index(drop=True)
        filtered_df.index += 1  # Add serial number

        st.markdown("### 📊 Number of Results")
        st.dataframe(filtered_df, use_container_width=True)

        # --- EXPORT CSV ---
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Result as CSV",
            data=csv,
            file_name=f"Top_Gainers_{year}.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Select a year, adjust filters, then click **Fetch Data** to view results.")
