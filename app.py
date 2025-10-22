import streamlit as st
import pandas as pd
import yfinance as yf
import time

# --- PHASE 2: LOAD STOCK LIST ---
@st.cache_data
def load_stock_list():
    try:
        df = pd.read_csv("NSE Stocks List.csv")
        symbols = df["Symbol"].dropna().unique().tolist()
        return symbols
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        return []

# --- PHASE 3: FETCH STOCK DATA ---
@st.cache_data(ttl=300)
def fetch_stock_data(symbols):
    data = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym + ".NS")
            hist = ticker.history(period="1d")
            if not hist.empty:
                last_row = hist.iloc[-1]
                data.append({
                    "Symbol": sym,
                    "Last Price": round(last_row["Close"], 2),
                    "Open": round(last_row["Open"], 2),
                    "High": round(last_row["High"], 2),
                    "Low": round(last_row["Low"], 2),
                    "Volume": int(last_row["Volume"])
                })
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    return pd.DataFrame(data)

# --- PHASE 4: STREAMLIT UI ---
st.set_page_config(page_title="Real-Time NSE Screener", layout="wide")

st.title("📈 Real-Time Custom Stock Screener (NSE)")
st.write("Built by Shubham Kishor — Live Data via Yahoo Finance")

# Load stock list
symbols = load_stock_list()
if not symbols:
    st.warning("⚠️ Upload or ensure the file 'NSE Stocks List.csv' exists in the repo root.")
else:
    st.sidebar.header("Filter Settings")
    selected_symbols = st.sidebar.multiselect("Choose Stocks", symbols[:50], default=symbols[:5])
    min_price = st.sidebar.number_input("Min Price", 0, 10000, 50)
    max_price = st.sidebar.number_input("Max Price", 0, 10000, 2000)

    if st.button("Fetch Live Data"):
        with st.spinner("Fetching latest prices..."):
            df = fetch_stock_data(selected_symbols)
            if not df.empty:
                filtered = df[(df["Last Price"] >= min_price) & (df["Last Price"] <= max_price)]
                st.dataframe(filtered, use_container_width=True)
            else:
                st.error("No data fetched. Try again in a few seconds.")

st.caption("Data refreshes every 5 minutes. Free API limits may cause delays.")
