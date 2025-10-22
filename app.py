import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

st.set_page_config(page_title="Custom Stock Screener", layout="wide")
st.title("📈 Custom Stock Screener — Minimal (yfinance)")

# Sidebar controls
with st.sidebar:
    st.header("Screener settings")
    tickers_input = st.text_area("Tickers (comma-separated)", "RELIANCE.NS, TCS.NS, HDFCBANK.NS")
    interval = st.selectbox("Interval", ["5m", "15m", "30m", "1h", "1d"], index=0)
    period = st.selectbox("Period", ["1d", "5d", "7d", "30d"], index=0)
    price_min = st.number_input("Price min (₹)", value=0.0, step=1.0)
    price_max = st.number_input("Price max (₹)", value=100000.0, step=1.0)
    pct_change_min = st.number_input("Min % Change", value=-100.0, step=0.1)
    pct_change_max = st.number_input("Max % Change", value=100.0, step=0.1)
    rsi_threshold = st.slider("RSI threshold (show RSI < value)", 0, 100, 30)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

@st.cache_data(ttl=60)
def fetch_multi(tickers, period, interval):
    try:
        df = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df_all = fetch_multi(tickers, period, interval)
if df_all is None or df_all.empty:
    st.info("No data yet — try different interval/period or fewer tickers.")
    st.stop()

results = []
for t in tickers:
    try:
        sub = df_all.copy() if len(tickers) == 1 else df_all[t].copy()
        last = sub.iloc[-1].copy()
        open_p = float(last['Open'])
        close_p = float(last['Close'])
        vol = int(last.get('Volume', 0) or 0)
        pct = (close_p - open_p) / open_p * 100 if open_p != 0 else 0.0
        close_series = sub['Close'].dropna()
        rsi = ta.rsi(close_series, length=14).iloc[-1] if len(close_series) >= 14 else np.nan
        results.append({
            "Ticker": t,
            "Close": round(close_p, 2),
            "Open": round(open_p, 2),
            "Volume": vol,
            "% Change": round(pct, 2),
            "RSI": round(float(rsi) if not pd.isna(rsi) else np.nan, 2)
        })
    except Exception as e:
        results.append({"Ticker": t, "Close": None, "Open": None, "Volume": None, "% Change": None, "RSI": None})

df_results = pd.DataFrame(results).set_index("Ticker")

filtered = df_results[
    (df_results["Close"].between(price_min, price_max)) &
    (df_results["% Change"].between(pct_change_min, pct_change_max)) &
    (df_results["RSI"].fillna(999) < rsi_threshold)
]

st.subheader("Screener results")
st.write(f"Interval: {interval} • Period: {period} • Showing {len(filtered)} / {len(df_results)}")
st.dataframe(filtered)

st.subheader("Ticker chart")
selected = st.selectbox("Select ticker to plot", tickers)
data_plot = (df_all[selected] if len(tickers) > 1 else df_all).copy()
st.line_chart(data_plot['Close'])
