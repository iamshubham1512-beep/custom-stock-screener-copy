import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ---------- Streamlit App Configuration ----------
st.set_page_config(page_title="Custom Stock Screener", layout="wide")
st.title("📊 Real-Time Custom Stock Screener (NSE)")

# ---------- Upload Stock List ----------
uploaded_file = st.file_uploader("📁 Upload your 'NSE Stocks List.csv' file", type=["csv"])

# ---------- Year Selection Dropdown ----------
year = st.selectbox("📅 Select Year to Fetch Data", options=list(range(2018, datetime.now().year + 1))[::-1])

# ---------- Stop Fetching Flag ----------
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False

def stop_fetching():
    st.session_state.stop_flag = True
    st.warning("⛔ Fetching stopped by user. You can restart by pressing 'Start Fetching Data' again.")

st.button("🛑 Stop Fetching Data", on_click=stop_fetching)

# ---------- Main Logic ----------
if uploaded_file is not None:
    try:
        df_symbols = pd.read_csv(uploaded_file)

        # --- Validate Symbol Column ---
        if "SYMBOL" not in df_symbols.columns:
            st.error("❌ CSV must have a column named 'SYMBOL'. Please check your file and re-upload.")
        else:
            if st.button("🚀 Start Fetching Data"):
                st.session_state.stop_flag = False
                results = []
                progress = st.progress(0)
                total = len(df_symbols)

                # --- Loop through each symbol ---
                for i, symbol in enumerate(df_symbols["SYMBOL"]):
                    if st.session_state.stop_flag:
                        break  # Stop operation if stop button pressed

                    try:
                        ticker = yf.Ticker(f"{symbol}.NS")
                        data = ticker.history(start=f"{year}-01-01", end=f"{year}-12-31")

                        # Skip empty data
                        if data.empty:
                            continue

                        # Extract key data points
                        open_price = data["Open"].iloc[0]
                        close_price = data["Close"].iloc[-1]
                        pct_change = ((close_price - open_price) / open_price) * 100
                        avg_volume = data["Volume"].mean()

                        # Keep only positive % change
                        if pct_change > 0:
                            results.append({
                                "Sl. No.": len(results) + 1,
                                "SYMBOL": symbol,
                                "Open Price": round(open_price, 2),
                                "Close Price": round(close_price, 2),
                                "% Change": round(pct_change, 2),
                                "Avg. Yearly Volume": int(avg_volume)
                            })

                    except Exception as e:
                        st.write(f"⚠️ Error fetching {symbol}: {str(e)}")
                        continue

                    # Update progress bar
                    progress.progress((i + 1) / total)

                # --- Display Results ---
                if results:
                    result_df = pd.DataFrame(results)
                    st.success(f"✅ Data fetched successfully for {len(result_df)} positive stocks in {year}.")
                    st.dataframe(result_df)

                    # --- Download Option ---
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Filtered Data as CSV",
                        data=csv,
                        file_name=f"NSE_Positive_{year}.csv",
                        mime='text/csv',
                    )
                else:
                    st.warning("⚠️ No data found with positive % change for this year.")

    except Exception as e:
        st.error(f"❌ Failed to process file: {str(e)}")

else:
    st.info("📂 Please upload your NSE Stocks List CSV file to begin.")
