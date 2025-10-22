import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

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
    """
    Loads the stock symbols from the uploaded CSV file.
    Expected file name: 'NSE Stocks List.csv' with a column named 'Symbol'.
    Returns a list of stock symbols.
    """
    try:
        df = pd.read_csv("NSE Stocks List.csv")
        if "Symbol" not in df.columns:
            st.error("❌ The file must contain a column named 'Symbol'.")
            return []
        return df["Symbol"].dropna().unique().tolist()
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
# 📈 FETCH YEARLY DATA
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year):
    """
    Fetches stock data for each symbol for the selected year using yfinance.
    Returns a DataFrame with Open Price, Close Price, % Change, and Avg. Volume.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    all_data = []

    for sym in symbols:
        try:
            df = yf.download(f"{sym}.NS", start=start_date, end=end_date, progress=False)
            if df.empty:
                continue

            open_price = df["Open"].iloc[0]
            close_price = df["Close"].iloc[-1]
            pct_change = round(((close_price - open_price) / open_price) * 100, 2)
            avg_volume = int(df["Volume"].mean())

            # Only include positive gainers
            if pct_change > 0:
                all_data.append([sym, round(open_price, 2), round(close_price, 2), pct_change, avg_volume])

        except Exception as e:
            # Handle common ticker or connection errors
            print(f"Error fetching {sym}: {e}")
            continue

    # Convert collected data into a DataFrame
    if all_data:
        df_final = pd.DataFrame(
            all_data, columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"]
        )
        df_final.sort_values(by="% Change", ascending=False, inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        df_final.index += 1  # Sl. No. starts from 1
        df_final.index.name = "Sl. No."
    else:
        df_final = pd.DataFrame(columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"])

    return df_final

# ======================
# 🔍 FETCH BUTTON
# ======================
if symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait."):
            df_result = fetch_yearly_data(symbols, year)

            if not df_result.empty:
                st.success(f"✅ Data fetched successfully for {len(df_result)} stocks.")
                st.dataframe(df_result, use_container_width=True)

                # Allow user to download CSV
                csv = df_result.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"Top_Gainers_{year}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ No positive gainers found for the selected year.")
    else:
        st.info("👆 Select a year and click **Fetch Yearly Data** to start.")
else:
    st.stop()

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Yahoo Finance | Built by Shubham Kishor | Updates once per hour via cache.")
