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
    """
    Loads the stock symbols from 'NSE Stocks List.csv' expected in repo root.
    Required column: 'SYMBOL' (uppercase).
    Returns a list of symbols (strings).
    """
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
# ⚡ FASTER YEARLY DATA FETCHING (Optimized)
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year, show_progress=False):
    """
    Optimized version:
    - Fetches monthly OHLCV data instead of daily (interval='1mo') → ~95% faster
    - Uses cached CSV if available to avoid refetching
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_filename = f"Fetched_Symbols_{year}.csv"

    # ======================
    # 🔁 Check Local Cache
    # ======================
    if os.path.exists(cache_filename):
        try:
            cached_df = pd.read_csv(cache_filename, index_col=0)
            st.info(f"📦 Loaded cached data from {cache_filename}")
            # Filter only positive gainers
            df_final = cached_df[cached_df["% Change"] > 0].sort_values(by="% Change", ascending=False)
            df_final.index = range(1, len(df_final) + 1)
            return df_final, {
                "requested": len(symbols),
                "fetched_with_data": len(cached_df),
                "positive_gainers": len(df_final),
                "skipped_empty": 0,
                "failed": 0,
                "failed_details": []
            }, cached_df
        except Exception as e:
            st.warning(f"⚠️ Failed to read cached file: {e}. Fetching fresh data...")

    # ======================
    # 🧠 Fetch Fresh Data (Monthly)
    # ======================
    collected, collected_all, failed, skipped_empty = [], [], [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        ticker_symbol = sym if "." in sym else f"{sym}.NS"
        try:
            df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1mo", progress=False)

            if df.empty:
                skipped_empty.append(sym)
                continue

            open_series = df["Open"].dropna()
            close_series = df["Close"].dropna()
            vol_series = df["Volume"].dropna()

            if open_series.empty or close_series.empty:
                skipped_empty.append(sym)
                continue

            open_price = float(open_series.iloc[0])
            close_price = float(close_series.iloc[-1])
            if open_price == 0:
                skipped_empty.append(sym)
                continue

            pct_change = round(((close_price - open_price) / open_price) * 100, 2)
            avg_volume = int(vol_series.mean()) if not vol_series.empty else 0

            row = {
                "SYMBOL": sym,
                "Open Price": round(open_price, 2),
                "Close Price": round(close_price, 2),
                "% Change": pct_change,
                "Avg. Volume": avg_volume
            }

            collected_all.append(row)
            if pct_change > 0:
                collected.append(row)

        except Exception as e:
            failed.append({"symbol": sym, "error": str(e)})
            continue

    # Build DataFrames
    df_all = pd.DataFrame(collected_all).sort_values(by="% Change", ascending=False).reset_index(drop=True)
    df_all.index += 1
    df_all.index.name = "Sl. No."

    df_final = df_all[df_all["% Change"] > 0].copy()
    df_final.index = range(1, len(df_final) + 1)

    # ======================
    # 💾 Save Cached Copy
    # ======================
    if not df_all.empty:
        df_all.to_csv(cache_filename, index=True)

    stats = {
        "requested": total,
        "fetched_with_data": len(df_all),
        "positive_gainers": len(df_final),
        "skipped_empty": len(skipped_empty),
        "failed": len(failed),
        "failed_details": failed[:10]
    }

    return df_final, stats, df_all

# ======================
# 🔍 FETCH BUTTON
# ======================
if symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_result, stats, df_all = fetch_yearly_data(symbols, year)

            if not df_result.empty:
                st.success(f"✅ Found {stats['positive_gainers']} positive gainers out of {stats['requested']} requested symbols.")
                st.dataframe(df_result, use_container_width=True)

                csv = df_result.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV (positive gainers)",
                    data=csv,
                    file_name=f"Top_Gainers_{year}.csv",
                    mime="text/csv",
                )

            else:
                st.warning("⚠️ No positive gainers found. Showing fallback view.")
                st.info(
                    f"Requested: {stats['requested']} • With data: {stats['fetched_with_data']} • "
                    f"Positive gainers: {stats['positive_gainers']} • Skipped empty: {stats['skipped_empty']} • Failed: {stats['failed']}"
                )

                if not df_all.empty:
                    st.subheader("Top movers (by % Change) — fallback view")
                    st.dataframe(df_all.head(20), use_container_width=True)
                    csv_all = df_all.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download fallback data (all fetched symbols)",
                        data=csv_all,
                        file_name=f"Fetched_Symbols_{year}.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("No symbol returned any data. Verify your symbol list or year selection.")
    else:
        st.info("👆 Select a year and click **Fetch Yearly Data** to start.")
else:
    st.stop()

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Yahoo Finance | Built by Shubham Kishor | Results cached for 1 hour.")
