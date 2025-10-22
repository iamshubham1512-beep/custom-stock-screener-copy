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
    Loads the stock symbols from 'NSE Stocks List.csv' expected in repo root.
    Required column: 'SYMBOL' (uppercase).
    Returns a list of symbols (strings).
    """
    try:
        df = pd.read_csv("NSE Stocks List.csv")
        if "SYMBOL" not in df.columns:
            st.error("❌ The file must contain a column named 'SYMBOL' (all caps).")
            return []
        # drop NA and strip whitespace
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
# 📈 FETCH YEARLY DATA
# ======================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols, year, show_progress=False):
    """
    Fetches each symbol's OHLCV for the given year using yfinance.
    Normalizes symbols (adds .NS if missing).
    Returns:
      df_final -> DataFrame of positive gainers sorted by % Change (may be empty)
      stats -> dict with counts and lists for diagnostics
      df_all -> DataFrame of all symbols with data (for fallback)
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    collected = []
    collected_all = []
    failed = []
    skipped_empty = []

    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        # normalize symbol: if user already provided suffix like '.NS', keep it
        ticker_symbol = sym if "." in sym else f"{sym}.NS"
        try:
            df = yf.download(f"{sym}.NS", start=f"{year}-01-01", end=f"{year}-12-31", interval="1mo", progress=False)
            if df is None or df.empty:
                skipped_empty.append(sym)
                continue

            # get first valid open and last valid close (skip NaNs)
            # ensure we select first non-null open and last non-null close
            open_series = df["Open"].dropna()
            close_series = df["Close"].dropna()
            vol_series = df["Volume"].dropna()

            if open_series.empty or close_series.empty:
                skipped_empty.append(sym)
                continue

            open_price = float(open_series.iloc[0])
            close_price = float(close_series.iloc[-1])

            # avoid division by zero
            if open_price == 0:
                skipped_empty.append(sym)
                continue

            pct_change = round(((close_price - open_price) / open_price) * 100, 2)
            avg_volume = int(vol_series.mean()) if not vol_series.empty else 0

            row = {
                "Symbol": sym,
                "Open Price": round(open_price, 2),
                "Close Price": round(close_price, 2),
                "% Change": pct_change,
                "Avg. Volume": avg_volume
            }

            collected_all.append(row)

            # keep only positive gainers for main result
            if pct_change > 0:
                collected.append(row)

        except Exception as e:
            failed.append({"symbol": sym, "error": str(e)})
            continue

    # Build DataFrames
    if collected:
        df_final = pd.DataFrame(collected).sort_values(by="% Change", ascending=False).reset_index(drop=True)
        df_final.index += 1
        df_final.index.name = "Sl. No."
    else:
        # empty DF with expected columns
        df_final = pd.DataFrame(columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"])

    if collected_all:
        df_all = pd.DataFrame(collected_all).sort_values(by="% Change", ascending=False).reset_index(drop=True)
        df_all.index += 1
        df_all.index.name = "Sl. No."
    else:
        df_all = pd.DataFrame(columns=["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"])

    stats = {
        "requested": total,
        "fetched_with_data": len(collected_all),
        "positive_gainers": len(collected),
        "skipped_empty": len(skipped_empty),
        "failed": len(failed),
        "failed_details": failed[:10]  # show up to first 10 for diagnostics
    }

    return df_final, stats, df_all

# ======================
# 🔍 FETCH BUTTON
# ======================
if symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait."):
            df_result, stats, df_all = fetch_yearly_data(symbols, year)

            # If we have positive gainers -> show them
            if not df_result.empty:
                st.success(f"✅ Found {stats['positive_gainers']} positive gainers out of {stats['requested']} requested symbols.")
                st.dataframe(df_result, use_container_width=True)

                # Download CSV (index contains Sl. No.)
                csv = df_result.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV (positive gainers)",
                    data=csv,
                    file_name=f"Top_Gainers_{year}.csv",
                    mime="text/csv",
                )

            else:
                # No positive gainers. Provide helpful feedback and fallback table.
                st.warning(
                    "⚠️ No positive gainers found for the selected year."
                    " Showing diagnostic counts and the top movers (may be negative)."
                )

                # Show diagnostic stats
                st.info(
                    f"Requested: {stats['requested']} • With data: {stats['fetched_with_data']} • "
                    f"Positive gainers: {stats['positive_gainers']} • Skipped empty: {stats['skipped_empty']} • Failed: {stats['failed']}"
                )

                if stats["failed_details"]:
                    st.text("First few fetch errors (symbol & message):")
                    for item in stats["failed_details"]:
                        st.write(f"- {item['symbol']}: {item['error']}")

                # If there is data for some symbols, show top movers (even if negative)
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
                    st.error("No symbol returned any data for the selected year. Check your symbol list formatting (e.g., SYMBOL vs SYMBOL.NS) or the selected year.")
    else:
        st.info("👆 Select a year and click **Fetch Yearly Data** to start.")
else:
    st.stop()

# ======================
# 🧾 FOOTNOTE
# ======================
st.caption("Data Source: Yahoo Finance | Built by Shubham Kishor | Results cached for 1 hour.")
