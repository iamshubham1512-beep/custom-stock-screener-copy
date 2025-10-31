import streamlit as st
import polars as pl
import numpy as np
import requests
import hashlib
import time
import datetime as dt
from pathlib import Path
import json

# ======================
# 🎯 APP CONFIGURATION
# ======================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")

st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year and filter the stocks based on various parameters.")

# ======================
# 📂 CACHE & SAVE PATHS
# ======================
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SAVE_DIR = Path("saved_filters")
SAVE_DIR.mkdir(exist_ok=True)


# ======================
# 🔄 CACHE HANDLER
# ======================
@st.cache_data(ttl=3600)
def fetch_data(year: int):
    """Fetch stock data for a given year from API or local cache."""
    cache_file = CACHE_DIR / f"{year}_data.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)
    else:
        # placeholder API call
        url = f"https://api.stockdata.com/{year}"  # example
        resp = requests.get(url)
        if resp.status_code != 200:
            st.error("❌ Failed to fetch data.")
            return None
        df = pl.DataFrame(resp.json())
        df.write_parquet(cache_file)
        return df


# ======================
# 🧮 FILTERING FUNCTIONS
# ======================
def filter_by_age_pl(df_pl, year, age_filter):
    """Filter by stock listing age."""
    cutoff = dt.datetime.now().date()
    if age_filter == "1Y":
        cutoff = cutoff.replace(year=cutoff.year - 1)
    elif age_filter == "3Y":
        cutoff = cutoff.replace(year=cutoff.year - 3)
    elif age_filter == "5Y":
        cutoff = cutoff.replace(year=cutoff.year - 5)

    lookup = {}
    for row in df_pl.iter_rows(named=True):
        symbol = row.get("symbol")
        date_str = row.get("date")
        if date_str and symbol:
            try:
                first_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                lookup[symbol] = first_date
            except Exception:
                continue

    valid_symbols = [
        s for s in df_pl["symbol"].to_list()
        if lookup.get(s, dt.datetime.now().date()) <= cutoff
    ]
    return df_pl.filter(pl.col("symbol").is_in(valid_symbols))


# ======================
# 💾 SAVE & LOAD DATA
# ======================
def save_filtered_data(df, filters, year):
    """Save filtered dataframe and metadata."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{year}_{timestamp}.parquet"
    meta_filename = f"{year}_{timestamp}.json"

    df.write_parquet(SAVE_DIR / filename)
    with open(SAVE_DIR / meta_filename, "w") as f:
        json.dump(
            {
                "year": year,
                "filters": filters,
                "saved_on": dt.datetime.now().strftime("%d %b %Y, %I:%M %p"),
            },
            f,
            indent=4,
        )

    st.sidebar.success(f"✅ Saved filtered data as `{filename}`")


def load_saved_data():
    """List saved parquet files sorted by date."""
    files = sorted(SAVE_DIR.glob("*.parquet"), reverse=True)
    return files


# ======================
# 🧭 SIDEBAR: YEAR SELECTION
# ======================
year = st.selectbox("Select Year", options=list(range(2016, dt.datetime.now().year + 1))[::-1])
st.session_state["fetched_year"] = year

# Load data for year
if "fetched_data_pl" not in st.session_state or st.session_state.get("fetched_year") != year:
    st.session_state["fetched_data_pl"] = fetch_data(year)

df_result_pl = st.session_state["fetched_data_pl"]

# ======================
# 🎛️ FILTERS + DISPLAY
# ======================
if df_result_pl is not None and not df_result_pl.is_empty():
    min_open, max_open = df_result_pl["open"].min(), df_result_pl["open"].max()

    st.sidebar.markdown(f"**Min Open ({min_open:.2f})**")
    open_min_val = st.sidebar.number_input(
        "Min Open", min_value=float(min_open), value=float(min_open)
    )

    st.sidebar.markdown(f"**Max Open ({max_open:.2f})**")
    open_max_val = st.sidebar.number_input(
        "Max Open", min_value=float(min_open), value=float(max_open)
    )

    pct_min_val = st.sidebar.number_input("Min % Change", value=-50.0, step=1.0)
    pct_max_val = st.sidebar.number_input("Max % Change", value=50.0, step=1.0)
    age_filter = st.sidebar.selectbox("Age Filter", ["All", "1Y", "3Y", "5Y"])

    # Apply filters
    filtered_pl = df_result_pl.filter(
        (pl.col("open") >= open_min_val)
        & (pl.col("open") <= open_max_val)
        & (pl.col("pct_change") >= pct_min_val)
        & (pl.col("pct_change") <= pct_max_val)
    )

    filtered_pl = filter_by_age_pl(filtered_pl, year, age_filter)

    # Store for later use
    st.session_state["last_filtered_pl"] = filtered_pl

    # ======================
    # 💾 SAVE FILTERED DATA FEATURE
    # ======================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Save Current Filtered Data")

    if (
        "last_filtered_pl" in st.session_state
        and st.session_state["last_filtered_pl"] is not None
        and not st.session_state["last_filtered_pl"].is_empty()
    ):
        if st.sidebar.button("💾 Save Current Filtered Data"):
            active_filters = {
                "year": year,
                "open_min_val": open_min_val,
                "open_max_val": open_max_val,
                "pct_min_val": pct_min_val,
                "pct_max_val": pct_max_val,
                "age_filter": age_filter,
            }
            save_filtered_data(filtered_pl, active_filters, year)
    else:
        st.sidebar.caption("⚙️ Save option will appear once filters are applied and results are visible.")

    # ======================
    # 📂 LOAD SAVED DATA
    # ======================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Load Saved Filtered Data")

    saved_files = load_saved_data()
    if saved_files:
        selected_file = st.sidebar.selectbox(
            "Select a saved dataset", saved_files, format_func=lambda x: x.name
        )
        if st.sidebar.button("🔄 Load Selected File"):
            loaded_df = pl.read_parquet(selected_file)
            st.session_state["last_filtered_pl"] = loaded_df
            st.sidebar.success(f"✅ Loaded `{selected_file.name}`")
    else:
        st.sidebar.info("No saved filtered data found yet.")

    # ======================
    # 📊 FINAL DISPLAY
    # ======================
    filtered_pd = st.session_state["last_filtered_pl"].to_pandas()
    filtered_pd = filtered_pd.reset_index(drop=True)
    filtered_pd.insert(0, "Sl. No.", np.arange(1, len(filtered_pd) + 1))

    # Rename symbol column
    if "symbol" in filtered_pd.columns:
        filtered_pd.rename(columns={"symbol": "Symbol"}, inplace=True)

    # Swap % Change and Avg. Volume columns if exist
    columns = list(filtered_pd.columns)
    if "pct_change" in columns and "avg_volume" in columns:
        pct_idx = columns.index("pct_change")
        vol_idx = columns.index("avg_volume")
        columns[pct_idx], columns[vol_idx] = columns[vol_idx], columns[pct_idx]
        filtered_pd = filtered_pd[columns]

    st.dataframe(
        filtered_pd,
        use_container_width=True,
        hide_index=True,
    )

# ======================
# 🕒 FOOTER
# ======================
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Built by Shubham Kishor | Cached for 1 hour | Cache last refreshed: {dt.datetime.now().strftime('%d %b %Y, %I:%M %p')}"
)
