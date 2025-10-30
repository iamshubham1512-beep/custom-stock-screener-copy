import streamlit as st
import polars as pl
import requests, io
import datetime as dt
import numpy as np
from typing import List, Dict, Tuple

# ======================================================
# 🎯 APP CONFIGURATION
# ======================================================
st.set_page_config(page_title="📈 Yearly Stock Screener", layout="wide")
st.title("📊 Yearly Top Gainers (NSE)")
st.write("Select a year to view top-performing NSE stocks based on yearly price change and average volume.")

# Hide number_input spinners
st.markdown("""
<style>
button.step-up, button.step-down { display: none !important; }
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button {
  -webkit-appearance: none; margin: 0;
}
input[type=number] { -moz-appearance: textfield; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 📂 DATA SOURCES (Hugging Face)
# ======================================================
DATA_URLS = {
    "2016-2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021-2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet"
}

# ======================================================
# ⚡ DATA LOADING WITH POLARS + CACHE
# ======================================================
@st.cache_data(ttl=3600, show_spinner=True)
def load_parquet_from_hf(url: str) -> pl.DataFrame:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        data = io.BytesIO(response.content)
        df = pl.read_parquet(data)

        df.columns = [c.lower() for c in df.columns]

        if df['date'].dtype != pl.Datetime:
            df = df.with_columns(pl.col("date").str.to_datetime(strict=False))

        df = df.drop_nulls(['date', 'open', 'close']).with_columns(
            pl.col('date').dt.year().alias('year')
        )

        return df
    except Exception as e:
        st.error(f"❌ Error loading {url}: {e}")
        return pl.DataFrame()

@st.cache_data(ttl=3600)
def load_master_table_parquet() -> pl.DataFrame:
    df1 = load_parquet_from_hf(DATA_URLS["2016-2020"])
    df2 = load_parquet_from_hf(DATA_URLS["2021-2024"])
    if not df1.is_empty() and not df2.is_empty():
        df = pl.concat([df1, df2], how="vertical_relaxed")
    else:
        df = df1 if not df1.is_empty() else df2
    return df

if "master_df" not in st.session_state:
    st.session_state["master_df"] = load_master_table_parquet()

master_df = st.session_state["master_df"]

@st.cache_data(ttl=3600)
def list_all_symbols() -> List[str]:
    return sorted(master_df.select(pl.col("symbol").unique()).to_series().to_list())

# ======================================================
# 📅 YEAR SELECTION
# ======================================================
year = st.selectbox("Select Year", options=list(range(2016, dt.datetime.now().year + 1))[::-1])

# ======================================================
# 🧠 AGE MAP + FILTERS
# ======================================================
@st.cache_data(ttl=86400)
def build_first_trade_lookup() -> Dict[str, dt.date]:
    df = (
        master_df.sort(["symbol", "date"])
        .group_by("symbol")
        .agg(pl.col("date").first().alias("FIRST_DATE"))
    )

    lookup: Dict[str, dt.date] = {}
    for r in df.iter_rows(named=True):
        sym = r["symbol"]
        first_dt = r["FIRST_DATE"]

        if first_dt is None:
            lookup[sym] = None
            continue

        if isinstance(first_dt, dt.datetime):
            lookup[sym] = first_dt.date()
        elif hasattr(first_dt, "item"):
            lookup[sym] = dt.datetime.utcfromtimestamp(first_dt.item() / 1e9).date()
        else:
            try:
                lookup[sym] = dt.datetime.strptime(str(first_dt), "%Y-%m-%d").date()
            except Exception:
                lookup[sym] = None

    return lookup


def filter_by_age_pl(df_pl: pl.DataFrame, year: int, age_option: str) -> pl.DataFrame:
    if df_pl.is_empty() or age_option == "All":
        return df_pl

    lookup = build_first_trade_lookup()
    year_cut_map = {"Older than 1 year": 1, "Older than 2 years": 2, "Older than 3 years": 3}
    N = year_cut_map.get(age_option, 0)

    cutoff = dt.date(year - N, 12, 31)
    valid_symbols = []

    for s in df_pl["Symbol"].to_list():
        first_date = lookup.get(s)
        if first_date and isinstance(first_date, dt.date) and first_date <= cutoff:
            valid_symbols.append(s)

    if not valid_symbols:
        return df_pl.filter(pl.lit(False))

    return df_pl.filter(pl.col("Symbol").is_in(valid_symbols))

# ======================================================
# 🧹 CACHE CLEAR BUTTON
# ======================================================
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared all caches. App will refresh.")

# ======================================================
# 🧮 YEARLY CALCULATION FUNCTION
# ======================================================
@st.cache_data(ttl=3600)
def fetch_yearly_data(symbols: List[str], year: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_year = master_df.filter(pl.col("year") == year)

    if df_year.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    yearly_data = (
        df_year.group_by("symbol")
        .agg([
            pl.col("open").first().alias("Open Price"),
            pl.col("close").last().alias("Close Price"),
            pl.col("volume").mean().alias("Avg. Volume"),
        ])
        .with_columns([
            ((pl.col("Close Price") - pl.col("Open Price")) / pl.col("Open Price") * 100)
            .round(2)
            .alias("% Change")
        ])
        .rename({"symbol": "Symbol"})
        .select(["Symbol", "Open Price", "Close Price", "% Change", "Avg. Volume"])
        .sort("% Change", descending=True)
    )

    gainers = yearly_data.filter(pl.col("% Change") > 0)
    return gainers, yearly_data

# ======================================================
# 🔍 FETCH YEARLY DATA
# ======================================================
all_symbols = list_all_symbols()

if all_symbols:
    if st.button("🔎 Fetch Yearly Data"):
        with st.spinner(f"Fetching data for {year}... Please wait (optimized)..."):
            df_final_pl, df_all_pl = fetch_yearly_data(all_symbols, year)
            if not df_final_pl.is_empty():
                st.session_state["fetched_data_pl"] = df_final_pl
                st.session_state["fetched_all_pl"] = df_all_pl
                st.session_state["fetched_year"] = year
                st.success(f"✅ Found {df_final_pl.height} positive gainers out of {df_all_pl.height} stocks.")
            else:
                st.warning("⚠️ No positive gainers found.")
                st.session_state["fetched_data_pl"] = None
else:
    st.stop()

# ======================================================
# 🎛️ FILTERS + DISPLAY
# ======================================================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result_pl = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Range bounds
    try:
        open_min_val = df_result_pl.select(pl.min("Open Price")).item()
        open_max_val = df_result_pl.select(pl.max("Open Price")).item()
        pct_min_bound = int(np.floor(df_result_pl.select(pl.min("% Change")).item() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result_pl.select(pl.max("% Change")).item() / 10) * 10)
    except Exception:
        open_min_val, open_max_val, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Dynamic labels showing values
    cmin, cmax = st.columns(2)
    st.session_state.open_min_val = cmin.number_input(
        f"Min Open (₹{open_min_val:.2f})", 
        min_value=int(np.floor(open_min_val / 10) * 10),
        max_value=int(np.ceil(open_max_val / 10) * 10),
        value=int(np.floor(open_min_val / 10) * 10)
    )
    st.session_state.open_max_val = cmax.number_input(
        f"Max Open (₹{open_max_val:.2f})", 
        min_value=int(np.floor(open_min_val / 10) * 10),
        max_value=int(np.ceil(open_max_val / 10) * 10),
        value=int(np.ceil(open_max_val / 10) * 10)
    )

    cpmin, cpmax = st.columns(2)
    st.session_state.pct_min_val = cpmin.number_input("Min % Change", min_value=pct_min_bound, max_value=pct_max_bound, value=pct_min_bound)
    st.session_state.pct_max_val = cpmax.number_input("Max % Change", min_value=pct_min_bound, max_value=pct_max_bound, value=pct_max_bound)

    # Apply filters
    filtered_pl = df_result_pl.filter(
        (pl.col("Open Price") >= st.session_state.open_min_val) &
        (pl.col("Open Price") <= st.session_state.open_max_val) &
        (pl.col("% Change") >= st.session_state.pct_min_val) &
        (pl.col("% Change") <= st.session_state.pct_max_val)
    )

    # Volume & age filters
    vol_filter = st.selectbox("Filter by Avg. Volume", ["All", "More than 100K", "More than 200K", "More than 300K"])
    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_pl = filtered_pl.filter(pl.col("Avg. Volume") > vol_threshold)

    age_filter = st.selectbox("Company Older Than", ["All", "Older than 1 year", "Older than 2 years", "Older than 3 years"])
    if age_filter != "All":
        with st.spinner(f"Filtering companies {age_filter.lower()}..."):
            filtered_pl = filter_by_age_pl(filtered_pl, st.session_state["fetched_year"], age_filter)

    # Display & download
    filtered_pd = filtered_pl.to_pandas().reset_index(drop=True)
    filtered_pd.insert(0, "Sl. No.", range(1, len(filtered_pd) + 1))  # Fixed serial numbering
    st.write(f"📈 Showing {len(filtered_pd)} results after filters:")

    # Disable reindexing during sorting
    st.dataframe(filtered_pd, use_container_width=True, hide_index=True)

    csv = filtered_pd.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv,
                       file_name=f"Filtered_Gainers_{st.session_state['fetched_year']}.csv",
                       mime="text/csv")

# ======================================================
# 🧾 FOOTNOTE
# ======================================================
st.caption(
    f"Data Source: Hugging Face Parquet (2016–2020 & 2021–2024) | "
    f"Built by Shubham Kishor | Cached for 1 hour | Cache last refreshed: {dt.datetime.now().strftime('%d %b %Y, %I:%M %p')}"
)

import streamlit as st

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Modern Dashboard",
    page_icon="💎",
    layout="wide",
)

# =====================
# CUSTOM CSS
# =====================
st.markdown("""
    <style>
    /* 🌈 Global App Style */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #2b2b4b);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    /* Hide Streamlit default footer & menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 💎 Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* 🌙 Button Style */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }

    /* 📊 Metric style override */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #d1d1d1;
    }

    /* 🧭 Sidebar style */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* ✨ Title styling */
    h1, h2, h3 {
        font-weight: 600;
        background: linear-gradient(90deg, #9f44d3, #4e9af1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Controls")
st.sidebar.slider("Opacity", 0.1, 1.0, 0.5)
st.sidebar.button("Refresh Data")

# =====================
# MAIN DASHBOARD
# =====================
st.title("💎 Modern Streamlit Dashboard")
st.write("A fully restyled example using glassmorphism + gradients.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Revenue", "₹12.3L", "+5.6%")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Users", "8,230", "+12%")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Engagement", "76%", "-2.3%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Activity Overview")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.line_chart({"Visitors": [50, 120, 90, 150, 300, 400, 500]})
st.markdown("</div>", unsafe_allow_html=True)
