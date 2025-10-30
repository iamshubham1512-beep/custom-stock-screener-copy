import streamlit as st
import polars as pl
import io
import requests

# ==========================================
# ⚙️ PAGE CONFIG
# ==========================================
st.set_page_config(page_title="📈 NSE Yearly Top Gainers", layout="wide")
st.title("📈 NSE Yearly Top Gainers Screener (Ultra Fast ⚡)")
st.markdown("Filter and view yearly top gainers based on % change and volume data.")

# ==========================================
# 🌐 HUGGING FACE PARQUET LINKS
# ==========================================
DATA_URLS = {
    "2016-2020": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2016_2020.parquet",
    "2021-2024": "https://huggingface.co/datasets/Chiron-S/NSE_Stocks_Data/resolve/main/NSE_Stocks_2021_2024.parquet"
}

# ==========================================
# ⚡ CACHE + LOAD PARQUET (WITH POLARS)
# ==========================================
@st.cache_data(show_spinner=True)
def load_data_from_hf(url: str) -> pl.DataFrame:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        parquet_bytes = io.BytesIO(response.content)
        df = pl.read_parquet(parquet_bytes)

        # --- Ensure 'date' column exists ---
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column in dataset")

        # --- Handle mixed formats ---
        if df["date"].dtype != pl.Datetime:
            df = df.with_columns([
                pl.when(pl.col("date").str.contains("-"))
                  .then(pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False))
                  .otherwise(pl.col("date").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False))
                  .alias("date")
            ])

        # --- Drop nulls & add year column ---
        df = df.drop_nulls(["date", "open", "close"])
        df = df.with_columns(pl.col("date").dt.year().alias("year"))
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pl.DataFrame()

# ==========================================
# 🧠 SIDEBAR - YEAR SELECTION
# ==========================================
st.sidebar.header("Filter Options")
year = st.sidebar.selectbox("Select Year", list(range(2016, 2025)), index=0)

# Select file based on year
data_url = DATA_URLS["2016-2020"] if year <= 2020 else DATA_URLS["2021-2024"]

# ==========================================
# 📦 LOAD DATA
# ==========================================
with st.spinner(f"Loading and filtering data for {year}..."):
    df = load_data_from_hf(data_url)

if df.is_empty():
    st.warning("No data loaded. Please check dataset URL or internet connection.")
    st.stop()

# ==========================================
# 🎯 FILTER DATA FOR SELECTED YEAR
# ==========================================
yearly_df = df.filter(pl.col("year") == year)

if yearly_df.is_empty():
    st.warning(f"No records found for {year}.")
    st.stop()

# ==========================================
# 📊 CALCULATE YEARLY TOP GAINERS
# ==========================================
try:
    gainers = (
        yearly_df
        .group_by("symbol")
        .agg([
            pl.col("open").first().alias("Open"),
            pl.col("close").last().alias("Close"),
            pl.col("volume").mean().round(0).alias("Avg. Volume")
        ])
        .with_columns([
            ((pl.col("Close") - pl.col("Open")) / pl.col("Open") * 100)
            .alias("% Change")
            .round(2)
        ])
        .sort("% Change", descending=True)
    )

    st.success(f"✅ Data processed successfully for {year}")
    st.metric(label="Total Stocks", value=gainers.height)

# ======================
# 🎛️ REAL-TIME FILTERS
# ======================
if "fetched_data_pl" in st.session_state and st.session_state["fetched_data_pl"] is not None:
    df_result_pl = st.session_state["fetched_data_pl"]
    st.subheader(f"📊 Filter Results for {st.session_state['fetched_year']}")

    # Compute bounds (Polars -> Python scalars)
    try:
        open_min_bound = int(np.floor(df_result_pl.select(pl.min("Open Price")).item() / 10) * 10)
        open_max_bound = int(np.ceil(df_result_pl.select(pl.max("Open Price")).item() / 10) * 10)
        pct_min_bound = int(np.floor(df_result_pl.select(pl.min("% Change")).item() / 10) * 10)
        pct_max_bound = int(np.ceil(df_result_pl.select(pl.max("% Change")).item() / 10) * 10)
    except Exception:
        open_min_bound, open_max_bound, pct_min_bound, pct_max_bound = 0, 1000, -100, 100

    # Immutable initial labels
    if "open_min_init" not in st.session_state or st.session_state.get("init_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_init = open_min_bound
        st.session_state.open_max_init = open_max_bound
        st.session_state.pct_min_init = pct_min_bound
        st.session_state.pct_max_init = pct_max_bound
        st.session_state.init_year = st.session_state["fetched_year"]

    # Editable inputs
    if "open_min_val" not in st.session_state or st.session_state.get("inputs_year") != st.session_state["fetched_year"]:
        st.session_state.open_min_val = open_min_bound
        st.session_state.open_max_val = open_max_bound
        st.session_state.pct_min_val = pct_min_bound
        st.session_state.pct_max_val = pct_max_bound
        st.session_state.inputs_year = st.session_state["fetched_year"]

    # Open Price inputs
    st.markdown(
        f"#### Open Price Range (₹) — Min ({st.session_state.open_min_init}) · Max ({st.session_state.open_max_init})"
    )
    cmin, cmax = st.columns(2)
    with cmin:
        st.session_state.open_min_val = st.number_input(
            "Min", min_value=open_min_bound, max_value=open_max_bound,
            value=max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val)),
            step=1, key="open_min_input"
        )
    with cmax:
        st.session_state.open_max_val = st.number_input(
            "Max", min_value=open_min_bound, max_value=open_max_bound,
            value=min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val)),
            step=1, key="open_max_input"
        )
    st.session_state.open_min_val = max(open_min_bound, min(st.session_state.open_min_val, st.session_state.open_max_val))
    st.session_state.open_max_val = min(open_max_bound, max(st.session_state.open_max_val, st.session_state.open_min_val))
    open_range = (st.session_state.open_min_val, st.session_state.open_max_val)

    # % Change inputs
    st.markdown(
        f"#### % Change Range — Min ({st.session_state.pct_min_init}) · Max ({st.session_state.pct_max_init})"
    )
    cpmin, cpmax = st.columns(2)
    with cpmin:
        st.session_state.pct_min_val = st.number_input(
            "Min", min_value=pct_min_bound, max_value=pct_max_bound,
            value=max(pct_min_bound, min(st.session_state.pct_min_val, st.session_state.pct_max_val)),
            step=1, key="pct_min_input"
        )
    with cpmax:
        st.session_state.pct_max_val = st.number_input(
            "Max", min_value=pct_min_bound, max_value=pct_max_bound,
            value=min(pct_max_bound, max(st.session_state.pct_max_val, st.session_state.pct_min_val)),
            step=1, key="pct_max_input"
        )
    st.session_state.pct_min_val = max(pct_min_bound, min(st.session_state.pct_min_val, st.session_state.pct_max_val))
    st.session_state.pct_max_val = min(pct_max_bound, max(st.session_state.pct_max_val, st.session_state.pct_min_val))
    pct_range = (st.session_state.pct_min_val, st.session_state.pct_max_val)

    # Apply filters in Polars
    filtered_pl = df_result_pl.filter(
        (pl.col("Open Price") >= open_range[0]) &
        (pl.col("Open Price") <= open_range[1]) &
        (pl.col("% Change") >= pct_range[0]) &
        (pl.col("% Change") <= pct_range[1])
    )

    # Vol filter
    vol_filter = st.selectbox("Filter by Avg. Volume", options=[
        "All", "More than 100K", "More than 150K", "More than 200K",
        "More than 250K", "More than 300K", "More than 350K",
        "More than 400K", "More than 500K"
    ], key="vol_select")
    if vol_filter != "All":
        vol_threshold = int(vol_filter.split(" ")[-1].replace("K", "000"))
        filtered_pl = filtered_pl.filter(pl.col("Avg. Volume") > vol_threshold)

    
    # ==========================================
    # 📋 DISPLAY RESULTS
    # ==========================================
    st.dataframe(gainers.to_pandas(), use_container_width=True, height=700)

    # ==========================================
    # 📥 DOWNLOAD OPTION
    # ==========================================
    csv_data = gainers.write_csv()
    st.download_button(
        label=f"📥 Download {year} Gainers (CSV)",
        data=csv_data,
        file_name=f"NSE_Top_Gainers_{year}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"⚠️ Error while processing data: {e}")
