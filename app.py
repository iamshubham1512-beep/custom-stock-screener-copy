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
