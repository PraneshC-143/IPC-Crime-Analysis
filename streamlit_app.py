import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="CrimeScope | IPC Crime Intelligence",
    layout="wide"
)

# ==================================================
# UI (CLEAN, POLICY-STYLE)
# ==================================================
st.markdown("""
<style>
.main { background-color: #fafafa; }
section[data-testid="stSidebar"] {
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}
.kpi {
    background: white;
    padding: 14px;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}
.kpi p { margin: 0; font-size: 12px; color: #6b7280; }
.kpi h2 { margin-top: 6px; font-size: 20px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "districtwise-ipc-crimes.xlsx")
    df = pd.read_excel(path, sheet_name="districtwise-ipc-crimes")
    df.drop(columns=["id", "state_code", "district_code"], inplace=True)

    crime_cols = df.select_dtypes(include="number").columns.drop("year")
    df["total_crimes"] = df[crime_cols].sum(axis=1)

    return df, crime_cols

df, crime_columns = load_data()

# ==================================================
# SIDEBAR FILTERS (THE DRIVER)
# ==================================================
st.sidebar.title("CrimeScope")
st.sidebar.caption("IPC Crime Intelligence System")

state = st.sidebar.selectbox(
    "State",
    sorted(df["state_name"].unique())
)

districts = df[df["state_name"] == state]["district_name"].unique()
district = st.sidebar.selectbox(
    "District",
    ["All Districts"] + sorted(districts)
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

crime_types = st.sidebar.multiselect(
    "Crime Types",
    crime_columns.tolist(),
    default=crime_columns.tolist()
)

# ==================================================
# SINGLE SOURCE OF TRUTH (PREPROCESSING)
# ==================================================
filtered = df[
    (df["state_name"] == state) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

if district != "All Districts":
    filtered = filtered[filtered["district_name"] == district]

filtered["crime_sum"] = filtered[crime_types].sum(axis=1)

# ==================================================
# HEADER + CONTEXT
# ==================================================
st.markdown("## IPC Crime Analysis")

st.markdown(
    f"""
    **State:** {state}  
    **District:** {district}  
    **Years:** {year_range[0]} – {year_range[1]}  
    **Crime Types:** {", ".join(crime_types)}
    """
)

# ==================================================
# KPIs (CONTEXT-AWARE)
# ==================================================
total_crimes = int(filtered["crime_sum"].sum())
peak_year = filtered.groupby("year")["crime_sum"].sum().idxmax()

k1, k2, k3 = st.columns(3)
k1.markdown(f"<div class='kpi'><p>Total Crimes</p><h2>{total_crimes:,}</h2></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><p>Peak Year</p><h2>{peak_year}</h2></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'><p>Scope</p><h2>{'District' if district!='All Districts' else 'State'}</h2></div>", unsafe_allow_html=True)

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Distribution", "Relationships", "Hotspots"]
)

# ==================================================
# OVERVIEW
# ==================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Districts")
        top = (
            filtered.groupby("district_name")["crime_sum"]
            .sum()
            .reset_index()
            .sort_values("crime_sum", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top,
            x="district_name",
            y="crime_sum",
            title="Crime Concentration"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Crime Trend")

        trend = filtered.groupby("year")["crime_sum"].sum().reset_index()

        fig = px.line(
            trend,
            x="year",
            y="crime_sum",
            markers=True
        )
        fig.update_traces(line=dict(width=4))
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DISTRIBUTION
# ==================================================
with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered["crime_sum"], bins=25, kde=True, ax=ax)
    ax.set_xlabel("Crime Count")
    st.pyplot(fig)

# ==================================================
# RELATIONSHIPS
# ==================================================
with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df[crime_columns].corr(method="spearman"),
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# ==================================================
# HOTSPOT MAP (HONEST & FILTER-DRIVEN)
# ==================================================
with tab4:
    st.subheader("Crime Hotspots (State-level Aggregation)")

    map_df = (
        filtered.groupby("state_name")["crime_sum"]
        .sum()
        .reset_index()
    )

    fig = px.choropleth_mapbox(
        map_df,
        geojson="https://raw.githubusercontent.com/datameet/maps/master/States/india_states.geojson",
        locations="state_name",
        featureidkey="properties.ST_NM",
        color="crime_sum",
        color_continuous_scale="Reds",
        mapbox_style="carto-positron",
        zoom=4,
        center={"lat": 22.5, "lon": 79},
        opacity=0.85
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Hotspots represent aggregated crime intensity at the state level. "
        "District-level geographic visualization requires district boundary data, "
        "which is not available in the dataset."
    )

# ==================================================
# FOOTER
# ==================================================
st.caption(
    "CrimeScope | Interactive IPC Crime Intelligence Dashboard. "
    "Designed for policy analysis and academic evaluation."
)
