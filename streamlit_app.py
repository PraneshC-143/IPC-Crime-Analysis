import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="CrimeScope – IPC Crime Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CLEAN ACADEMIC UI
# ==================================================
st.markdown("""
<style>
.main { background-color: #fafafa; }
section[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #e5e7eb;
}
.kpi {
    background: white;
    padding: 14px;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}
.kpi p {
    margin: 0;
    font-size: 13px;
    color: #6b7280;
}
.kpi h2 {
    margin-top: 4px;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================
@st.cache_data
def load_data(file):
    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_excel(path, sheet_name="districtwise-ipc-crimes")
    df = df.drop(columns=["id", "state_code", "district_code"])
    crime_cols = df.select_dtypes(include="number").columns.drop("year")
    df["total_crimes"] = df[crime_cols].sum(axis=1)
    features = crime_cols.tolist() + ["year"]
    return df, crime_cols, features

df, crime_columns, feature_columns = load_data(
    "districtwise-ipc-crimes.xlsx"
)

# ==================================================
# INDIA STATES GEOJSON (NO LAT/LON NEEDED)
# ==================================================
@st.cache_data
def india_states_geojson():
    return (
        "https://raw.githubusercontent.com/"
        "datameet/maps/master/States/india_states.geojson"
    )

# ==================================================
# MODEL (ACADEMIC – UNCHANGED LOGIC)
# ==================================================
@st.cache_resource
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model, x_test, y_test

# ==================================================
# SIDEBAR – FILTERS & PREPROCESSING
# ==================================================
st.sidebar.title("CrimeScope")
st.sidebar.caption("District-wise IPC Crime Analysis")

state = st.sidebar.selectbox(
    "State",
    sorted(df["state_name"].unique())
)

df_state = df[df["state_name"] == state]

district = st.sidebar.selectbox(
    "District",
    ["All Districts"] + sorted(df_state["district_name"].unique())
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

selected_crimes = st.sidebar.multiselect(
    "Crime Types",
    crime_columns.tolist(),
    default=crime_columns.tolist()
)

aggregation_mode = st.sidebar.radio(
    "Aggregation Mode",
    ["Total Crimes", "Average per Year"]
)

# ==================================================
# INTERACTIVE PREPROCESSING
# ==================================================
df_pre = df_state[
    (df_state["year"] >= year_range[0]) &
    (df_state["year"] <= year_range[1])
]

if district != "All Districts":
    df_pre = df_pre[df_pre["district_name"] == district]

df_pre["selected_total_crimes"] = df_pre[selected_crimes].sum(axis=1)

if aggregation_mode == "Average per Year":
    df_pre["selected_total_crimes"] = (
        df_pre.groupby("district_name")["selected_total_crimes"]
        .transform("mean")
    )

# ==================================================
# HEADER & KPI
# ==================================================
st.markdown("## District-wise IPC Crime Analysis")

total_crimes = int(df_pre["selected_total_crimes"].sum())
peak_year = df_pre.groupby("year")["selected_total_crimes"].sum().idxmax()

k1, k2, k3 = st.columns(3)

k1.markdown(
    f"<div class='kpi'><p>Total Crimes</p><h2>{total_crimes:,}</h2></div>",
    unsafe_allow_html=True
)
k2.markdown(
    f"<div class='kpi'><p>State</p><h2>{state}</h2></div>",
    unsafe_allow_html=True
)
k3.markdown(
    f"<div class='kpi'><p>Peak Year</p><h2>{peak_year}</h2></div>",
    unsafe_allow_html=True
)

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Distribution", "Relationships", "Prediction", "Hotspots"]
)

# ==================================================
# OVERVIEW TAB
# ==================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Districts")
        top10 = (
            df_pre.groupby("district_name")["selected_total_crimes"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(top10)

    with col2:
        st.subheader("Crime Trend")
        trend = df_pre.groupby("year")["selected_total_crimes"].sum().reset_index()
        fig = px.line(trend, x="year", y="selected_total_crimes", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DISTRIBUTION TAB
# ==================================================
with tab2:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df_pre["selected_total_crimes"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Total Crimes")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ==================================================
# RELATIONSHIPS TAB
# ==================================================
with tab3:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        df[crime_columns].corr(method="spearman"),
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# ==================================================
# PREDICTION TAB (ACADEMIC DEMO)
# ==================================================
with tab4:
    x = df[feature_columns]
    y = df["total_crimes"]

    model, x_test, y_test = train_model(x, y)
    y_pred = model.predict(x_test)

    st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
    st.markdown(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ==================================================
# HOTSPOT MAP – STATE-WISE CHOROPLETH
# ==================================================
with tab5:
    st.subheader("Crime Hotspot Map (State-wise)")

    state_map_df = (
        df_pre.groupby("state_name")["selected_total_crimes"]
        .sum()
        .reset_index()
    )

    if state_map_df.empty:
        st.info("No data available for selected filters.")
    else:
        fig = px.choropleth_mapbox(
            state_map_df,
            geojson=india_states_geojson(),
            locations="state_name",
            featureidkey="properties.ST_NM",
            color="selected_total_crimes",
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",
            zoom=3.5,
            center={"lat": 22.5, "lon": 79},
            opacity=0.75,
            hover_name="state_name",
            height=600
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.caption(
    "CrimeScope | Interactive preprocessing and state-wise crime hotspot analysis. "
    "Choropleth visualization is used due to the absence of incident-level coordinates."
)
