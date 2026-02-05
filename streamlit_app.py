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
# NEUTRAL, LOGICAL WEB UI (NO FLASHY COLORS)
# ==================================================
st.markdown("""
<style>

/* Page background */
.main {
    background-color: #fafafa;
}

/* Sidebar – same as page (web-style) */
section[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* Headings */
h1, h2, h3 {
    font-weight: 500;
    color: #111827;
}

/* KPI cards – minimal, research-style */
.kpi {
    background-color: white;
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
    color: #111827;
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

df, crime_columns, feature_columns = load_data("districtwise-ipc-crimes.xlsx")

# ==================================================
# MODEL (ACADEMIC DEMONSTRATION)
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
# SIDEBAR FILTERS (LOGICAL HIERARCHY)
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

df_filtered = (
    df_state if district == "All Districts"
    else df_state[df_state["district_name"] == district]
)

# ==================================================
# HEADER
# ==================================================
st.markdown("## District-wise IPC Crime Analysis")
st.markdown(
    "This dashboard provides a structured analysis of **where crime is concentrated**, "
    "**how it changes over time**, and **how crime is distributed across districts**."
)

# ==================================================
# KPI SUMMARY (LOGICAL, NOT DECORATIVE)
# ==================================================
total_crimes = int(df_filtered["total_crimes"].sum())
peak_year = df_filtered.groupby("year")["total_crimes"].sum().idxmax()

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(
        f"<div class='kpi'><p>Total Crimes</p><h2>{total_crimes:,}</h2></div>",
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"<div class='kpi'><p>State</p><h2>{state}</h2></div>",
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"<div class='kpi'><p>Peak Crime Year</p><h2>{peak_year}</h2></div>",
        unsafe_allow_html=True
    )

# ==================================================
# TABS – ANALYTICAL SEPARATION (NO CLUTTER)
# ==================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Distribution", "Relationships", "Prediction"]
)

# ==================================================
# OVERVIEW TAB
# ==================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Districts by Total Crimes")
        top10 = (
            df_filtered.groupby("district_name")["total_crimes"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(top10)

    with col2:
        st.subheader("Year-wise Crime Trend")
        yearly = df_filtered.groupby("year")["total_crimes"].sum().reset_index()
        fig = px.line(yearly, x="year", y="total_crimes", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DISTRIBUTION TAB (ONE STATISTICAL CHART ONLY)
# ==================================================
with tab2:
    st.subheader("Distribution of Total Crimes")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df_filtered["total_crimes"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Total Crimes")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ==================================================
# RELATIONSHIPS TAB (ADVANCED ANALYSIS)
# ==================================================
with tab3:
    st.subheader("Correlation Between Crime Types")

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
    st.subheader("Predictive Modeling (Academic Demonstration)")

    x = df[feature_columns]
    y = df["total_crimes"]

    model, x_test, y_test = train_model(x, y)
    y_pred = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"**R² Score:** {r2:.3f}")
    st.markdown(f"**RMSE:** {rmse:.2f}")

    st.markdown("---")
    st.markdown("### Predict Total IPC Crimes")

    with st.form("prediction_form"):
        user_input = {}

        year = st.number_input(
            "Year (Exploratory)",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()) + 10,
            value=int(df["year"].max())
        )
        user_input["year"] = year

        cols = st.columns(4)
        for i, col in enumerate(crime_columns[:16]):
            user_input[col] = cols[i % 4].number_input(col, min_value=0, value=0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([user_input])
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]

            prediction = int(max(0, model.predict(input_df)[0]))
            st.success(f"Predicted Total IPC Crimes: {prediction}")

# ==================================================
# FOOTER
# ==================================================
st.caption(
    "CrimeScope – District-wise IPC Crime Analytics | "
    "Predictions are indicative and intended for academic demonstration."
)
