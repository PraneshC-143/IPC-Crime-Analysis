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
    page_title="CrimeScope | IPC Crime Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS (PROFESSIONAL CRIME THEME)
# ==================================================
st.markdown("""
<style>
.main { background-color: #f5f7fb; }

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] select {
    color: black !important;
    background-color: white !important;
}

h1, h2, h3 { font-weight: 600; }

.kpi-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    text-align: center;
}

.kpi-card p {
    margin: 0;
    color: #64748b;
    font-size: 14px;
}

.kpi-card h2 {
    margin-top: 6px;
    color: #2563eb;
}

.section { margin-top: 35px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADING
# ==================================================
@st.cache_data
def load_data(file_path):
    base = os.path.dirname(__file__)
    path = os.path.join(base, file_path)

    df = pd.read_excel(path, sheet_name="districtwise-ipc-crimes")
    df = df.drop(columns=["id", "state_code", "district_code"])

    crime_cols = df.select_dtypes(include="number").columns.drop("year")
    df["total_crimes"] = df[crime_cols].sum(axis=1)

    features = crime_cols.tolist() + ["year"]
    return df, crime_cols, features

df, crime_columns, feature_columns = load_data("districtwise-ipc-crimes.xlsx")

# ==================================================
# MODEL TRAINING
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
# SIDEBAR (FILTERS)
# ==================================================
st.sidebar.title("CrimeScope")
st.sidebar.caption("IPC Crime Analytics")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["state_name"].unique())
)

df_state = df[df["state_name"] == state]

district = st.sidebar.selectbox(
    "Select District",
    ["All Districts"] + sorted(df_state["district_name"].unique())
)

if district == "All Districts":
    df_filtered = df_state
else:
    df_filtered = df_state[df_state["district_name"] == district]

# ==================================================
# HEADER
# ==================================================
st.markdown("## District-wise IPC Crime Analysis & Prediction")
st.markdown(
    "A professional crime analytics dashboard to explore IPC crime patterns "
    "and predict total crimes using machine learning."
)

# ==================================================
# KPI CARDS
# ==================================================
total_crimes = int(df_filtered["total_crimes"].sum())

if district == "All Districts":
    top_district = (
        df_state.groupby("district_name")["total_crimes"]
        .sum()
        .idxmax()
    )
else:
    top_district = district

k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"<div class='kpi-card'><p>Total Crimes</p><h2>{total_crimes:,}</h2></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><p>Top District</p><h2>{top_district}</h2></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><p>Crime Types</p><h2>{len(crime_columns)}</h2></div>", unsafe_allow_html=True)
k4.markdown("<div class='kpi-card'><p>Model</p><h2>Random Forest</h2></div>", unsafe_allow_html=True)

# ==================================================
# CHARTS (BAR + TREND)
# ==================================================
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.subheader("Top 10 Districts by Total Crimes")
    top10 = (
        df_filtered.groupby("district_name")["total_crimes"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top10)

with right:
    st.subheader("Year-wise IPC Crime Trend")
    yearly = df_filtered.groupby("year")["total_crimes"].sum().reset_index()
    fig = px.line(yearly, x="year", y="total_crimes", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# HISTOGRAM
# ==================================================
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Distribution of Total Crimes")

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_filtered["total_crimes"], bins=30, kde=True, ax=ax, color="#2563eb")
ax.set_xlabel("Total Crimes")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ==================================================
# BOXPLOT
# ==================================================
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Crime Distribution Across Districts")

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(
    data=df_state,
    x="district_name",
    y="total_crimes",
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# ==================================================
# HEATMAP
# ==================================================
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Crime Type Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(df[crime_columns].corr(method="spearman"), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ==================================================
# PREDICTIVE MODELING
# ==================================================
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.header("Predictive Modeling")

x = df[feature_columns]
y = df["total_crimes"]

model, x_test, y_test = train_model(x, y)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.markdown(f"**R² Score:** {r2:.3f}")
st.markdown(f"**RMSE:** {rmse:.2f}")

# ==================================================
# PREDICTION FORM
# ==================================================
st.subheader("Predict Total IPC Crimes")

with st.form("prediction_form"):
    user_input = {}

    year = st.number_input(
        "Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()) + 5,
        value=int(df["year"].max())
    )
    user_input["year"] = year

    st.markdown("### Enter Crime Counts")
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
        st.success(f"Predicted Total IPC Crimes for year {year}: {prediction}")

# ==================================================
# FOOTER
# ==================================================
st.caption("CrimeScope – District-wise IPC Crime Analysis & Prediction Dashboard")
