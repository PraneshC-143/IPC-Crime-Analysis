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

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data
def load_data(file_path):
    script_dir = os.path.dirname(__file__)
    absolute_file_path = os.path.join(script_dir, file_path)

    df = pd.read_excel(
        absolute_file_path,
        sheet_name="districtwise-ipc-crimes"
    )

    # Drop unused columns
    df = df.drop(columns=["id", "state_code", "district_code"])

    # Crime columns (exclude year)
    crime_columns = df.select_dtypes(include="number").columns.drop("year")

    # Target
    df["total_crimes"] = df[crime_columns].sum(axis=1)

    # Features = crimes + year
    feature_columns = crime_columns.tolist() + ["year"]

    return df, crime_columns, feature_columns


df, crime_columns, feature_columns = load_data("districtwise-ipc-crimes.xlsx")

# -------------------------------
# MODEL TRAINING
# -------------------------------
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


# -------------------------------
# STREAMLIT APP
# -------------------------------
if df is not None:

    st.title("District-wise IPC Crime Analysis & Prediction Dashboard")

    st.markdown(
        "This dashboard explores IPC crime patterns and predicts **total IPC crimes** "
        "using crime data and year."
    )

    # -------------------------------
    # SIDEBAR FILTER
    # -------------------------------
    st.sidebar.header("Filters")
    state_filter = st.sidebar.selectbox(
        "Select State",
        sorted(df["state_name"].unique())
    )

    df_state = df[df["state_name"] == state_filter]

    # -------------------------------
    # VISUALIZATIONS
    # -------------------------------
    st.subheader(f"Top 10 Districts in {state_filter} by Total IPC Crimes")

    top_districts = (
        df_state.groupby("district_name")["total_crimes"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_districts)

    # Heatmap
    st.subheader("Crime Type Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[crime_columns].corr(method="spearman")
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution
    st.subheader("Distribution of Total Crimes")
    fig, ax = plt.subplots()
    sns.histplot(df["total_crimes"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader("Total Crimes per State (Top 10 States)")
    top_states = (
        df.groupby("state_name")["total_crimes"]
        .sum()
        .nlargest(10)
        .index
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df[df["state_name"].isin(top_states)],
        x="state_name",
        y="total_crimes",
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Pairplot (sampled)
    st.subheader("Pairplot of Selected Crimes")
    selected_cols = crime_columns[:5].tolist() + ["total_crimes"]
    sample_df = df[selected_cols].sample(
        min(2000, len(df)),
        random_state=42
    )
    pairplot_fig = sns.pairplot(sample_df)
    st.pyplot(pairplot_fig)

    # Year trend
    st.subheader("Year-wise IPC Crime Trend")
    yearly = df.groupby("year")["total_crimes"].sum().reset_index()
    fig = px.line(yearly, x="year", y="total_crimes", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # PREDICTIVE MODELING
    # -------------------------------
    st.header("Predictive Modeling")

    x = df[feature_columns]
    y = df["total_crimes"]

    model, x_test, y_test = train_model(x, y)
    y_pred = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown("### Model Performance")
    st.markdown(f"- **R² Score:** {r2:.3f}")
    st.markdown(f"- **RMSE:** {rmse:.2f}")

    # -------------------------------
    # PREDICTION FORM
    # -------------------------------
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
            user_input[col] = cols[i % 4].number_input(
                col,
                min_value=0,
                value=0
            )

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([user_input])

            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_columns]

            prediction = model.predict(input_df)[0]
            prediction = max(0, int(prediction))

            st.success(
                f"Predicted Total IPC Crimes for year {year}: {prediction}"
            )

    st.markdown("---")
    st.markdown("Developed for crime analysis and forecasting.")
