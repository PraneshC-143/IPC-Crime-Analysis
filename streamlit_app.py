import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os # Import the os module

# --- Data Loading and Caching ---
# Use st.cache_data to load the data only once
@st.cache_data
def load_data(file_path):
    """Loads and cleans the crime dataset."""
    try:
        # Use os.path.join to create the file path relative to the script's directory
        script_dir = os.path.dirname(__file__)
        absolute_file_path = os.path.join(script_dir, file_path)
        df = pd.read_excel(absolute_file_path, sheet_name="districtwise-ipc-crimes")
        # Data cleaning
        df = df.drop(columns=['id', 'state_code', 'district_code'])
        # Identify numeric columns used for crime calculation
        crime_columns = df.select_dtypes(include='number').columns.drop(['year'])
        df['total_crimes'] = df[crime_columns].sum(axis=1)
        return df, crime_columns
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the script.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None, None

df, crime_columns = load_data("districtwise-ipc-crimes.xlsx") # Pass only the filename here

# --- Model Training and Caching ---
@st.cache_resource
def train_model(x, y):
    """Trains the RandomForestRegressor model and caches it."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model, x_test, y_test

# Check if data is loaded before proceeding
if df is not None:
    # Title and description
    st.title("District-wise IPC Crime Analysis Dashboard")
    st.markdown("""
    This app helps government officials and researchers explore crime patterns
    and predict total IPC crimes based on input variables.
    """)

    # Sidebar filter
    st.sidebar.header("Filters")
    state_filter = st.sidebar.selectbox("Select State", df['state_name'].unique())
    df_state = df[df['state_name'] == state_filter]

    # --- Main Page Layout ---

    # Top districts by total crimes
    st.subheader(f"Top 10 Districts in {state_filter} by Total IPC Crimes")
    top_districts = df_state.groupby('district_name')['total_crimes'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_districts)

    # Crime type heatmap
    st.subheader("Crime Type Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[crime_columns].corr(), cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Distribution of total crimes
    st.subheader("Distribution of Total Crimes")
    fig, ax = plt.subplots()
    sns.histplot(df['total_crimes'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot of total crimes per state
    st.subheader("Total Crimes per State (Boxplot)")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x='state_name', y='total_crimes', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    # Pairplot of selected crimes
    st.subheader("Pairplot of Selected Crimes")
    # Ensure we don't select more columns than available
    num_cols_to_plot = min(5, len(crime_columns))
    selected_cols = crime_columns[:num_cols_to_plot].tolist() + ['total_crimes']
    pairplot_fig = sns.pairplot(df[selected_cols])
    st.pyplot(pairplot_fig)

    # --- Predictive Modeling ---
    st.header("Predictive Modeling")
    st.subheader("Predict Total IPC Crimes")

    x = df[crime_columns]
    y = df['total_crimes']

    model, x_test, y_test = train_model(x, y)
    y_pred = model.predict(x_test)

    st.markdown(f"**Model Performance (on a test set):**")
    st.markdown(f"- **R² Score:** {r2_score(y_test, y_pred):.2f}")
    # Calculate RMSE manually
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.markdown(f"- **RMSE:** {rmse:.2f}")

    # Custom input prediction
    st.subheader("Enter Crime Values to Predict Total IPC Crimes")
    with st.form("prediction_form"):
        user_input = {}
        # Use columns for a better layout
        cols = st.columns(4)
        # Limit to the first 16 for a cleaner UI
        for i, col_name in enumerate(crime_columns[:16]):
            user_input[col_name] = cols[i % 4].number_input(f"{col_name}", min_value=0, value=0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            # Create a DataFrame with columns in the correct order
            input_df = pd.DataFrame([user_input], columns=crime_columns)
            # Fill any missing columns with 0 (if user_input has fewer than all crime_columns)
            input_df = input_df.fillna(0)
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Total IPC Crimes: {int(prediction)}")

    st.markdown("---")
    st.markdown("Developed for government crime analysis and prediction.")
