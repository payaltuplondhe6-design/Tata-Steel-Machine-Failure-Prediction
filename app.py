
import streamlit as st
import pickle
import gzip
import pandas as pd
import math

st.set_page_config(
    page_title="Tata Steel | Machine Failure Prediction",
    page_icon="🏭",
    layout="wide"
)

# Load trained model
with gzip.open("random_forest_machine_failure.pkl.gz", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Sidebar navigation
st.sidebar.title("🏭 Tata Steel")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Machine Failure"]
)

# HOME PAGE
if page == "Home":

    st.title("🏭 Tata Steel Machine Failure Prediction")

    st.subheader("Predictive Maintenance Using Machine Learning")

    st.write("""
    Welcome to the Machine Failure Prediction application.

    This project uses Machine Learning to predict potential
    machine failures based on operating conditions.

    The objective is to support predictive maintenance,
    reduce unexpected downtime and improve operational efficiency.
    """)

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Models", "3")
    col2.metric("Selected Model", "Random Forest")
    col3.metric("Validation Accuracy", "99%")

    st.divider()

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - Streamlit
    """)

    st.subheader("Machine Learning Workflow")

    st.write("""
    1. Data Cleaning and Preprocessing
    2. Exploratory Data Analysis
    3. Feature Engineering
    4. Model Building
    5. Model Evaluation
    6. Random Forest Model Selection
    7. Streamlit Deployment
    """)

    st.subheader("Model Performance")

    st.write("""
    The selected Random Forest model achieved the following
    validation results for the failure class:
    """)

    st.write("- Precision: 61%")
    st.write("- Recall: 30%")
    st.write("- F1-score: 40%")

    st.info(
        "Note: The model is an academic project prototype. "
        "It may miss actual machine failures and should not "
        "be used as the sole basis for maintenance decisions."
    )

    st.success(
        "Use the Predict Machine Failure section "
        "to test machine operating conditions."
    )

# PREDICTION PAGE
elif page == "Predict Machine Failure":

    st.title("Machine Failure Prediction")

    st.write("Enter the machine operating details below.")

    air_temp = st.number_input(
        "Air Temperature (K)", value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)", value=310.0
    )

    rot_speed = st.number_input(
        "Rotational Speed (rpm)", value=1500
    )

    torque = st.number_input(
        "Torque (Nm)", value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)", value=100
    )

    machine_type = st.selectbox(
        "Machine Type", ["L", "M", "H"]
    )

    if st.button("Predict Machine Failure"):

        temp_difference = process_temp - air_temp

        mechanical_power = (
            torque * rot_speed * (2 * math.pi / 60) / 1000
        )

        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Temperature Difference [K]": temp_difference,
            "Mechanical Power [kW]": mechanical_power,
            "Type_H": 1 if machine_type == "H" else 0,
            "Type_L": 1 if machine_type == "L" else 0,
            "Type_M": 1 if machine_type == "M" else 0
        }])

        numerical_features = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temperature Difference [K]",
            "Mechanical Power [kW]"
        ]

        input_data[numerical_features] = scaler.transform(
            input_data[numerical_features]
        )

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Possible Machine Failure Detected!")
        else:
            st.success("✅ No Machine Failure Predicted.")
