
import streamlit as st
import pickle
import gzip
import pandas as pd
import math

# Load trained Random Forest model
with gzip.open("random_forest_machine_failure.pkl.gz", "rb") as file:
    model = pickle.load(file)

# Load StandardScaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Application title
st.title("Tata Steel Machine Failure Prediction")

st.write(
    "Enter machine operating details to predict possible machine failure."
)

# Input fields
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

# Prediction button
if st.button("Predict Machine Failure"):

    # Feature engineering
    temp_difference = process_temp - air_temp

    mechanical_power = (
        torque * rot_speed * (2 * math.pi / 60) / 1000
    )

    # Create input DataFrame
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

    # Numerical features to scale
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature Difference [K]",
        "Mechanical Power [kW]"
    ]

    # Apply saved scaler
    input_data[numerical_features] = scaler.transform(
        input_data[numerical_features]
    )

    # Predict
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("Possible Machine Failure Detected!")
    else:
        st.success("No Machine Failure Predicted.")
