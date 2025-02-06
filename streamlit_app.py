import streamlit as st
import joblib
import numpy as np

# Load the trained model
clf = joblib.load("Boosting_model.pkl")

# Streamlit app
st.title("Self-Analysis Mental Health Predictor")
st.write(
    "Enter the details below to predict if you may require mental health treatment."
)

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Trans"])
family_history = st.selectbox("Family History of Mental Health Issues?", ["Yes", "No"])
no_employees = st.number_input(
    "Number of Employees at Workplace", min_value=1, max_value=10000, step=1
)
benefits = st.selectbox(
    "Does your workplace provide mental health benefits?", ["Yes", "No"]
)
care_options = st.selectbox(
    "Are mental health care options available at work?", ["Yes", "No"]
)
anonymity = st.selectbox("Is seeking mental health help anonymous?", ["Yes", "No"])
leave = st.selectbox("Is leave available for mental health reasons?", ["Yes", "No"])
work_interfere = st.selectbox(
    "Does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"]
)

# Encode categorical inputs
gender_map = {"Male": 0, "Female": 1, "Trans": 2}
family_map = {"No": 0, "Yes": 1}
binary_map = {"No": 0, "Yes": 1}
interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

input_features = np.array(
    [
        age,
        gender_map[gender],
        family_map[family_history],
        no_employees,
        binary_map[benefits],
        binary_map[care_options],
        binary_map[anonymity],
        binary_map[leave],
        interfere_map[work_interfere],
    ]
).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = clf.predict(input_features)[0]
    if prediction == 1:
        st.error(
            "You may need mental health treatment. Please consider seeking professional help."
        )
    else:
        st.success("No mental health treatment needed based on this input.")
