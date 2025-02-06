import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

clf = joblib.load("RandomForest_model.pkl")

st.title("Self-Analysis Mental Health Predictor")
st.write(
    "Enter the details below to predict if you may require mental health treatment."
)

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

# Encodings
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

# Predictions
if st.button("Predict"):
    prediction = clf.predict(input_features)[0]

    if prediction == 1:
        st.error(
            "You may need mental health treatment. Please consider seeking professional help."
        )
    else:
        st.success("No mental health treatment needed based on this input.")

    # SHAP Explanation
feature_names = [
    "Age",
    "Gender",
    "Family History",
    "No. of Employees",
    "Workplace Benefits",
    "Care Options",
    "Anonymity",
    "Mental Health Leave",
    "Work Interference",
]

# SHAP Explanation - for explainging black box model behaviour
explainer = shap.Explainer(clf)
shap_values = explainer(input_features)


shap_values.feature_names = feature_names

st.subheader("Feature Importance for This Prediction")
shap_values_class_1 = shap_values[..., 1]
fig, ax = plt.subplots()
shap.waterfall_plot(shap_values_class_1[0])
st.pyplot(fig)

# SHAP Summary Plot (Global Feature Importance)
st.subheader("Overall Feature Importance")
fig2, ax2 = plt.subplots()
shap.summary_plot(
    shap_values_class_1, input_features, feature_names=feature_names, show=False
)
st.pyplot(fig2)
