import streamlit as st
import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# LangChain Model
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant to assist users in solving their mental health issues. Please respond to user queries.",
        ),
        ("user", "Question:{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


st.title("Mental Health Assistant")

# Section 1: Question Answering
st.header("Questions")
input_text = st.text_input("Ask a question:")
if input_text:
    st.write(chain.invoke({"question": input_text}))

# Section 2: Mental Health Predictor
st.header("Health Indicator")
st.write(
    "Enter the details below to predict if you may require mental health treatment."
)

# Load the trained model
clf = joblib.load("RandomForest_model.pkl")

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

# Encoding
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

# Prediction
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
    explainer = shap.Explainer(clf)
    shap_values = explainer(input_features)
    shap_values.feature_names = feature_names

    st.subheader("Feature Importance for This Prediction")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[..., 1][0])
    st.pyplot(fig)

    st.subheader("Overall Feature Importance")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(
        shap_values[..., 1], input_features, feature_names=feature_names, show=False
    )
    st.pyplot(fig2)
