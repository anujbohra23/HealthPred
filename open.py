import streamlit as st
import joblib
import openai

# Load the trained model
model = joblib.load("Boosting_model.pkl")

# OpenAI API Key (Ensure you set this securely in Streamlit Secrets)
openai.api_key = st.secrets["KEY"]


# Function to get an explanation from GPT-4
def get_explanation(condition):
    prompt = f"""
    Explain the mental health condition "{condition}" in simple terms. 
    Provide:
    1. A brief explanation of the condition.
    2. Common symptoms.
    3. Three coping strategies.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a compassionate mental health expert.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]


# Streamlit UI
st.title("Mental Health Prediction & Guidance")

# Collect user input
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Trans"])
family_history = st.sidebar.selectbox(
    "Family History of Mental Health Issues?", ["Yes", "No"]
)
no_employees = st.sidebar.number_input("Number of Employees", min_value=1, step=1)
benefits = st.sidebar.selectbox(
    "Does your company provide mental health benefits?", ["Yes", "No"]
)
care_options = st.sidebar.selectbox("Access to mental health care?", ["Yes", "No"])
anonymity = st.sidebar.selectbox(
    "Anonymity when seeking mental health help?", ["Yes", "No"]
)
leave = st.sidebar.selectbox("Is mental health leave offered?", ["Yes", "No"])
work_interfere = st.sidebar.selectbox(
    "Work interference due to mental health?", ["Never", "Rarely", "Sometimes", "Often"]
)

# Convert inputs into a format for model prediction
user_data = [
    [
        age,
        gender,
        family_history,
        no_employees,
        benefits,
        care_options,
        anonymity,
        leave,
        work_interfere,
    ]
]
prediction = model.predict(user_data)[0]  # Predict mental health condition

# Display Prediction
st.subheader("Prediction:")
st.write(f"**{prediction}**")

# Get explanation from OpenAI
if st.button("Get Explanation & Coping Strategies"):
    with st.spinner("Fetching expert advice..."):
        explanation = get_explanation(prediction)
    st.subheader("Explanation & Coping Strategies")
    st.write(explanation)
