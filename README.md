# Mental Health Assistance and Prediction App

## Overview

This is a Streamlit-based web application that provides two functionalities:

1. **Mental Health Q&A Chatbot** – Uses OpenAI’s GPT-3.5-turbo to answer user queries related to mental health.
2. **Mental Health Self-Assessment** – A machine learning model predicts whether a user may need mental health treatment based on their input.

## Features

- **Conversational Q&A:** Users can ask mental health-related questions and receive AI-generated responses.
- **Self-Assessment Form:** Users enter personal and workplace details to get a prediction on whether they might need mental health treatment.
- **Machine Learning Prediction:** A trained Random Forest model predicts mental health treatment needs.
- **SHAP Explanation:** The model’s decision is explained using SHAP visualizations for transparency.

## Technologies Used

- **Python**
- **Streamlit** – For building the interactive web application.
- **OpenAI API** – For the Q&A chatbot functionality.
- **LangChain** – For handling the chatbot prompts and responses.
- **Scikit-learn** – For the trained Random Forest classifier.
- **SHAP** – For model explainability.
- **Joblib** – For saving and loading the trained model.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anujbohra23/HealthPred
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the App

Start the Streamlit app by running:

```bash
streamlit run app.py
```

This will launch the application in your web browser.

## Usage

1. **Mental Health Q&A**

   - Enter your mental health-related question in the search box.
   - The AI assistant will generate a response based on your input.

2. **Mental Health Self-Assessment**
   - Fill in your details like age, gender, workplace conditions, etc.
   - Click the **Predict** button to see if you may need mental health treatment.
   - SHAP visualizations explain the importance of each input factor.

## Model Details

- The prediction model is a **Random Forest Classifier** trained on a mental health survey dataset.
- Features used for prediction include age, gender, family history, workplace conditions, and mental health policies.
- SHAP is used to provide interpretability into how the model makes predictions.

## Future Improvements

- **Expand Q&A Database:** Fine-tune a model specifically for mental health support.
- **Improve Model Accuracy:** Use more advanced ML models like deep learning-based classifiers.
- **Add More Features:** Include additional factors for a more accurate assessment.

## License

This project is licensed under the MIT License.

## Contributors

- [Anuj Bohra](https://github.com/anujbohra23)
