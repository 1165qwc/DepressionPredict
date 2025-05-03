import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Collection ---
# Define the questions
questions = {
    "suicidal_thoughts": "Have you ever had suicidal thoughts?",
    "academic_pressure": "Are you experiencing academic pressure?",
    "financial_stress": "Are you experiencing financial stress?",
    "work_study_hours": "Do you have long work/study hours?",
    "unhealthy_diet": "Do you have unhealthy dietary habits?",
    "sleep_problems": "Do you have sleep problems?",
    "loss_of_interest": "Have you experienced a loss of interest in activities?",
    "feeling_down": "Do you often feel down or hopeless?",
    "fatigue": "Do you often feel tired or fatigued?",
    "appetite_changes": "Have you experienced changes in appetite or weight?",
    "concentration_difficulties": "Do you have trouble concentrating?",
    "worthlessness_guilt": "Do you feel worthless or excessively guilty?",
}

# --- 2. Data Preprocessing (Simulated Data) ---
# Create a function to simulate data (for demonstration purposes)
def generate_simulated_data(num_samples=100):
    """
    Generates simulated data for depression assessment.  The data is
    designed to show some correlation between the questions and the
    likelihood of depression, but it's not a substitute for real data.

    Args:
        num_samples (int): The number of simulated data points to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated data, with
                      columns corresponding to the questions and a 'depression'
                      column (0 for no, 1 for yes).
    """
    np.random.seed(42)  # Ensure consistent results

    data = {
        "suicidal_thoughts": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "academic_pressure": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        "financial_stress": np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        "work_study_hours": np.random.choice([0, 1], num_samples, p=[0.4, 0.6]),
        "unhealthy_diet": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "sleep_problems": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        "loss_of_interest": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "feeling_down": np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        "fatigue": np.random.choice([0, 1], num_samples, p=[0.5, 0.5]),
        "appetite_changes": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        "concentration_difficulties": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "worthlessness_guilt": np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
    }

    # Simulate 'depression' outcome with some dependencies on the input features
    depression = (
        0.2 * data["suicidal_thoughts"] +
        0.2 * data["academic_pressure"] +
        0.2 * data["financial_stress"] +
        0.1 * data["work_study_hours"] +
        0.1 * data["unhealthy_diet"] +
        0.1 * data["sleep_problems"] +
        0.2 * data["loss_of_interest"] +
        0.2 * data["feeling_down"] +
        0.1 * data["fatigue"] +
        0.1 * data["appetite_changes"] +
        0.1 * data["concentration_difficulties"] +
        0.2 * data["worthlessness_guilt"] +
        np.random.normal(0, 0.3, num_samples)  # Add some noise
    ).round().astype(int)  # Round to 0 or 1

    depression = np.clip(depression, 0, 1) # Ensure values are 0 or 1


    df = pd.DataFrame(data)
    df['depression'] = depression
    return df

# Generate the simulated data
df = generate_simulated_data(num_samples=100)

# Separate features (X) and target variable (y)
X = df.drop('depression', axis=1)
y = df['depression']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Model Training ---
# Train a logistic regression model (a simple model for demonstration)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# --- 4. User Interaction (Streamlit App) ---
def main():
    st.title("Depression Assessment Tool")
    st.write("Please answer the following questions to assess your likelihood of depression.")

    # Collect user responses.  Initialize the responses dictionary.
    user_responses = {}
    for key, question in questions.items():
        user_responses[key] = st.selectbox(question, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Convert user responses to a DataFrame
    user_df = pd.DataFrame([user_responses])

    # --- 5. Prediction and Interpretation ---
    if st.button("Assess"):
        # Make a prediction using the trained model
        prediction = model.predict(user_df)[0]

        # Display the result
        st.subheader("Assessment Result:")
        if prediction == 1:
            st.error("Based on your responses, you may be experiencing symptoms of depression. It is important to seek professional help.")
        else:
            st.success("Based on your responses, you are less likely to be experiencing depression. However, if you are feeling distressed, consider seeking support.")

        st.write("Please remember this is not a substitute for a professional diagnosis. If you have concerns about your mental health, please consult a healthcare provider.")

        # Show model performance
        st.subheader("Model Performance (on simulated data):")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Display classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Display confusion matrix
        st.subheader("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)  # Use st.pyplot() to display the Matplotlib plot

if __name__ == "__main__":
    main()
