import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io  # Import the io module


# --- 1. Data Collection ---
# Define the questions.  These should match the column names in your data.
questions = {
    "Have you ever had suicidal thoughts": "Have you ever had suicidal thoughts?",
    "Academic Pressure": "Are you experiencing academic pressure?",
    "Financial Stress": "Are you experiencing financial stress?",
    "Work / Study Hours": "Do you have long work/study hours?",
    "Dietary Habits": "Do you have unhealthy dietary habits?",
    "Sleep Duration": "Do you have sleep problems?",
    # Add more questions as needed, matching your data's column names
}

# --- 2. Data Preprocessing ---
# Function to load and preprocess data
def load_and_preprocess_data(file=None):
    """
    Loads and preprocesses the data from a CSV or Excel file.

    Args:
        file (streamlit.UploadedFile, optional): The uploaded file. Defaults to None.

    Returns:
        pd.DataFrame: The preprocessed DataFrame, or None if there's an error.
    """
    try:
        if file is not None:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
        else:
            # Load default data
            csv_data = """Gender,Age,City,Profession,Academic Pressure,Work Pressure,CGPA,Study Satisfaction,Job Satisfaction,Sleep Duration,Dietary Habits,Degree,Have you ever had suicidal thoughts,Work / Study Hours,Financial Stress,Family History of Mental Illness,Depression
Male,23,Visakhapatnam,Student,2,5,8.97,3,0,05-6 hours,Healthy,B.Pharm,No,3,No,No,0
Female,24,Bangalore,Student,2,5,5.9,2,0,06-8 hours,Moderate,BSc,Yes,3,Yes,No,0
Male,21,Srinagar,Student,3,3,7.03,3,0,07-8 hours,Healthy,BA,No,1,Yes,No,0
Female,28,Varanasi,Student,3,3,5.59,2,0,07-8 hours,Moderate,BCA,Yes,4,Yes,No,0
Female,25,Jaipur,Student,2,4,6.13,3,0,05-6 hours,Moderate,M.Tech,Yes,1,No,No,0
Male,29,Pune,Student,2,5,5.7,2,0,Less than 5 hours,Healthy,PhD,No,4,No,No,0
Male,30,Thane,Student,3,4,9.54,4,0,07-8 hours,Healthy,BSc,No,1,No,No,0
Female,30,Chennai,Student,2,4,8.04,4,0,Less than 5 hours,Unhealthy,B.Ed,Yes,12,Yes,Yes,1
Male,28,Nagpur,Student,3,3,9.79,3,0,07-8 hours,Moderate,B.Ed,No,3,No,No,0
Male,31,Nashik,Student,2,3,6.38,3,0,Less than 5 hours,Moderate,LLB,Yes,2,Yes,No,1
Male,24,Nagpur,Student,3,4,6.1,2,0,05-6 hours,Moderate,Class 12,Yes,11,Yes,No,1
Male,33,Vadodara,Student,3,4,7.03,3,0,Less than 5 hours,Healthy,BE,No,10,Yes,No,1
Male,27,Kalyan,Student,5,4,7.04,4,0,Less than 5 hours,Moderate,M.Tech,Yes,10,Yes,No,1
Female,19,Raipur,Student,2,4,8.52,4,0,Less than 5 hours,Unhealthy,Class 12,No,6,Yes,No,0
Female,29,Srinagar,Student,5,4,5.64,2,0,Less than 5 hours,Moderate,Class 12,Yes,4,Yes,No,1
Male,26,Student,3,4,8.58,3,0,More than 8 hours,Moderate,M.Tech,Yes,10,Yes,No,1
Male,25,Nashik,Student,5,5,6.51,2,0,Less than 5 hours,Unhealthy,M.Ed,Yes,2,Yes,No,1
Female,20,Ahmedabad,Student,5,5,7.25,3,0,05-6 hours,Healthy,Class 12,Yes,3,No,No,0
Male,19,Chennai,Student,2,4,7.85,3,0,07-8 hours,Unhealthy,Class 12,No,6,No,No,0
Male,25,Kalyan,Student,3,4,9.93,3,0,05-6 hours,Moderate,B.Ed,No,3,Yes,No,0
Male,29,Kolkata,Student,3,4,8.74,3,0,05-6 hours,Moderate,B.Ed,Yes,1,No,No,0
Male,29,Kolkata,Student,3,4,6.78,3,0,07-8 hours,Moderate,M.Tech,No,1,No,No,0
Female,25,Ahmedabad,Student,3,4,5.57,2,0,More than 8 hours,Unhealthy,MSc,Yes,10,No,No,0
Male,23,Thane,Student,1,4,6.59,4,0,07-8 hours,Healthy,BHM,No,11,No,No,0
Male,18,Bangalore,Student,4,5,7.1,3,0,More than 8 hours,Unhealthy,Class 12,Yes,2,Yes,No,1
Female,20,Mumbai,Student,2,4,8.58,4,0,07-8 hours,Moderate,Class 12,No,2,Yes,No,0
"""
            df = pd.read_csv(io.StringIO(csv_data))

        # Check for required columns (case-insensitive)
        required_columns = [
            "Have you ever had suicidal thoughts", "Academic Pressure", "Financial Stress",
            "Work / Study Hours", "Dietary Habits", "Sleep Duration",
            # Add more columns as needed
        ]
        missing_columns = [col for col in required_columns if col in df.columns]
        if missing_columns:
            #st.error(f"Missing required columns: {', '.join(missing_columns)}.  Please ensure your file contains these columns (case-sensitive).")
            #return None

        # Standardize column names to be exactly as they appear in the data
        df.columns = ["Gender","Age","City","Profession","Academic Pressure","Work Pressure","CGPA","Study Satisfaction","Job Satisfaction","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts","Work / Study Hours","Financial Stress","Family History of Mental Illness","Depression"]

        # Ensure that the depression column is numeric
        if 'Depression' in df.columns and not pd.api.types.is_numeric_dtype(df['Depression']):
            st.error("The 'Depression' column must be numeric (0 or 1). Please check your data.")
            return None

       # Convert specified columns to numeric, raising errors if conversion isn't possible
        for col in required_columns:
            if col in df.columns:
                #st.write(f"Converting column: {col}") #for debugging
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                #check if the values are 0 or 1
                if not set(df[col].unique()).issubset({0, 1}):
                    st.error(f"Column '{col}' must contain only 0 or 1 values. Please check your data.")
                    return None

        #check if the values in 'Depression' column are 0 or 1
        if 'Depression' in df.columns and not set(df['Depression'].unique()).issubset({0, 1}):
            st.error(f"Column 'Depression' must contain only 0 or 1 values. Please check your data.")
            return None
        return df
    except Exception as e:
        st.error(f"An error occurred while loading or preprocessing your data: {e}")
        return None



# --- 3. Model Training ---
def train_model(X_train, y_train):
    """
    Trains a logistic regression model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.

    Returns:
        LogisticRegression: The trained model.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 4. User Interaction (Streamlit App) ---
def main():
    st.title("Depression Assessment Tool")
    st.write("You can upload your own data (CSV or Excel) or use the default data for the assessment.")

    # File upload
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xls", "xlsx"])

    # Load data
    df = load_and_preprocess_data(uploaded_file) # Pass the uploaded file

    if df is not None: # only proceed if a file was uploaded or default data was loaded
        # Separate features (X) and target variable (y)
        try:
            X = df.drop('Depression', axis=1)
            y = df['Depression']
        except KeyError:
            st.error(
                "The 'Depression' column was not found in your data. Please make sure your data includes a column named 'Depression' indicating the target variable."
            )
            return  # Stop processing if 'Depression' column is missing

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Get user responses for prediction
        st.subheader("Enter your responses to the following questions:")
        user_responses = {}
        for key, question in questions.items():
            # st.write(f"Column name: {key.lower()}")  # for debugging
            if key in X.columns:
                user_responses[key] = st.selectbox(question, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            else:
                st.write(f"Question '{question}' not found in the uploaded data. Skipping.")
                user_responses[key] = 0  # set default

        # Convert user responses to a DataFrame
        user_df = pd.DataFrame([user_responses])

        # Ensure the user_df has the same columns as the training data, X_train
        user_df = user_df[X_train.columns]

        # --- 5. Prediction and Interpretation ---
        if st.button("Assess"):
            # Make a prediction using the trained model
            prediction = model.predict(user_df)[0]

            # Display the result
            st.subheader("Assessment Result:")
            if prediction == 1:
                st.error(
                    "Based on your responses, you may be experiencing symptoms of depression. It is important to seek professional help."
                )
            else:
                st.success(
                    "Based on your responses, you are less likely to be experiencing depression. However, if you are feeling distressed, consider seeking support."
                )

            st.write(
                "Please remember this is not a substitute for a professional diagnosis. If you have concerns about your mental health, please consult a healthcare provider."
            )

            # Show model performance
            st.subheader("Model Performance:")
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
            st.pyplot(plt)

if __name__ == "__main__":
    main()
