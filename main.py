import streamlit as st

def calculate_depression_score(data):
    """
    Calculates a depression score based on the provided data.

    Args:
        data (dict): A dictionary containing the input data with the following keys:
            "suicidal_thoughts" (int/float):  Score for suicidal thoughts.
            "academic_pressure" (int/float): Score for academic pressure.
            "financial_stress" (int/float): Score for financial stress.
            "work_study_hours" (int/float): Score for work/study hours.
            "unhealthy_diet" (int/float): Score for unhealthy diet.
            "sleep_problems" (int/float): Score for sleep problems.
            "Study Satisfaction" (int/float): Score for study satisfaction.

    Returns:
        float: The calculated depression score.
    """
    score = (
        0.2 * data["suicidal_thoughts"] +
        0.2 * data["academic_pressure"] +
        0.2 * data["financial_stress"] +
        0.1 * data["work_study_hours"] +
        0.1 * data["unhealthy_diet"] +
        0.1 * data["sleep_problems"] +
        0.2 * data["Study Satisfaction"]
    )
    return score

def get_depression_level(score):
    """
    Categorizes the depression level based on the calculated score.

    Args:
        score (float): The depression score.

    Returns:
        str: The depression level category (e.g., "Not Depressed", "Mild Depression", etc.).
    """
    if score < 3:
        return "Not Depressed"
    elif 3 <= score < 5:
        return "Mild Depression"
    elif 5 <= score < 7:
        return "Moderate Depression"
    elif 7 <= score < 9:
        return "Moderately Severe Depression"
    else:
        return "Severe Depression"
    
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Depression Assessment Tool")
    st.write("Please answer the following questions to assess your depression level.")

    # Input fields for the depression factors.  Using a slider
    # Added some explanation text to the questions.
    suicidal_thoughts = st.slider(
        "Suicidal Thoughts (0-10):  0 = Never, 10 = Very Frequently", min_value=0, max_value=10, value=0
    )
    academic_pressure = st.slider(
        "Academic Pressure (0-10): 0 = None, 10 = Extreme Pressure", min_value=0, max_value=10, value=0
    )
    financial_stress = st.slider(
        "Financial Stress (0-10): 0 = No Stress, 10 = Severe Stress", min_value=0, max_value=10, value=0
    )
    work_study_hours = st.slider(
        "Work/Study Hours per Week (0-80):", min_value=0, max_value=80, value=0
    )
    unhealthy_diet = st.slider(
        "Unhealthy Diet (0-10): 0 = Very Healthy, 10 = Very Unhealthy", min_value=0, max_value=10, value=0
    )
    sleep_problems = st.slider(
        "Sleep Problems (0-10): 0 = No Problems, 10 = Severe Problems", min_value=0, max_value=10, value=0
    )
    study_satisfaction = st.slider(
        "Study Satisfaction (0-10): 0 = Very Dissatisfied, 10 = Very Satisfied", min_value=0, max_value=10, value=5
    )

    # Store the data in a dictionary
    data = {
        "suicidal_thoughts": suicidal_thoughts,
        "academic_pressure": academic_pressure,
        "financial_stress": financial_stress,
        "work_study_hours": work_study_hours,
        "unhealthy_diet": unhealthy_diet,
        "sleep_problems": sleep_problems,
        "Study Satisfaction": study_satisfaction,
    }

    # Calculate and display the result when the user clicks the button.
    if st.button("Calculate Depression Level"):
        score = calculate_depression_score(data)
        level = get_depression_level(score)

        st.subheader("Results:")
        st.write(f"Depression Score: {score:.2f}")  # limit to 2 decimal places
        st.write(f"Depression Level: {level}")

        # Provide some basic advice/recommendations.  Keep it general.
        st.write("") # add a newline for space
        st.write("It's important to remember that this is a simplified assessment and not a substitute for a professional diagnosis.")
        if level in ["Moderate Depression", "Moderately Severe Depression", "Severe Depression"]:
            st.write("It's strongly recommended that you seek help from a qualified healthcare professional.")
            st.write("Here are some general resources that may be helpful:")
            st.write("-  [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/): 988")
            st.write("-  [The Crisis Text Line](https://www.crisistextline.org/): Text HOME to 741741")
            st.write("-  Your university's counseling center (if applicable)")
        else:
            st.write("Congratulation, you are not depressed at all, keep up with that")

if __name__ == "__main__":
    main()
