import streamlit as st
import pandas as pd
import joblib
import os

# Function to load a .pkl file with error handling
def load_pkl_file(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        st.error(f"Error: The file '{filename}' was not found. Please ensure all .pkl files are in the same directory as app.py.")
        st.stop() # Stop the Streamlit app if files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading '{filename}': {e}")
        st.stop()

# Load the trained model and preprocessing objects
rf_model = load_pkl_file('random_forest_model.pkl') # Corrected filename
imputer = load_pkl_file('median_imputer_stroke.pkl')
scaler = load_pkl_file('scaler_stroke.pkl')
feature_columns = load_pkl_file('feature_columns_stroke.pkl') # Corrected filename

# Streamlit App Title
st.title('Stroke Prediction App')
st.write('Enter patient details to predict stroke risk.')

# Input fields for user data
st.header('Patient Information')

# Define the numerical columns that need imputation and scaling
numerical_features = ['age', 'avg_glucose_level', 'bmi']

age = st.slider('Age', 0.08, 82.0, 40.0)
hypertension = st.radio('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.radio('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
avg_glucose_level = st.slider('Average Glucose Level', 55.12, 271.74, 100.0)
bmi = st.slider('BMI', 10.3, 97.6, 25.0)

gender_options = {'Male': 1, 'Female': 0, 'Other': 2}
gender_input = st.selectbox('Gender', list(gender_options.keys()))

ever_married_input = st.radio('Ever Married', ['Yes', 'No'])
work_type_input = st.selectbox('Work Type', ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'])
residence_type_input = st.radio('Residence Type', ['Urban', 'Rural'])
smoking_status_input = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Create a dictionary to hold the raw user inputs, including for OHE
user_input_dict = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender': gender_options[gender_input],
    'ever_married': 1 if ever_married_input == 'Yes' else 0, # Binary encoding
    'Residence_type': 1 if residence_type_input == 'Urban' else 0, # Binary encoding
}

# Initialize all one-hot encoded columns to 0 first
for col in feature_columns:
    if col.startswith('work_') or col.startswith('smoking_'):
        user_input_dict[col] = 0

# Set the value for the selected work_type
if work_type_input == 'Never_worked':
    user_input_dict['work_Never_worked'] = 1
elif work_type_input == 'Private':
    user_input_dict['work_Private'] = 1
elif work_type_input == 'Self-employed':
    user_input_dict['work_Self-employed'] = 1
elif work_type_input == 'children':
    user_input_dict['work_children'] = 1
elif work_type_input == 'Govt_job': # Added Govt_job as it's a valid work_type
    user_input_dict['work_Govt_job'] = 1

# Set the value for the selected smoking_status
if smoking_status_input == 'formerly smoked':
    user_input_dict['smoking_formerly smoked'] = 1
elif smoking_status_input == 'never smoked':
    user_input_dict['smoking_never smoked'] = 1
elif smoking_status_input == 'smokes':
    user_input_dict['smoking_smokes'] = 1
elif smoking_status_input == 'Unknown':
    user_input_dict['smoking_Unknown'] = 1


# Convert to DataFrame, ensuring all feature columns are present and in the correct order
input_df = pd.DataFrame([user_input_dict])

# Reindex to ensure all feature_columns are present and in the correct order
# Fill any missing OHE columns that weren't explicitly set with 0
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Apply imputation and scaling only to numerical features
# Create a temporary copy of the numerical columns for processing
input_numerical_processed = input_df[numerical_features].copy()

# Impute missing values
input_numerical_processed = imputer.transform(input_numerical_processed)

# Scale numerical features
input_numerical_processed = scaler.transform(input_numerical_processed)

# Update the original DataFrame with the processed numerical features
input_df[numerical_features] = input_numerical_processed

# input_df is now ready for prediction
input_scaled = input_df


if st.button('Predict Stroke Risk'):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Based on the provided information, the model predicts a HIGH risk of stroke.')
    else:
        st.success('Based on the provided information, the model predicts a LOW risk of stroke.')

    st.write(f"Probability of No Stroke: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Stroke: {prediction_proba[0][1]:.2f}")

    st.write('---')
    st.write('Disclaimer: This is a predictive model and should not be used as a substitute for professional medical advice.')
