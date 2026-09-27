import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("random_forest_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}")
    st.stop()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Stroke Prediction App")
st.write("Enter patient information to predict the model's stroke risk.")


# =========================
# INPUT
# =========================
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value=0.08,
        max_value=82.0,
        value=40.0
    )

    gender_input = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    hypertension = st.radio(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart_disease = st.radio(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    ever_married_input = st.radio(
        "Ever Married",
        ["Yes", "No"]
    )


with col2:
    avg_glucose_level = st.slider(
        "Average Glucose Level",
        min_value=55.12,
        max_value=271.74,
        value=100.0
    )

    bmi = st.slider(
        "BMI",
        min_value=10.3,
        max_value=97.6,
        value=25.0
    )

    work_type_input = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "children",
            "Govt_job",
            "Never_worked"
        ]
    )

    residence_type_input = st.radio(
        "Residence Type",
        ["Urban", "Rural"]
    )

    smoking_status_input = st.selectbox(
        "Smoking Status",
        [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown"
        ]
    )


# =========================
# ENCODING
# =========================

gender_mapping = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

user_input_dict = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender_mapping[gender_input],
    "ever_married": 1 if ever_married_input == "Yes" else 0,
    "Residence_type": 1 if residence_type_input == "Urban" else 0
}


# =========================
# ONE-HOT ENCODING
# =========================

# Set all work columns to 0
work_columns = [
    "work_Govt_job",
    "work_Never_worked",
    "work_Private",
    "work_Self-employed",
    "work_children"
]

for col in work_columns:
    user_input_dict[col] = 0


# Set selected work type to 1
if work_type_input == "Govt_job":
    user_input_dict["work_Govt_job"] = 1

elif work_type_input == "Never_worked":
    user_input_dict["work_Never_worked"] = 1

elif work_type_input == "Private":
    user_input_dict["work_Private"] = 1

elif work_type_input == "Self-employed":
    user_input_dict["work_Self-employed"] = 1

elif work_type_input == "children":
    user_input_dict["work_children"] = 1


# Set all smoking columns to 0
smoking_columns = [
    "smoking_Unknown",
    "smoking_formerly smoked",
    "smoking_never smoked",
    "smoking_smokes"
]

for col in smoking_columns:
    user_input_dict[col] = 0


# Set selected smoking status to 1
if smoking_status_input == "Unknown":
    user_input_dict["smoking_Unknown"] = 1

elif smoking_status_input == "formerly smoked":
    user_input_dict["smoking_formerly smoked"] = 1

elif smoking_status_input == "never smoked":
    user_input_dict["smoking_never smoked"] = 1

elif smoking_status_input == "smokes":
    user_input_dict["smoking_smokes"] = 1


# =========================
# CREATE DATAFRAME
# =========================

input_df = pd.DataFrame([user_input_dict])

# Make sure feature order is EXACTLY the same as training
input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict Stroke Risk", use_container_width=True):

    try:
        prediction = model.predict(input_df)[0]

        prediction_proba = model.predict_proba(input_df)[0]

        stroke_probability = prediction_proba[1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Model Prediction: Stroke")
        else:
            st.success("✅ Model Prediction: No Stroke")

        st.metric(
            "Stroke Probability",
            f"{stroke_probability * 100:.2f}%"
        )

        st.write(
            f"Probability of No Stroke: "
            f"{prediction_proba[0] * 100:.2f}%"
        )

        st.write(
            f"Probability of Stroke: "
            f"{prediction_proba[1] * 100:.2f}%"
        )

        st.info(
            "Disclaimer: This is a machine learning model prediction "
            "and is not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
