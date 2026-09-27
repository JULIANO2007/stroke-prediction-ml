import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns_stroke.pkl")

# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Stroke Prediction")
st.write("Masukkan data pasien untuk melihat hasil prediksi model.")

st.divider()

# =========================
# INPUT
# =========================

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=120.0,
    value=40.0
)

avg_glucose = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

hypertension = st.radio(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

heart_disease = st.radio(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

ever_married = st.radio(
    "Ever Married",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

residence = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

work = st.selectbox(
    "Work Type",
    [
        "Govt_job",
        "Never_worked",
        "Private",
        "Self-employed",
        "children"
    ]
)

smoking = st.selectbox(
    "Smoking Status",
    [
        "Unknown",
        "formerly smoked",
        "never smoked",
        "smokes"
    ]
)

st.divider()

# =========================
# PREDICT
# =========================

if st.button("🔍 PREDICT STROKE", use_container_width=True):

    # Gender
    gender_value = {
        "Female": 0,
        "Male": 1,
        "Other": 2
    }[gender]

    # Residence
    residence_value = 1 if residence == "Urban" else 0

    # Work type
    work_values = {
        "Govt_job": 1,
        "Never_worked": 1,
        "Private": 1,
        "Self-employed": 1,
        "children": 1
    }

    # Smoking
    smoking_values = {
        "Unknown": 1,
        "formerly smoked": 1,
        "never smoked": 1,
        "smokes": 1
    }

    # =========================
    # BUILD DATA
    # =========================

    data = {
        "gender": gender_value,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "Residence_type": residence_value,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,

        "work_Govt_job":
            1 if work == "Govt_job" else 0,

        "work_Never_worked":
            1 if work == "Never_worked" else 0,

        "work_Private":
            1 if work == "Private" else 0,

        "work_Self-employed":
            1 if work == "Self-employed" else 0,

        "work_children":
            1 if work == "children" else 0,

        "smoking_Unknown":
            1 if smoking == "Unknown" else 0,

        "smoking_formerly smoked":
            1 if smoking == "formerly smoked" else 0,

        "smoking_never smoked":
            1 if smoking == "never smoked" else 0,

        "smoking_smokes":
            1 if smoking == "smokes" else 0
    }

    df = pd.DataFrame([data])

    # Pastikan urutan fitur sama dengan model
    df = df.reindex(columns=feature_columns, fill_value=0)

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    no_stroke = probability[0] * 100
    stroke = probability[1] * 100

    st.divider()

    # =========================
    # RESULT
    # =========================

    if prediction == 1:
        st.error("⚠️ HASIL: STROKE")

    else:
        st.success("✅ HASIL: TIDAK STROKE")

    st.subheader("Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tidak Stroke",
            f"{no_stroke:.2f}%"
        )

    with col2:
        st.metric(
            "Stroke",
            f"{stroke:.2f}%"
        )

    st.progress(int(stroke))

    st.caption(
        "Hasil ini merupakan prediksi dari model machine learning "
        "dan bukan diagnosis medis."
    )
