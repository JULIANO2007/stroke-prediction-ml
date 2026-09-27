import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🫀",
    layout="wide"
)


# =========================================================
# CLEAN STYLE
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    font-size: 42px !important;
}

h2 {
    font-size: 28px !important;
}

h3 {
    font-size: 22px !important;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

div[data-testid="stMetricValue"] {
    font-size: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL & PREPROCESSING
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("random_forest_model.pkl")

    feature_columns = joblib.load(
        "feature_columns_stroke.pkl"
    )

    imputer = joblib.load(
        "median_imputer_stroke.pkl"
    )

    scaler = joblib.load(
        "scaler_stroke.pkl"
    )

    return model, feature_columns, imputer, scaler


try:

    model, feature_columns, imputer, scaler = load_models()

except FileNotFoundError as e:

    st.error(
        f"File tidak ditemukan: {e}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"Gagal memuat model: {e}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🫀 Stroke Prediction")

st.write(
    "Enter patient information to obtain a prediction "
    "from the trained Random Forest model."
)

st.caption(
    "This application is for machine learning demonstration "
    "and is not a medical diagnosis."
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.header("Patient Information")


col1, col2 = st.columns(2)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.slider(
        "Age",
        min_value=0.08,
        max_value=82.0,
        value=40.0,
        step=0.1
    )

    gender_input = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )

    hypertension = st.radio(
        "Hypertension",
        [0, 1],
        horizontal=True,
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    heart_disease = st.radio(
        "Heart Disease",
        [0, 1],
        horizontal=True,
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    ever_married_input = st.radio(
        "Ever Married",
        ["Yes", "No"],
        horizontal=True
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    avg_glucose_level = st.slider(
        "Average Glucose Level",
        min_value=55.12,
        max_value=271.74,
        value=100.0,
        step=0.01
    )

    bmi = st.slider(
        "BMI",
        min_value=10.3,
        max_value=97.6,
        value=25.0,
        step=0.1
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
        ["Urban", "Rural"],
        horizontal=True
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


# =========================================================
# ENCODING
# =========================================================

gender_mapping = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}


user_input_dict = {

    "gender":
        gender_mapping[gender_input],

    "age":
        age,

    "hypertension":
        hypertension,

    "heart_disease":
        heart_disease,

    "ever_married":
        1 if ever_married_input == "Yes" else 0,

    "Residence_type":
        1 if residence_type_input == "Urban" else 0,

    "avg_glucose_level":
        avg_glucose_level,

    "bmi":
        bmi
}


# =========================================================
# WORK TYPE ONE-HOT ENCODING
# =========================================================

work_columns = [

    "work_Govt_job",

    "work_Never_worked",

    "work_Private",

    "work_Self-employed",

    "work_children"
]


for col in work_columns:

    user_input_dict[col] = 0


if work_type_input == "Govt_job":

    user_input_dict[
        "work_Govt_job"
    ] = 1

elif work_type_input == "Never_worked":

    user_input_dict[
        "work_Never_worked"
    ] = 1

elif work_type_input == "Private":

    user_input_dict[
        "work_Private"
    ] = 1

elif work_type_input == "Self-employed":

    user_input_dict[
        "work_Self-employed"
    ] = 1

elif work_type_input == "children":

    user_input_dict[
        "work_children"
    ] = 1


# =========================================================
# SMOKING ONE-HOT ENCODING
# =========================================================

smoking_columns = [

    "smoking_Unknown",

    "smoking_formerly smoked",

    "smoking_never smoked",

    "smoking_smokes"
]


for col in smoking_columns:

    user_input_dict[col] = 0


if smoking_status_input == "Unknown":

    user_input_dict[
        "smoking_Unknown"
    ] = 1

elif smoking_status_input == "formerly smoked":

    user_input_dict[
        "smoking_formerly smoked"
    ] = 1

elif smoking_status_input == "never smoked":

    user_input_dict[
        "smoking_never smoked"
    ] = 1

elif smoking_status_input == "smokes":

    user_input_dict[
        "smoking_smokes"
    ] = 1


# =========================================================
# CREATE DATAFRAME
# =========================================================

input_df = pd.DataFrame(
    [user_input_dict]
)


# =========================================================
# FEATURE ORDER
# =========================================================

input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# =========================================================
# PREPROCESSING
# IMPORTANT:
# SAME PROCESS AS TRAINING
# =========================================================

numerical_features = [

    "age",

    "avg_glucose_level",

    "bmi"
]


# Median imputation
input_numerical = input_df[
    numerical_features
].copy()


input_numerical = imputer.transform(
    input_numerical
)


# Standard scaling
input_numerical = scaler.transform(
    input_numerical
)


# Put processed values back
input_df[
    numerical_features
] = input_numerical


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

if st.button(
    "🔍 Predict Stroke Risk",
    use_container_width=True
):

    try:

        # Model prediction
        prediction = model.predict(
            input_df
        )[0]


        # Prediction probability
        probabilities = model.predict_proba(
            input_df
        )[0]


        no_stroke_probability = probabilities[0]

        stroke_probability = probabilities[1]


        # =================================================
        # RESULT
        # =================================================

        st.header("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Model Prediction: Stroke"
            )

        else:

            st.success(
                "✅ Model Prediction: No Stroke"
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader(
            "Model Probability"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "No Stroke",
                f"{no_stroke_probability * 100:.2f}%"
            )


        with result_col2:

            st.metric(
                "Stroke",
                f"{stroke_probability * 100:.2f}%"
            )


        # =================================================
        # PROBABILITY BAR
        # =================================================

        st.write(
            "Probability distribution:"
        )


        probability_df = pd.DataFrame(
            {
                "Class": [
                    "No Stroke",
                    "Stroke"
                ],

                "Probability": [
                    no_stroke_probability,
                    stroke_probability
                ]
            }
        )


        st.bar_chart(
            probability_df.set_index("Class")
        )


        # =================================================
        # EXPLANATION
        # =================================================

        st.info(
            "The final prediction follows the class "
            "predicted by the trained Random Forest model. "
            "The displayed percentages represent the model's "
            "predicted class probabilities and are not clinical "
            "risk percentages."
        )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            "Disclaimer: This application is a machine "
            "learning demonstration and must not be used "
            "as a medical diagnosis or as a substitute "
            "for professional medical advice."
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
