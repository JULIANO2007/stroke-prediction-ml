import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f7f9fc;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #eef4ff 100%
        );
        padding: 32px;
        border-radius: 22px;
        border: 1px solid #e4eaf3;
        margin-bottom: 28px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        color: #172033;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #667085;
        margin-top: 10px;
        line-height: 1.6;
    }

    /* Section title */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #172033;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #667085;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e4eaf3;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.035);
        margin-bottom: 20px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4eaf3;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 750;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 13px;
        font-size: 17px;
        font-weight: 700;
        border: none;
    }

    /* Divider */
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
        border-color: #e4eaf3;
    }

    /* Info box */
    .info-card {
        background: #f0f6ff;
        border: 1px solid #cfe0ff;
        border-radius: 16px;
        padding: 18px 20px;
        color: #344054;
        line-height: 1.6;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff9e8;
        border: 1px solid #f2df9b;
        border-radius: 16px;
        padding: 18px 20px;
        color: #594b20;
        line-height: 1.6;
        margin-top: 25px;
    }

    /* Mobile */
    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 24px;
        }

        .hero-title {
            font-size: 31px;
        }

        .hero-subtitle {
            font-size: 15px;
        }

        .section-title {
            font-size: 22px;
        }
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
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
        f"Required model file was not found: {e}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"Unable to load the model files: {e}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🫀 Stroke Prediction
    </div>

    <div class="hero-subtitle">
        Enter the required patient information to obtain
        a prediction from a trained Random Forest machine
        learning model.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Please complete the information below before running '
    'the prediction.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT COLUMNS
# =========================================================

left_col, right_col = st.columns(
    2,
    gap="large"
)


# =========================================================
# LEFT COLUMN
# =========================================================

with left_col:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("👤 Basic Information")

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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right_col:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Health & Lifestyle")

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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# ENCODING
# =========================================================

gender_mapping = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}


user_input = {

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
# WORK TYPE ENCODING
# =========================================================

work_columns = [
    "work_Govt_job",
    "work_Never_worked",
    "work_Private",
    "work_Self-employed",
    "work_children"
]

for column in work_columns:
    user_input[column] = 0


work_mapping = {
    "Govt_job": "work_Govt_job",
    "Never_worked": "work_Never_worked",
    "Private": "work_Private",
    "Self-employed": "work_Self-employed",
    "children": "work_children"
}

user_input[
    work_mapping[work_type_input]
] = 1


# =========================================================
# SMOKING ENCODING
# =========================================================

smoking_columns = [
    "smoking_Unknown",
    "smoking_formerly smoked",
    "smoking_never smoked",
    "smoking_smokes"
]

for column in smoking_columns:
    user_input[column] = 0


smoking_mapping = {
    "Unknown": "smoking_Unknown",
    "formerly smoked": "smoking_formerly smoked",
    "never smoked": "smoking_never smoked",
    "smokes": "smoking_smokes"
}

user_input[
    smoking_mapping[smoking_status_input]
] = 1


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame(
    [user_input]
)


# =========================================================
# ENSURE FEATURE ORDER
# =========================================================

input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# =========================================================
# PREPROCESSING
# =========================================================

numerical_features = [
    "age",
    "avg_glucose_level",
    "bmi"
]

numeric_input = input_df[
    numerical_features
].copy()


try:

    # Median imputation
    numeric_input = imputer.transform(
        numeric_input
    )

    # Standard scaling
    numeric_input = scaler.transform(
        numeric_input
    )

except Exception as e:

    st.error(
        "Preprocessing failed. "
        "Please make sure the preprocessing files "
        "were created with the same scikit-learn version "
        "used by the application."
    )

    st.stop()


# Put transformed numerical values back
input_df.loc[
    :,
    numerical_features
] = numeric_input


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Run the trained Random Forest model using the '
    'information provided above.'
    '</div>',
    unsafe_allow_html=True
)


predict_button = st.button(
    "🔍 Predict",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # Model prediction
        prediction = model.predict(
            input_df
        )[0]

        # Model probabilities
        probabilities = model.predict_proba(
            input_df
        )[0]

        # Find probability based on model classes
        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        no_stroke_probability = class_probabilities.get(
            0,
            0
        )

        stroke_probability = class_probabilities.get(
            1,
            0
        )


        # =================================================
        # RESULT HEADER
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            'Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # MAIN RESULT
        # =================================================

        if prediction == 1:

            st.error(
                "⚠️ Model Prediction: Stroke"
            )

        else:

            st.success(
                "✅ Model Prediction: No Stroke"
            )


        # =================================================
        # PROBABILITY METRICS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            'Model Probability'
            '</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(
            2,
            gap="large"
        )


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
        # PROBABILITY VISUALIZATION
        # =================================================

        st.markdown(
            '<div class="section-title">'
            'Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "No Stroke",
                    "Stroke"
                ],

                "Probability": [
                    no_stroke_probability * 100,
                    stroke_probability * 100
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Class"),
            height=300
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.markdown(
            '<div class="info-card">'
            '<strong>How to read the result</strong><br><br>'
            'The prediction shown above comes directly from '
            'the trained Random Forest model. The percentages '
            'represent the model&#39;s predicted class '
            'probabilities based on the input features. '
            'They are not clinical risk percentages and '
            'should not be interpreted as a medical diagnosis.'
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.markdown(
            '<div class="disclaimer">'
            '<strong>⚠️ Important</strong><br><br>'
            'This application is a machine learning '
            'demonstration. It is not intended to diagnose, '
            'treat, or replace professional medical advice.'
            '</div>',
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
