import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stroke AI Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS — 3D / GLASS UI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 180, 255, 0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(150, 70, 255, 0.18), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(0, 255, 180, 0.10), transparent 35%),
        linear-gradient(135deg, #07111f 0%, #0b1630 50%, #10152b 100%);
    color: white;
}


/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}


/* Hero */

.hero {
    padding: 35px;
    border-radius: 28px;
    margin-bottom: 30px;

    background:
        linear-gradient(
            135deg,
            rgba(25, 118, 210, 0.28),
            rgba(126, 87, 194, 0.25)
        );

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.45),
        inset 0 1px 1px rgba(255,255,255,0.12);

    transform: perspective(1000px) rotateX(1deg);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #66d9ff,
        #b58cff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #b8c7dc;
    font-size: 17px;
}


/* Cards */

.card {
    padding: 25px;
    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.09),
            rgba(255,255,255,0.035)
        );

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 20px 40px rgba(0,0,0,0.35),
        inset 0 1px 1px rgba(255,255,255,0.10);

    transition: all 0.25s ease;
}

.card:hover {
    transform:
        translateY(-5px)
        perspective(900px)
        rotateX(1deg);

    box-shadow:
        0 30px 60px rgba(0,0,0,0.45),
        0 0 30px rgba(75, 190, 255, 0.08);
}


/* Section title */

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
}


/* Result */

.result-card {
    padding: 30px;
    margin-top: 25px;
    border-radius: 25px;

    background:
        linear-gradient(
            145deg,
            rgba(17, 28, 55, 0.95),
            rgba(21, 35, 67, 0.85)
        );

    border: 1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.5),
        inset 0 1px 1px rgba(255,255,255,0.08);
}


/* Metric */

.metric-card {
    text-align: center;
    padding: 25px;
    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(0, 170, 255, 0.16),
            rgba(130, 70, 255, 0.12)
        );

    border: 1px solid rgba(100,200,255,0.18);

    box-shadow:
        0 15px 35px rgba(0,0,0,0.35);
}

.metric-value {
    font-size: 38px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #5ee7ff,
        #a98cff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* Button */

.stButton > button {
    width: 100%;
    border-radius: 16px;
    border: none;

    padding: 15px;

    font-size: 17px;
    font-weight: 700;

    color: white;

    background:
        linear-gradient(
            135deg,
            #008cff,
            #7c4dff
        );

    box-shadow:
        0 12px 25px rgba(50,100,255,0.35);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);

    box-shadow:
        0 18px 35px rgba(80,120,255,0.45);
}


/* Inputs */

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.07);
    border-radius: 12px;
}

.stSlider {
    padding-top: 5px;
}


/* Footer */

.footer {
    text-align: center;
    margin-top: 40px;
    color: #7f91aa;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load("random_forest_model.pkl")

    feature_columns = joblib.load(
        "feature_columns_stroke.pkl"
    )

    scaler = joblib.load(
        "scaler_stroke.pkl"
    )

except FileNotFoundError as e:

    st.error(
        f"❌ File tidak ditemukan: {e}"
    )

    st.info(
        "Pastikan file .pkl berada di folder yang sama dengan app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Gagal memuat model: {e}"
    )

    st.stop()


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🫀 Stroke AI Prediction
</div>

<div class="hero-subtitle">
Machine Learning powered stroke prediction system
</div>

<div style="
margin-top:15px;
color:#8fa8c5;
font-size:14px;
">
Random Forest Classifier • Interactive Prediction • AI Assisted
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2, gap="large")


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    age = st.slider(
        "🎂 Age",
        min_value=0.08,
        max_value=82.0,
        value=40.0
    )

    gender_input = st.selectbox(
        "⚥ Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )

    hypertension = st.radio(
        "🩺 Hypertension",
        [0, 1],
        horizontal=True,
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    heart_disease = st.radio(
        "❤️ Heart Disease",
        [0, 1],
        horizontal=True,
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    ever_married_input = st.radio(
        "💍 Ever Married",
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

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    avg_glucose_level = st.slider(
        "🧪 Average Glucose Level",
        min_value=55.12,
        max_value=271.74,
        value=100.0
    )

    bmi = st.slider(
        "⚖️ BMI",
        min_value=10.3,
        max_value=97.6,
        value=25.0
    )

    work_type_input = st.selectbox(
        "💼 Work Type",
        [
            "Private",
            "Self-employed",
            "children",
            "Govt_job",
            "Never_worked"
        ]
    )

    residence_type_input = st.radio(
        "🏠 Residence Type",
        ["Urban", "Rural"],
        horizontal=True
    )

    smoking_status_input = st.selectbox(
        "🚬 Smoking Status",
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


user_input_dict = {

    "age": age,

    "hypertension": hypertension,

    "heart_disease": heart_disease,

    "avg_glucose_level": avg_glucose_level,

    "bmi": bmi,

    "gender": gender_mapping[gender_input],

    "ever_married":
        1 if ever_married_input == "Yes" else 0,

    "Residence_type":
        1 if residence_type_input == "Urban" else 0
}


# =========================================================
# WORK TYPE
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
# SMOKING STATUS
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
# DATAFRAME
# =========================================================

input_df = pd.DataFrame(
    [user_input_dict]
)


# Ensure exact training feature order

input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "🔮  PREDICT STROKE RISK  ",
    use_container_width=True
):

    try:

        # -------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # -------------------------------------------------

        numerical_features = [
            "age",
            "avg_glucose_level",
            "bmi"
        ]


        input_numerical = input_df[
            numerical_features
        ].copy()


        input_numerical_scaled = scaler.transform(
            input_numerical
        )


        input_df[
            numerical_features
        ] = input_numerical_scaled


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        prediction_proba = model.predict_proba(
            input_df
        )[0]


        stroke_probability = (
            prediction_proba[1] * 100
        )

        no_stroke_probability = (
            prediction_proba[0] * 100
        )


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🔬 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "⚠️ Model Prediction: Stroke"
            )

        else:

            st.success(
                "✅ Model Prediction: No Stroke"
            )


        # =================================================
        # METRICS
        # =================================================

        metric1, metric2 = st.columns(2)


        with metric1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div>
                🫀 Stroke Probability
                </div>

                <div class="metric-value">
                {stroke_probability:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with metric2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div>
                💚 No Stroke Probability
                </div>

                <div class="metric-value">
                {no_stroke_probability:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY BAR
        # =================================================

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.progress(
            min(
                max(
                    stroke_probability / 100,
                    0.0
                ),
                1.0
            )
        )


        st.caption(
            f"Model probability for Stroke: "
            f"{stroke_probability:.2f}%"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.warning(
            "⚠️ Disclaimer: This application provides "
            "a machine-learning prediction for educational "
            "purposes. It is not a medical diagnosis and "
            "should not replace professional medical advice."
        )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🫀 Stroke AI Prediction System

<br><br>

Built with Python • Streamlit • Random Forest

<br>

Machine Learning Project

</div>
""", unsafe_allow_html=True)
