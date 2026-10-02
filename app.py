import streamlit as st
import pandas as pd
import joblib


# Fungsi untuk memuat file .pkl
def load_pkl_file(filename):
    return joblib.load(filename)


# Memuat model dan preprocessing
rf_model = load_pkl_file("random_forest_model.pkl")
imputer = load_pkl_file("median_imputer_stroke.pkl")
scaler = load_pkl_file("scaler_stroke.pkl")
feature_columns = load_pkl_file("feature_columns_stroke.pkl")


# Judul aplikasi
st.title("Aplikasi Prediksi Risiko Stroke")
st.write("Masukkan detail pasien untuk memprediksi risiko stroke.")


# Informasi pasien
st.header("Informasi Pasien")

numerical_features = [
    "age",
    "avg_glucose_level",
    "bmi"
]


# Input numerik
age = st.slider(
    "Usia",
    min_value=0.08,
    max_value=82.0,
    value=40.0
)

hypertension = st.radio(
    "Hipertensi",
    [0, 1],
    format_func=lambda x: "Ya" if x == 1 else "Tidak"
)

heart_disease = st.radio(
    "Penyakit Jantung",
    [0, 1],
    format_func=lambda x: "Ya" if x == 1 else "Tidak"
)

avg_glucose_level = st.slider(
    "Rata-rata Tingkat Glukosa",
    min_value=55.12,
    max_value=271.74,
    value=100.0
)

bmi = st.slider(
    "BMI",
    min_value=10.3,
    max_value=51.0,
    value=25.0
)


# Input kategori
gender_options = {
    "Perempuan": 0,
    "Laki-laki": 1
}

gender_input = st.selectbox(
    "Jenis Kelamin",
    list(gender_options.keys())
)

ever_married_input = st.radio(
    "Pernah Menikah",
    ["Ya", "Tidak"]
)

work_type_input = st.selectbox(
    "Tipe Pekerjaan",
    [
        "Swasta",
        "Wiraswasta",
        "Anak-anak",
        "PNS",
        "Tidak Pernah Bekerja"
    ]
)

residence_type_input = st.radio(
    "Tipe Tempat Tinggal",
    ["Perkotaan", "Pedesaan"]
)

smoking_status_input = st.selectbox(
    "Status Merokok",
    [
        "Dulu Merokok",
        "Tidak Pernah Merokok",
        "Merokok",
        "Tidak Diketahui"
    ]
)


# Membuat input dictionary
user_input_dict = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender_options[gender_input],
    "ever_married": 1 if ever_married_input == "Ya" else 0,
    "Residence_type": 1 if residence_type_input == "Perkotaan" else 0
}


# Inisialisasi kolom one-hot
for col in feature_columns:
    if col.startswith("work_") or col.startswith("smoking_"):
        user_input_dict[col] = 0


# Work type
if work_type_input == "Tidak Pernah Bekerja":
    user_input_dict["work_Never_worked"] = 1

elif work_type_input == "Swasta":
    user_input_dict["work_Private"] = 1

elif work_type_input == "Wiraswasta":
    user_input_dict["work_Self-employed"] = 1

elif work_type_input == "Anak-anak":
    user_input_dict["work_children"] = 1

elif work_type_input == "PNS":
    user_input_dict["work_Govt_job"] = 1


# Smoking status
if smoking_status_input == "Dulu Merokok":
    user_input_dict["smoking_formerly smoked"] = 1

elif smoking_status_input == "Tidak Pernah Merokok":
    user_input_dict["smoking_never smoked"] = 1

elif smoking_status_input == "Merokok":
    user_input_dict["smoking_smokes"] = 1

elif smoking_status_input == "Tidak Diketahui":
    user_input_dict["smoking_Unknown"] = 1


# Membuat DataFrame
input_df = pd.DataFrame([user_input_dict])

# Menyesuaikan urutan kolom dengan saat training
input_df = input_df.reindex(
    columns=feature_columns,
    fill_value=0
)


# Preprocessing fitur numerik
input_numerical_processed = input_df[numerical_features].copy()

# Imputasi
input_numerical_processed = pd.DataFrame(
    imputer.transform(input_numerical_processed),
    columns=numerical_features,
    index=input_df.index,
)

# Scaling
input_numerical_processed = pd.DataFrame(
    scaler.transform(input_numerical_processed),
    columns=numerical_features,
    index=input_df.index,
)

# Masukkan kembali hasil preprocessing
input_df[numerical_features] = input_numerical_processed


# Tombol prediksi
if st.button("Prediksi Risiko Stroke"):

    prediction = rf_model.predict(input_df)

    prediction_proba = rf_model.predict_proba(input_df)

    st.subheader("Hasil Prediksi")

    if prediction[0] == 1:
        st.error(
            "Berdasarkan informasi yang diberikan, "
            "model memprediksi risiko STROKE."
        )
    else:
        st.success(
            "Berdasarkan informasi yang diberikan, "
            "model memprediksi risiko TIDAK STROKE."
        )

    st.write(
        f"Probabilitas Tidak Stroke: "
        f"**{prediction_proba[0][0]:.2%}**"
    )

    st.write(
        f"Probabilitas Stroke: "
        f"**{prediction_proba[0][1]:.2%}**"
    )

    st.write("---")

    st.write(
        "Disclaimer: Ini adalah model prediktif dan "
        "tidak boleh digunakan sebagai pengganti nasihat medis profesional."
    )