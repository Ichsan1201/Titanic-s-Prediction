import streamlit as st
import pickle
import pandas as pd

# Load model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fungsi prediksi
def predict_survival(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked
):
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    prediction = model.predict(data)

    return prediction[0]


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Masukkan data penumpang untuk memprediksi apakah penumpang akan selamat atau tidak."
)

# Input User

pclass = st.selectbox(
    "Kelas Penumpang",
    [1, 2, 3]
)

sex_label = st.radio(
    "Jenis Kelamin",
    ["Laki-laki", "Perempuan"]
)

sex = 1 if sex_label == "Laki-laki" else 0

age = st.slider(
    "Usia",
    min_value=0,
    max_value=100,
    value=25
)

sibsp = st.slider(
    "Jumlah Saudara/Pasangan",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.slider(
    "Jumlah Orang Tua/Anak",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Harga Tiket",
    min_value=0.0,
    value=32.0,
    step=1.0
)

embarked_label = st.selectbox(
    "Pelabuhan Keberangkatan",
    ["C", "Q", "S"]
)

embarked = {
    "C": 0,
    "Q": 1,
    "S": 2
}[embarked_label]

# Prediksi

if st.button("Prediksi"):

    result = predict_survival(
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    )

    if result == 1:
        st.success("✅ Diprediksi Selamat")
    else:
        st.error("❌ Diprediksi Tidak Selamat")