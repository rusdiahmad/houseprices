import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------
# 🧩 KONFIGURASI HALAMAN
# -------------------------------------------
st.set_page_config(page_title="My AI & ML Portfolio: House Price Prediction Dashboard",
                   page_icon="🤖",
                   layout="wide")

st.title("🤖 My AI & ML Portfolio: House Price Prediction Dashboard")
st.markdown("""
Selamat datang di **My AI & Machine Learning Portfolio**!  
Aplikasi ini menampilkan implementasi *end-to-end pipeline* untuk prediksi harga rumah 
menggunakan dataset **House Prices: Advanced Regression Techniques** dari Kaggle.

Anda dapat melihat profil saya, proyek-proyek AI/ML, 
analisis data interaktif, serta mencoba langsung model prediksi harga rumah.
""")

# -------------------------------------------
# 📍 SIDEBAR NAVIGASI
# -------------------------------------------
menu = st.sidebar.radio("Navigasi", [
    "Tentang Saya",
    "Proyek Saya",
    "Visualisasi Data",
    "Prediksi Harga Rumah"
])

# -------------------------------------------
# 👤 TENTANG SAYA
# -------------------------------------------
if menu == "Tentang Saya":
    st.header("👋 Tentang Saya")

    # tampilkan foto profil
    st.image("Pas Photo.jpg", width=180, caption="Rusdi Ahmad", use_container_width=False)

    st.markdown("""
    **Nama:** Rusdi Ahmad  
    **Latar Belakang:** Guru Matematika & AI/ML Enthusiast  
    **Bootcamp:** Artificial Intelligence & Machine Learning  
    **Keahlian:**  
    - Machine Learning & Predictive Modeling  
    - Data Visualization  
    - Streamlit, Pandas, Scikit-Learn  
    - Pendidikan dan Data Analysis  

    > “Artificial Intelligence bukan hanya tentang algoritma, tapi tentang memahami data dan memecahkan masalah nyata.”  
    """)

# -------------------------------------------
# 💼 PROYEK SAYA
# -------------------------------------------
elif menu == "Proyek Saya":
    st.header("💼 Proyek Saya")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("House Price Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png")
        st.write("Model ML untuk memprediksi harga rumah berdasarkan dataset Kaggle.")

    with col2:
        st.subheader("EDA ChatGPT Usage")
        st.image("https://cdn-icons-png.flaticon.com/512/6016/6016960.png")
        st.write("Exploratory Data Analysis terhadap data simulasi penggunaan ChatGPT.")

    with col3:
        st.subheader("UTBK Data Analysis")
        st.image("https://cdn-icons-png.flaticon.com/512/4341/4341139.png")
        st.write("Analisis data nilai UTBK untuk memahami faktor kelulusan siswa.")

# -------------------------------------------
# 📊 VISUALISASI DATA
# -------------------------------------------
elif menu == "Visualisasi Data":
    st.header("📊 Visualisasi Dataset House Prices")

    df = pd.read_csv("train.csv")
    st.write("### Cuplikan Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    st.write("### Distribusi Fitur")
    selected_col = st.selectbox("Pilih kolom untuk dilihat distribusinya:", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, color="teal", ax=ax)
    st.pyplot(fig)

    st.write("### Korelasi antar Fitur Numerik")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(fig)

# -------------------------------------------
# 🤖 PREDIKSI HARGA RUMAH
# -------------------------------------------
elif menu == "Prediksi Harga Rumah":
    st.header("🏠 Prediksi Harga Rumah Menggunakan Model ML")

    df = pd.read_csv("train.csv")
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### 🔎 Evaluasi Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    st.write("---")
    st.write("### 📤 Upload File CSV untuk Prediksi Baru")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("Cuplikan Data:")
        st.dataframe(new_data.head())

        new_data = new_data[X.columns.intersection(new_data.columns)].fillna(0)
        preds = model.predict(new_data)
        st.success("✅ Prediksi Selesai!")
        result_df = pd.DataFrame({"Predicted_Price": preds})
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Hasil Prediksi", data=csv,
                           file_name="house_price_predictions.csv", mime="text/csv")
