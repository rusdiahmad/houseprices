import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------
# 🧩 Konfigurasi Halaman
# -------------------------------------------
st.set_page_config(page_title="My Portfolio with Streamlit",
                   page_icon="🏡",
                   layout="wide")

st.title("🏡 My Portfolio with Streamlit")
st.markdown("""
Selamat datang di aplikasi portofolio saya!  
Di sini Anda dapat melihat **profil**, **proyek**, **visualisasi data House Prices**,  
serta mencoba langsung **prediksi harga rumah** berbasis Machine Learning.
""")

# -------------------------------------------
# 📍 Sidebar Navigasi
# -------------------------------------------
menu = st.sidebar.radio("Navigasi", [
    "Tentang Saya",
    "Proyek Saya",
    "Visualisasi Data",
    "Prediksi Harga Rumah"
])

# -------------------------------------------
# 🧑‍💼 Tentang Saya
# -------------------------------------------
if menu == "Tentang Saya":
    st.header("👋 Tentang Saya")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    st.markdown("""
    **Nama:** Rusdi Ahmad  
    **Latar Belakang:** Guru Matematika & Data Science Enthusiast  
    **Keahlian:**  
    - Machine Learning  
    - Data Visualization  
    - Streamlit, Pandas, Scikit-Learn  
    - Pendidikan & Analisis Data  
    
    > “Data bukan hanya angka, tapi cerita yang menunggu untuk diceritakan.”  
    """)

# -------------------------------------------
# 💼 Proyek Saya
# -------------------------------------------
elif menu == "Proyek Saya":
    st.header("💼 Proyek Saya")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Analisis Nilai UTBK")
        st.image("https://cdn-icons-png.flaticon.com/512/4341/4341139.png")
        st.write("Proyek analisis data nilai UTBK untuk mengetahui faktor kelulusan.")

    with col2:
        st.subheader("EDA ChatGPT Usage")
        st.image("https://cdn-icons-png.flaticon.com/512/6016/6016960.png")
        st.write("Exploratory Data Analysis penggunaan ChatGPT dengan data simulasi.")

    with col3:
        st.subheader("Prediksi Harga Rumah")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png")
        st.write("Model Machine Learning untuk memprediksi harga rumah dari dataset Kaggle.")

# -------------------------------------------
# 📊 Visualisasi Data
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
# 🤖 Prediksi Harga Rumah
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
    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    st.write("---")
    st.write("### 📤 Upload File CSV untuk Prediksi Baru")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("Cuplikan Data yang Diupload:")
        st.dataframe(new_data.head())

        new_data = new_data[X.columns.intersection(new_data.columns)].fillna(0)
        preds = model.predict(new_data)
        st.success("Prediksi Selesai ✅")
        result_df = pd.DataFrame({"Predicted_Price": preds})
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Hasil Prediksi", data=csv,
                           file_name="house_price_predictions.csv", mime="text/csv")
