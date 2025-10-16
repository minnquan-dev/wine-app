import streamlit as st
import joblib
import numpy as np

# --- Cấu hình trang ---
st.set_page_config(page_title="Wine Quality Predictor 🍷", page_icon="🍇", layout="centered")

# --- Tải mô hình & scaler ---
model = joblib.load("wine_mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- CSS tùy chỉnh giao diện ---
st.markdown("""
    <style>
        .main {
            background-color: #fafafa;
            font-family: "Segoe UI", sans-serif;
        }
        h1 {
            color: #8B0000;
            text-align: center;
        }
        .stButton>button {
            background-color: #8B0000;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Tiêu đề ---
st.title("🍷 Wine Quality Prediction App")
st.write("Nhập các chỉ số hóa học để dự đoán **điểm chất lượng rượu vang (3–8)**.")

# --- Tạo cột nhập dữ liệu ---
col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 15.0, 7.4)
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 1.9)
    chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.076)

with col2:
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1, 70, 11)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6, 250, 34)
    density = st.number_input("Density", 0.9900, 1.0050, 0.9978)
    pH = st.number_input("pH", 2.5, 4.0, 3.51)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

# --- Nút dự đoán ---
if st.button("🔮 Dự đoán"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction = max(3, min(8, prediction))  # Giới hạn 3–8

    st.success(f"🎯 **Điểm chất lượng dự đoán: {prediction:.2f}/8**")
