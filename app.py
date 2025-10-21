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
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)

with col2:
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0)
    density = st.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.51)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
    alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4)

# --- Nút dự đoán ---
if st.button("🔮 Dự đoán"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    
    # --- Trường hợp 1: Tất cả đều bằng 0 ---
    if np.all(input_data == 0):
        st.error("🚫 Dữ liệu không hợp lệ! Tất cả giá trị đều bằng 0 — không thể dự đoán.")
    
    # --- Trường hợp 2: Có ít nhất một giá trị bằng 0 ---
    elif np.any(input_data == 0):
        st.warning("⚠️ Một số giá trị bằng 0 — hệ thống coi đây là dữ liệu bất thường.")
        prediction = 3.0
        st.success(f"🎯 **Điểm chất lượng dự đoán (giảm do dữ liệu bất thường): {prediction:.2f}/8**")
    
    # --- Trường hợp 3: Dữ liệu hợp lệ ---
    else:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = max(3, min(8, prediction))  # Giới hạn trong 3–8
        st.success(f"🎯 **Điểm chất lượng dự đoán: {prediction:.2f}/8**")
