import joblib
import numpy as np

# 1️⃣ Tải model và scaler
model = joblib.load("wine_mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2️⃣ Nhập dữ liệu rượu (13 đặc trưng)
print("Nhập các giá trị đặc trưng của rượu vang:")
fixed_acidity = float(input("1. fixed acidity: "))
volatile_acidity = float(input("2. volatile acidity: "))
citric_acid = float(input("3. citric acid: "))
residual_sugar = float(input("4. residual sugar: "))
chlorides = float(input("5. chlorides: "))
free_sulfur_dioxide = float(input("6. free sulfur dioxide: "))
total_sulfur_dioxide = float(input("7. total sulfur dioxide: "))
density = float(input("8. density: "))
pH = float(input("9. pH: "))
sulphates = float(input("10. sulphates: "))
alcohol = float(input("11. alcohol: "))



# 3️⃣ Tạo mảng dữ liệu để dự đoán
features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                      density, pH, sulphates, alcohol]])

# 4️⃣ Chuẩn hóa dữ liệu
features_scaled = scaler.transform(features)

# 5️⃣ Dự đoán
predicted_quality = model.predict(features_scaled)[0]
print(f"\n🍷 Dự đoán chất lượng rượu: {predicted_quality:.2f} / 10")
