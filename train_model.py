# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Đọc dữ liệu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')  # Dữ liệu dùng dấu chấm phẩy ';'

# 2. Chia dữ liệu thành đầu vào (X) và đầu ra (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# 3. Tách tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Chuẩn hóa dữ liệu (giúp MLP học tốt hơn)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Huấn luyện mô hình MLP (Multi-Layer Perceptron)
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(X_test_scaled)
score = r2_score(y_test, y_pred)
print(f"🎯 Độ chính xác R²: {score:.3f}")

# 7. Lưu mô hình và scaler
joblib.dump(model, "wine_mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Đã huấn luyện và lưu mô hình thành công!")