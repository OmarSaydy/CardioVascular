# train_model.py
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import joblib

# -------------------------
# 1️⃣ Load the dataset
# -------------------------
df = pd.read_csv("cardio_train.csv", delimiter=';')

# -------------------------
# 2️⃣ Select features and target
# -------------------------
X = df[['age','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','active']]
y = df['cardio']

# -------------------------
# 3️⃣ Scale numeric features
# -------------------------
numeric_cols = ['age','height','weight','ap_hi','ap_lo']
scaler = RobustScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# -------------------------
# 4️⃣ Train XGBoost model
# -------------------------
ratio = (y == 0).sum() / (y == 1).sum()  # optional for imbalanced data
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X, y)

# -------------------------
# 5️⃣ Save model and scaler
# -------------------------
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler have been saved successfully!")
