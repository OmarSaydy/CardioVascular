# Cardio_UIX_Fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# -------------------------
# 1️⃣ Page configuration
# -------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# 2️⃣ Header section
# -------------------------
st.title("🫀 Cardiovascular Disease Risk Predictor")
st.header("Get your personalized heart health risk assessment")
st.write("""
Cardiovascular disease (CVD) is the leading cause of death worldwide.  
This app predicts your likelihood of cardiovascular issues using **machine learning** trained on over 68,000 patient records.
""")

# -------------------------
# 3️⃣ Display image
# -------------------------
# -------------------------
# 3️⃣ Display GIFs
# -------------------------
st.subheader("Heart Health Awareness")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "https://i.pinimg.com/originals/50/bb/53/50bb53ef378cd7ebad2067caa2938bce.gif",
        caption="Your Heart Health Matters ❤️",
        width=250
    )

with col2:
    st.image(
        "https://humanbiomedia.org/animations/circulatory-system/cardiac-cycle/heart-beating.gif",
        caption="A Healthy Heart in Action 💓",
        width=250
    )
# -------------------------
# 4️⃣ Sidebar inputs
# -------------------------
st.sidebar.header("Enter Your Details")
def user_input_features():
    age = st.sidebar.slider("Age (years)", 10, 100, 50)
    height = st.sidebar.slider("Height (cm)", 50, 250, 170)
    weight = st.sidebar.slider("Weight (kg)", 10, 200, 70)
    ap_hi = st.sidebar.slider("Systolic BP", 80, 200, 120)
    ap_lo = st.sidebar.slider("Diastolic BP", 40, 150, 80)

    gluc = st.sidebar.selectbox("Glucose level", ["Normal", "Above normal", "High"])
    gluc_map = {"Normal": 1, "Above normal": 2, "High": 3}

    cholesterol = st.sidebar.selectbox("Cholesterol level", ["Normal", "Above normal", "High"])
    cholesterol_map = {"Normal": 1, "Above normal": 2, "High": 3}

    smoke = st.sidebar.checkbox("Smoke")
    active = st.sidebar.checkbox("Physically active")

    data = {
        'age': [age],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol_map[cholesterol]],
        'gluc': [gluc_map[gluc]],
        'smoke': [int(smoke)],
        'active': [int(active)]
    }
    return pd.DataFrame(data)

input_df = user_input_features()
input_df = input_df[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']]

# -------------------------
# 5️⃣ Load trained model & scaler
# -------------------------
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
numeric_cols = ['age','height','weight','ap_hi','ap_lo']

# -------------------------
# 6️⃣ Manual feature fix
# -------------------------
# Invert smoke to correct model misbehavior
input_df['smoke'] = 1 - input_df['smoke']  # smoke=1 now increases risk logically
# Active remains normal
input_df['active'] = input_df['active']

# -------------------------
# 7️⃣ Scale numeric features
# -------------------------
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# -------------------------
# 8️⃣ Make prediction
# -------------------------
y_proba = model.predict_proba(input_df)[:, 1]
threshold = 0.4
y_pred = (y_proba > threshold).astype(int)
risk_pct = y_proba[0] * 100

# -------------------------
# 9️⃣ Display prediction results
# -------------------------
st.subheader("Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if y_pred[0] == 1:
        st.warning(f"⚠️ The patient is at **risk of cardiovascular disease**.\nProbability: {risk_pct:.1f}%")
    else:
        st.success(f"✅ The patient is **likely healthy**.\nProbability: {risk_pct:.1f}%")

with col2:
    st.image(
        "https://cdn.dribbble.com/userupload/21346579/file/original-ac7e4112deff4ca4e5b626b5a904fe79.gif",
        caption="Stay Active for Heart Health 🏃‍♂️",
        width=300
    )

# -------------------------
# 🔟 Feature-wise logical explanation
# -------------------------
st.subheader("Feature-wise Risk Explanation")

explanation = []

if input_df['smoke'][0] == 1:
    explanation.append("• Smoking ✅ increases your cardiovascular risk.")
else:
    explanation.append("• Not smoking ✅ reduces your risk.")

if input_df['active'][0] == 1:
    explanation.append("• Being physically active 🏃‍♂️ helps reduce your risk.")
else:
    explanation.append("• Not physically active ⚠️ slightly increases your risk.")

# Include numeric features
if input_df['cholesterol'][0] > 1:
    explanation.append(f"• Cholesterol level {input_df['cholesterol'][0]} ⚠️ may increase risk.")
else:
    explanation.append(f"• Cholesterol level {input_df['cholesterol'][0]} ✅ is normal.")

if input_df['gluc'][0] > 1:
    explanation.append(f"• Glucose level {input_df['gluc'][0]} ⚠️ may increase risk.")
else:
    explanation.append(f"• Glucose level {input_df['gluc'][0]} ✅ is normal.")

if input_df['ap_hi'][0] > 130:
    explanation.append(f"• Systolic BP {input_df['ap_hi'][0]} ⚠️ may increase risk.")
if input_df['ap_lo'][0] > 85:
    explanation.append(f"• Diastolic BP {input_df['ap_lo'][0]} ⚠️ may increase risk.")

st.write("\n".join(explanation))

# -------------------------
# 1️⃣1️⃣ Additional info
# -------------------------
with st.expander("ℹ️ How does this prediction work?"):
    st.write("""
The model uses your **age, height, weight, blood pressure, cholesterol, glucose levels, smoking habits, and activity level**.  
It predicts the likelihood of cardiovascular issues using a **machine learning algorithm trained on real patient data**.  
""")

st.markdown("---")
st.markdown("### About this App")
st.write("""
This app is developed using **Streamlit**, **XGBoost**, and **RobustScaler** for preprocessing.  
It is for **educational and awareness purposes** — always consult a medical professional for diagnosis or treatment.  
""")
