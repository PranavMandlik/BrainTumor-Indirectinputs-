import streamlit as st
import joblib
import pandas as pd

# Load model, encoders, and scaler
encoders = joblib.load("tumor_label_encoders_multi.pkl")
target_encoder = joblib.load("tumor_target_encoder_multi.pkl")
scaler = joblib.load("tumor_scaler.pkl")  # ✅ Load the saved scaler
model = joblib.load("tumor_multi_model_mlp.pkl")

st.title("🧠 Brain Tumor Prediction App (Single Input & Bulk CSV)")

# Encoding function
def encode_input(df):
    df = df.copy()
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df

# 🧍 Single Prediction Input
st.header("🔍 Single Patient Prediction")
def get_input():
    return pd.DataFrame([{
        "Blood_Pressure": st.slider("Blood Pressure", 90, 180, 120),
        "Cholesterol": st.slider("Cholesterol", 150, 300, 200),
        "Blood_Glucose": st.slider("Blood Glucose", 70, 200, 100),
        "Family_History": st.selectbox("Family History", ["Yes", "No"]),
        "Age": st.slider("Age", 10, 90, 30),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Triglycerides": st.slider("Triglycerides", 100, 300, 150),
        "HbA1c": st.slider("HbA1c", 4.0, 10.0, 6.0),
        "Headaches": st.selectbox("Headaches", ["Yes", "No"]),
        "Seizures": st.selectbox("Seizures", ["Yes", "No"]),
        "Vision_Problems": st.selectbox("Vision Problems", ["Yes", "No"]),
        "Nausea": st.selectbox("Nausea", ["Yes", "No"]),
        "Cognitive_Changes": st.selectbox("Cognitive Changes", ["Yes", "No"]),
        "Motor_Symptoms": st.selectbox("Motor Symptoms", ["Yes", "No"]),
        "Speech_Problems": st.selectbox("Speech Problems", ["Yes", "No"]),
        "Balance_Problems": st.selectbox("Balance Problems", ["Yes", "No"]),
        "Hearing_Loss": st.selectbox("Hearing Loss", ["Yes", "No"])
    }])

input_df = get_input()
if st.button("Predict Single Patient"):
    encoded = encode_input(input_df)
    scaled = scaler.transform(encoded)  # ✅ Scale the encoded input
    prediction = model.predict(scaled)[0]
    result = target_encoder.inverse_transform([prediction])[0]
    st.success(f"🧬 Predicted Tumor Status: **{result}**")

# 📁 Bulk CSV Upload & Prediction
st.header("📊 Bulk Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV with patient records", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Tumor_Status' in df.columns:
        actual = df['Tumor_Status']
        df_features = df.drop(columns=["Tumor_Status", "Aggressiveness"], errors='ignore')
    else:
        actual = None
        df_features = df.copy()

    encoded_df = encode_input(df_features)
    scaled_df = scaler.transform(encoded_df)  # ✅ Apply scaling
    preds = model.predict(scaled_df)
    df['Predicted_Tumor_Status'] = target_encoder.inverse_transform(preds)

    if actual is not None:
        df['Actual_Tumor_Status'] = actual
        df['Match'] = df['Predicted_Tumor_Status'] == df['Actual_Tumor_Status']

    st.subheader("📄 Prediction Results")
    st.dataframe(df)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", data=csv, file_name="tumor_predictions.csv", mime='text/csv')