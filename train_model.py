import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("augmented_tumor_dataset_with_status.csv")

# Filter out empty labels if any
df = df[df["Tumor_Status"].notna()]

# Encode categorical features
label_encoders = {}
for col in ['Family_History', 'Gender'] + df.columns[8:-2].tolist():  # symptom columns
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target (multi-class: No Tumor, Benign Tumor, Aggressive Tumor)
target_encoder = LabelEncoder()
df["Tumor_Status_Encoded"] = target_encoder.fit_transform(df["Tumor_Status"])

X = df.drop(columns=["Aggressiveness", "Tumor_Status", "Tumor_Status_Encoded"])
y = df["Tumor_Status_Encoded"]

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Scale features (important for MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the MLPClassifier with 'lbfgs' solver
mlp = MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', max_iter=3000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save everything
joblib.dump(mlp, "tumor_multi_model_mlp.pkl")
joblib.dump(scaler, "tumor_scaler.pkl")
joblib.dump(label_encoders, "tumor_label_encoders_multi.pkl")
joblib.dump(target_encoder, "tumor_target_encoder_multi.pkl")

# Output
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("📊 Confusion Matrix:\n", conf_matrix)
print("📋 Classification Report:\n", report)
print("💾 MLP model, scaler, and encoders saved.")