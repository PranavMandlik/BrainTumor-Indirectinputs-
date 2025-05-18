import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

# Improve model with hyperparameter tuning
clf = RandomForestClassifier(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(clf, params, cv=3, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save everything
joblib.dump(best_model, "tumor_multi_model.pkl")
joblib.dump(label_encoders, "tumor_label_encoders_multi.pkl")
joblib.dump(target_encoder, "tumor_target_encoder_multi.pkl")

# Output
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("📊 Confusion Matrix:\n", conf_matrix)
print("📋 Classification Report:\n", report)
print("💾 Model and encoders saved.")
