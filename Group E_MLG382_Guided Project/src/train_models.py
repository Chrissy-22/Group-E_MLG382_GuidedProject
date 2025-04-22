# python src/web_app.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
df = pd.read_csv("data/train_engineered.csv")
df = df.drop(columns=["StudentID"])

# Separate features and label
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1]

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "artifacts/label_encoder.pkl")

# Scale numeric features
numeric_cols = ['StudyTimeWeekly', 'Absences', 'GPA']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
joblib.dump(scaler, "artifacts/scaler.pkl")  # ✅ This is the correct way to save the scaler

# Save feature names
joblib.dump(list(X.columns), "artifacts/feature_names.pkl")

# One-hot encode target for DL
num_classes = len(np.unique(y_encoded))
y_cat = to_categorical(y_encoded, num_classes)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)

results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
joblib.dump(lr, "artifacts/model_logistic.pkl")
results['Logistic Regression'] = classification_report(y_val, lr.predict(X_val), output_dict=True)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "artifacts/model_rf.pkl")
results['Random Forest'] = classification_report(y_val, rf.predict(X_val), output_dict=True)

# XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train, y_train)
joblib.dump(xgb, "artifacts/model_xgb.pkl")
results['XGBoost'] = classification_report(y_val, xgb.predict(X_val), output_dict=True)

# Deep Learning
dl = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dl.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0)
dl.save("artifacts/model_dl.keras")

# Evaluate DL
y_dl_pred = np.argmax(dl.predict(X_val), axis=1)
results['Deep Learning'] = classification_report(y_val, y_dl_pred, output_dict=True)

# Save performance metrics
perf_df = pd.DataFrame([{
    "Model": model,
    "Accuracy": metrics["accuracy"],
    "Precision": metrics["weighted avg"]["precision"],
    "Recall": metrics["weighted avg"]["recall"],
    "F1": metrics["weighted avg"]["f1-score"]
} for model, metrics in results.items()])
perf_df.to_csv("artifacts/model_performance.csv", index=False)

print("All models trained and artifacts saved.")
