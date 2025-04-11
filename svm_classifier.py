import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
import joblib
from imblearn.over_sampling import SMOTE  # Handle class imbalance

# Load dataset
data_path = "dataset/clustered_water_data.xlsx"
df = pd.read_excel(data_path)
df = df.dropna()

# Display dataset shape and class distribution
print(f"Dataset Shape: {df.shape}")
print("Class Distribution:\n", df["Predicted Use"].value_counts())


numeric_df = df.select_dtypes(include=[np.number])

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
X = df.iloc[:, :-2]
y = df.iloc[:, -1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best SVM Parameters:", best_params)
svm_model = SVC(**best_params)
svm_model.fit(X_train, y_train)

#predicting
y_pred = svm_model.predict(X_test)

#metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Model F1 Score: {f1:.2f}")

#Saving the Model
model_path = "/models/svm_model.joblib"
joblib.dump(svm_model, model_path)
print(f"Model saved to {model_path}")

#scalar
scaler_path = "models/scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

#Label Encoding
label_encoder_path = "models/label_encoder.joblib"
joblib.dump(label_encoder, label_encoder_path)
print(f"Label Encoder saved to {label_encoder_path}")

def predict_usecase(parameters):
    input_data = np.array(parameters).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = svm_model.predict(input_scaled)
    return label_encoder.inverse_transform(prediction)[0]

example_parameters = [7.38, 2.2456, -10.10, 1.15, 8572.40, 4380.50, 11510.50, 2155.30, 98.50, 1025.30, 612.30, 11820.40, 2185.30]
print("Predicted Use Case:", predict_usecase(example_parameters))