import os

import numpy as np
import joblib

class WaterPurposeModel:
    def __init__(self):
        # Load the trained SVM model
        self.model_path = "models/svm_model.joblib"
        #os import to resolve path error
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, "models", "svm_model.joblib")
        self.scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
        self.label_encoder_path = os.path.join(base_dir, "models", "label_encoder.joblib")
        self.svm_model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

    # Function to predict use case
    def predict(self, parameters):
        input_data = np.array(parameters).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        prediction = self.svm_model.predict(input_scaled)
        result = self.label_encoder.inverse_transform(prediction)[0]
        print("Prediction Result: ", result)
        return result