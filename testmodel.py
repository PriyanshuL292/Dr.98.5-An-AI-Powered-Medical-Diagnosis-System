import pandas as pd
import numpy as np
from joblib import load


model = load("models/disease_prediction_model.joblib")
scaler = load("models/scaler.joblib")


label_mapping = pd.read_csv("models/label_mapping.csv")  
inverse_label_mapping = dict(zip(label_mapping["Encoded"], label_mapping["Disease"]))  


df = pd.read_csv("data/processed_data.csv")
trained_features = [col for col in df.columns if col != "Disease"]


def predict_disease(input_symptoms):
    if not isinstance(input_symptoms, list):
        print("Error: Input should be a list of symptoms.")
        return

    
    test_input = pd.DataFrame(np.zeros((1, len(trained_features))), columns=trained_features)

    
    for symptom in input_symptoms:
        if symptom in trained_features:
            test_input[symptom] = 1

    
    test_input_scaled = scaler.transform(test_input)

    
    predicted_disease_encoded = model.predict(test_input_scaled)[0]
    predicted_disease = inverse_label_mapping.get(predicted_disease_encoded, "Unknown Disease")

    print(f"🩺 Predicted Disease: {predicted_disease}")


example_symptoms = ["Runny Nose", "Sneezing", "Cough", "Sore Throat", "Mild Fever"]  
predict_disease(example_symptoms)
