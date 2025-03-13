import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from joblib import dump


df = pd.read_csv("data/medical_dataset.csv")


label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])


label_mapping = pd.DataFrame({"Disease": label_encoder.classes_, "Encoded": label_encoder.transform(label_encoder.classes_)})
label_mapping.to_csv("models/label_mapping.csv", index=False)  


X = df.drop("Disease", axis=1)
y = df["Disease"]


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


dump(scaler, "models/scaler.joblib")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)
dump(model, "models/disease_prediction_model.joblib")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Model trained with accuracy: {accuracy:.2f}%")
