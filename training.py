import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("data/processed_data.csv")
features = dataset.drop("Disease", axis=1)  
labels = dataset["Disease"]  


X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    features, labels, test_size=0.2, random_state=42
)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_set, y_train_set)


y_predictions = rf_classifier.predict(X_test_set)


accuracy = accuracy_score(y_test_set, y_predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
