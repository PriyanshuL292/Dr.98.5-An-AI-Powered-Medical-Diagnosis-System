import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_FOLDER = "data/"

def load_data():
    all_data = []
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="ISO-8859-1")  
        elif file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            continue  
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        raise FileNotFoundError("No valid CSV or Excel files found in 'data/' folder.")

def preprocess_data(df):
    df.fillna(0, inplace=True)  
    df['Disease'] = df['Disease'].astype(str)


    label_encoder = LabelEncoder()
    df['Disease'] = label_encoder.fit_transform(df['Disease'])

    symptoms = df.columns[1:]  

    
    scaler = MinMaxScaler()
    df[symptoms] = scaler.fit_transform(df[symptoms])

    return df  

df = load_data()
processed_data = preprocess_data(df)
processed_data.to_csv("data/processed_data.csv", index=False)
print("Data preprocessing complete. Saved as 'processed_data.csv'")
