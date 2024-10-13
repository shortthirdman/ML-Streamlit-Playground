import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.dummy import DummyRegressor

load_dotenv()

def main():
    """
     Dummy Regressor: Naively choosing the best number for all of your prediction
    """
    dataset_dict = {
        'Outlook': ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain', 'sunny', 'overcast', 'rain', 'sunny', 'sunny', 'rain', 'overcast', 'rain', 'sunny', 'overcast', 'sunny', 'overcast', 'rain', 'overcast'],
        'Temperature': [85.0, 80.0, 83.0, 70.0, 68.0, 65.0, 64.0, 72.0, 69.0, 75.0, 75.0, 72.0, 81.0, 71.0, 81.0, 74.0, 76.0, 78.0, 82.0, 67.0, 85.0, 73.0, 88.0, 77.0, 79.0, 80.0, 66.0, 84.0],
        'Humidity': [85.0, 90.0, 78.0, 96.0, 80.0, 70.0, 65.0, 95.0, 70.0, 80.0, 70.0, 90.0, 75.0, 80.0, 88.0, 92.0, 85.0, 75.0, 92.0, 90.0, 85.0, 88.0, 65.0, 70.0, 60.0, 95.0, 70.0, 78.0],
        'Wind': [False, True, False, False, False, True, True, False, False, False, True, True, False, True, True, False, False, True, False, True, True, False, True, False, False, True, False, False],
        'Num_Players': [52,39,43,37,28,19,43,47,56,33,49,23,42,13,33,29,25,51,41,14,34,29,49,36,57,21,23,41]
    }
    df = pd.DataFrame(dataset_dict)

    # One-hot encode 'Outlook' column
    df = pd.get_dummies(df, columns=['Outlook'], prefix='', prefix_sep='', dtype=int)

    # Convert 'Wind' column to binary
    df['Wind'] = df['Wind'].astype(int)

    # Split data into features and target, then into training and test sets
    X, y = df.drop(columns='Num_Players'), df['Num_Players']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)

    # Choose a strategy for your DummyRegressor ('mean', 'median', 'constant')
    strategy = 'median'
    
    # Initialize and train the model
    dummy_reg = DummyRegressor(strategy=strategy)
    dummy_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = dummy_reg.predict(X_test)

    print("Label     :",list(y_test))
    print("Prediction:",list(y_pred))

    # Calculate and print RMSE
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse.round(2)}")

    mse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {mse.round(2)}")


if __name__ == "__main__":
    main()