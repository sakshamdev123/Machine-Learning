# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib

def load_model_and_scaler():
    model = load_model("trained_model.h5")
    sc = joblib.load("scaler.pkl")
    return model, sc

def predict_stock_price(model, sc, input_data):
    input_data_scaled = sc.transform(input_data)
    X_test = []

    for i in range(60, len(input_data_scaled)):
        X_test.append(input_data_scaled[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_pred_scaled = model.predict(X_test)
    predicted_price = sc.inverse_transform(y_pred_scaled)

    return predicted_price

def main():
    st.title("Stock Price Prediction App")

    # Load and preprocess the test data
    test_data = pd.read_csv('Google_test_data.csv')
    test_data["Close"] = pd.to_numeric(test_data["Close"], errors='coerce')
    test_data = test_data.dropna()
    test_data = test_data.iloc[:, 4:5]

    # Load the trained model and scaler
    model, sc = load_model_and_scaler()

    # Predict stock prices
    input_closing = test_data.iloc[:, 0:].values
    predicted_price = predict_stock_price(model, sc, input_closing)

    # Plotting
    st.line_chart(pd.DataFrame({
        'Actual Stock Price': test_data.iloc[60:, 0].values,
        'Predicted Stock Price': predicted_price.flatten()
    }))

    # Additional Information or Widgets if needed
    st.write("Additional Information or Widgets can be added here.")

if __name__ == '__main__':
    main()
