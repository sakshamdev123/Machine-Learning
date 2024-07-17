# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import joblib

def train_model(data):
    sc = MinMaxScaler(feature_range=(0, 1))
    closing_prices = data.iloc[:, 4:5].values
    closing_prices_scaled = sc.fit_transform(closing_prices)

    X_train = []
    y_train = []

    for i in range(60, len(data)):
        X_train.append(closing_prices_scaled[i-60:i, 0])
        y_train.append(closing_prices_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    model.save("trained_model.h5")
    joblib.dump(sc, "scaler.pkl")

if __name__ == "__main__":
    data = pd.read_csv('Google_train_data.csv')
    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
    data = data.dropna()
    train_model(data)
