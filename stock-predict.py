import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Define the stock symbol and the time period
stock_symbol = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-01-01'

# Download stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Drop any missing values
stock_data.dropna(inplace=True)

# Use only the 'Close' price for prediction
close_prices = stock_data['Close'].values

# Reshape the data for MinMaxScaler
close_prices = close_prices.reshape(-1, 1)

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create the training and testing datasets
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=100)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f'Root Mean Squared Error: {rmse}')

# Plot predictions vs actual
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index[train_size + time_step + 1:], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Price')
plt.plot(stock_data.index[train_size + time_step + 1:], predictions, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Visualize the entire data with predictions
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Actual Price')
plt.plot(stock_data.index[train_size + time_step + 1:], predictions, label='Predicted Price')
plt.axvline(stock_data.index[train_size + time_step], color='red', linestyle='--', label='Train/Test Split')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Save the model
model.save('stock_model.h5')
