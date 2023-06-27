from matplotlib import pyplot as plt
from math import pi, cos
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

ijmuiden = pd.read_pickle("../data/boei-data/IJmuiden-Buitenhaven(IJMH).pkl")
K13 = pd.read_pickle("../data/boei-data/K13-Alpha(K13).pkl")
A12 = pd.read_pickle("../data/boei-data/A12-platform(A12).pkl")

columns_in = ["wave-height", "wave-period", "wind-speed", "wave-dir"]
columns_out = ["wave-height"]

# Define the sequence length (number of time steps to consider)
seq_length = 2*24*6

def create_sequences(input_data, output_data, seq_length):
    X = []
    y = []
    for i in range(len(input_data) - seq_length):
        X.append(input_data[i:i+seq_length])
        y.append(output_data[i+seq_length])
    return np.array(X), np.array(y)

# drop nan data
ijmuiden = ijmuiden.drop('tide-height', axis=1)
ijmuiden = ijmuiden.dropna(axis=0)
K13 = K13.drop('tide-height', axis=1)
K13 = K13.dropna(axis=0)
A12 = A12.drop('tide-height', axis=1)
A12 = A12.dropna(axis=0)

# Rotate angles for 180 = betwe
K13['wave-dir'] = K13['wave-dir'] - 140 % 360
K13['wave-dir'] = np.cos(K13['wave-dir'].values/360*2*pi)

# Drop unnecessary columns
A12 = A12.drop('tide-height', axis=1)
A12 = A12.dropna(axis=0)

# Rotate angles for 180 = betwe
A12['wave-dir'] = A12['wave-dir'] - 140 % 360
A12['wave-dir'] = np.cos(A12['wave-dir'].values / 360 * 2 * np.pi)

# Make data equivalent
common_index = ijmuiden.index.intersection(A12.index)
A12 = A12.loc[common_index]

# Make data equivalent
common_index = ijmuiden.index.intersection(K13.index)
K13 = K13.loc[common_index]
ijmuiden = ijmuiden.loc[common_index]

# Extract necesdsary columns
input_data = K13[columns_in].values
output_data = ijmuiden[columns_out].values
input_data_a12 = A12[columns_in].values
output_data_a12 = ijmuiden[columns_out].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_input_data = scaler.fit_transform(input_data)
scaled_output_data = scaler.fit_transform(output_data)

# Split the data into training and testing sets
train_size = int(0.7 * len(input_data))
train_index = K13.iloc[train_size+seq_length:].index
train_input_data = scaled_input_data[:train_size]
train_output_data = scaled_output_data[:train_size]
test_input_data = scaled_input_data[train_size:]
test_output_data = scaled_output_data[train_size:]

train_input_data_a12 = scaled_input_data_a12[:train_size]
train_output_data_a12 = scaled_output_data_a12[:train_size]
test_input_data_a12 = scaled_input_data_a12[train_size:]
test_output_data_a12 = scaled_output_data_a12[train_size:]


# Create training sequences and labels
X_train, y_train = create_sequences(train_input_data, train_output_data, seq_length)
X_train_a12, y_train_a12 = create_sequences(train_input_data_a12, train_output_data_a12, seq_length)


X_train_combined = np.concatenate((X_train, X_train_a12), axis=0)
y_train_combined = np.concatenate((y_train, y_train_a12), axis=0)

# Create testing sequences and labels
X_test, y_test = create_sequences(test_input_data, test_output_data, seq_length)
X_test_combined = np.concatenate((X_test, X_test_a12), axis=0)




model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, len(columns_in))))
model.add(Dense(len(columns_out)))
model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(X_train_combined, y_train_combined, epochs=30, batch_size=16)

loss = model.evaluate(X_test_combined, y_test)
print(f'Test loss: {loss}')

# Make predictions on the testing data
predictions = model.predict(X_test_combined)

# Inverse transform the predictions and actual values to obtain the original scale
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(train_index, predictions[:, 0], label='Predicted')
plt.plot(train_index, actual_values[:, 0], label='Actual')
plt.xlabel('Time')
plt.ylabel('Wave Height')
plt.legend()
plt.title('Predictions vs Actual Values')
plt.show()
