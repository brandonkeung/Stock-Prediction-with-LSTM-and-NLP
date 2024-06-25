import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def split_data(data, training_split, features):
  split_row = int(data.shape[0]*training_split)
  training_set = data[features].iloc[:split_row].values
  testing_set = data[features].iloc[split_row:].values
  return training_set, testing_set

def get_x_y(dataset, window_size, label_feature, feature_count):
  X, y = [], []
  for i in range(window_size, len(dataset)):
    X.append(dataset[i-window_size:i])
    y.append(dataset[i, label_feature])

  X, y = np.array(X), np.array(y)
  X = np.reshape(X, (X.shape[0], window_size, feature_count))
  return X, y

def build_model(window_size, feature_count):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(window_size, feature_count), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(d))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    return model


def main(symbol):
    company_df = pd.read_csv(f'data/{symbol}/{symbol}_2024-06-24.csv')

    market_df

    training_set, testing_set = split_data(df, 0.8, ['open', 'high', 'low', 'close', 'volume'])
    print("training_set: ", training_set.shape)
    print("testing_set: ", testing_set.shape)

    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)
    testing_set = scaler.fit_transform(testing_set)


    X_train, y_train = get_x_y(training_set, 14, 3, 5)      # Change this line if you added more features
    val_split_row = int(X_train.shape[0]*0.8)               # 20% will be used for validation
    X_train, X_val = X_train[:val_split_row], X_train[val_split_row:]
    y_train, y_val = y_train[:val_split_row], y_train[val_split_row:]
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_val: ", X_val.shape)
    print("y_val: ", y_val.shape)

    window_size = 14        # Example window size
    feature_count = 5       # Number of features (e.g., open, high, low, close, volume)
    model = build_model(window_size, feature_count)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=200, batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Get X and y from testing set
    X_test, y_test = get_x_y(testing_set, 14, 3, 5)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    y_pred = model.predict(X_test)
    
    full_test_set = np.zeros((len(y_test), feature_count))
    full_test_set[:, 3] = y_test
    y_test_inverse = scaler.inverse_transform(full_test_set)[:, 3]

    full_pred_set = np.zeros((len(y_pred), feature_count))
    full_pred_set[:, 3] = y_pred[:, 0]  # Ensure y_pred is 2D
    y_pred_inverse = scaler.inverse_transform(full_pred_set)[:, 3]

    print("Inverted (using stock prices)")
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    print(f'Mean Absolute Error (MAE): {mae}')

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    print(f'Mean Squared Error (MSE): {mse}')

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    print("Using MinMaxScaler")
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {mae}')

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse}')

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test)
    print("test loss, test acc:", results)

    model.save_weights(f'weights/{symbol}_model2.weights.h5')


if __name__ == "__main__":
   main("AAPL")