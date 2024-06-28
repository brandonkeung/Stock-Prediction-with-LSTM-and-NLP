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

def MASE(Actual, Predicted):
    n = len(Actual)
    
    # Calculate the mean absolute error of the naive forecast
    naive_errors = [abs(Actual[i] - Actual[i-1]) for i in range(1, n)]
    naive_mae = np.mean(naive_errors)
    
    # Calculate the MASE
    mase_values = [abs(Actual[i] - Predicted[i]) / naive_mae for i in range(n)]
    
    return np.mean(mase_values)

def grid_search_build_model(window_size, feature_count, lstm_units, d, dense_units):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(window_size, feature_count), return_sequences=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dropout(d))
    model.add(Dense(dense_units, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    return model


def main(symbol):
    company_df = pd.read_csv(f'./data/{symbol}/{symbol}_2024-06-24.csv')

    market_df = pd.read_csv(f'./data/market_data/market_data_2024-06-24.csv')
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    company_df['timestamp'] = pd.to_datetime(company_df['timestamp'])
    df = pd.merge(market_df, company_df, left_on='Date', right_on='timestamp', how='inner')

    # training_set, testing_set = split_data(df, 0.8, [
    #     'Value_interest', 'Value_inflation', 'Value_gdp',
    #     'Value_unemployment', 'Value_cci', 'Open_sp500', 'High_sp500',
    #     'Low_sp500', 'Close_sp500', 'Volume_sp500', 'Open_nasdaq',
    #     'High_nasdaq', 'Low_nasdaq', 'Close_nasdaq', 'Volume_nasdaq',
    #     'Open_dow_jones', 'High_dow_jones', 'Low_dow_jones', 'Close_dow_jones',
    #     'Volume_dow_jones', 'Open_tech_sector', 'High_tech_sector',
    #     'Low_tech_sector', 'Close_tech_sector', 'Volume_tech_sector',
    #     'open', 'high', 'low', 'close', 'volume'
    #     ])

    training_set, testing_set = split_data(df, 0.8, [
        'Close_sp500', 'Close_nasdaq', 
        'Close_dow_jones','Close_tech_sector',
        'open', 'high', 'low', 
        'close', 'volume'])
    
    print("training_set: ", training_set.shape)
    print("testing_set: ", testing_set.shape)

    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)
    testing_set = scaler.fit_transform(testing_set)

    # window_size = 14        # Example window size
    feature_count = 9       # Number of features (e.g., open, high, low, close, volume)
    label_index = 7
    window_size = [15, 30, 60]
    lstm_units = [32, 64]           # lstm_units = [128, 256]       DO THIS LATER SO THAT WE SPLIT IT IN HALF JUST IN CASE SOMETHING HAPPENS
    lstm_dropout = [0.1, 0.2, 0.3]
    dense_units = [16, 32]
    batch_size = [16, 32, 64]
    epochs = [50, 100, 200]
    # window_size = [15]
    # lstm_units = [32]
    # lstm_dropout = [0.1]
    # dense_units = [16]
    # batch_size = [16]
    # epochs = [25, 50]

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    dictionary_of_training = {'window' : [], 'lstm_unit': [], 'lstm_dropout': [], 'dense_units' : [], 'batch_size': [], 'epochs': [], 'training_error_rate': [], 'training_accuracy': [], 'testing_error_rate': [], 'testing_accuracy' : [], 'MASE' : [], 'history' : [], 'model' : []}
    for window in window_size:
        for unit in lstm_units:
            for dropout in lstm_dropout:
                for dense_unit in dense_units:
                    for batch in batch_size:
                        for epoch in epochs: 
                            X_train, y_train = get_x_y(training_set, window, label_index, feature_count)      # Change this line if you added more features
                            val_split_row = int(X_train.shape[0]*0.8)               # 20% will be used for validation
                            X_train, X_val = X_train[:val_split_row], X_train[val_split_row:]
                            y_train, y_val = y_train[:val_split_row], y_train[val_split_row:]
                            print("X_train: ", X_train.shape)
                            print("y_train: ", y_train.shape)
                            print("X_val: ", X_val.shape)
                            print("y_val: ", y_val.shape)
                            # Add current hyperparameters to the dictionary
                            dictionary_of_training['window'].append(window)
                            dictionary_of_training['lstm_unit'].append(unit)
                            dictionary_of_training['lstm_dropout'].append(dropout)
                            dictionary_of_training['dense_units'].append(dense_unit)
                            dictionary_of_training['batch_size'].append(batch)
                            dictionary_of_training['epochs'].append(epoch)

                            # Create and compile the model
                            model = grid_search_build_model(window, feature_count, unit, dropout, dense_unit)

                            # Train the model
                            history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch,
                                                validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopping, reduce_lr])

                            # Evaluate the model on training data
                            train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
                            dictionary_of_training['training_error_rate'].append(train_loss)
                            dictionary_of_training['training_accuracy'].append(train_accuracy)

                            # Evaluate the model on validation data
                            test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
                            dictionary_of_training['testing_error_rate'].append(test_loss)
                            dictionary_of_training['testing_accuracy'].append(test_accuracy)

                            X_test, y_test = get_x_y(testing_set, window, label_index, feature_count)

                            y_pred = model.predict(X_test)
    
                            full_test_set = np.zeros((len(y_test), feature_count))
                            full_test_set[:, label_index] = y_test
                            y_test_inverse = scaler.inverse_transform(full_test_set)[:, label_index]

                            full_pred_set = np.zeros((len(y_pred), feature_count))
                            full_pred_set[:, label_index] = y_pred[:, 0]  # Ensure y_pred is 2D
                            y_pred_inverse = scaler.inverse_transform(full_pred_set)[:, label_index]

                            mae = mean_absolute_error(y_test, y_pred)

                            y_test = np.array(y_test)  # Convert to numpy array if not already
                            y_pred = np.array(y_pred)  # Convert to numpy array if not already

                            # Calculate MASE
                            mase = MASE(y_test, y_pred)
                            dictionary_of_training['MASE'].append(mase)

                            dictionary_of_training['model'].append(model)
                            dictionary_of_training['history'].append(history)

                            # Print progress
                            print(f"Completed training for window={window}, unit={unit}, dropout={dropout}, dense_units={dense_unit}, batch_size={batch}, epochs={epoch}, training_error_rate={train_loss}, training_accuracy={train_accuracy}, testing_error_rate={test_loss}, testing_accuracy={test_accuracy}, MASE: {mase}\n")

                            f = open("ML/model_performances_grid_search.txt", "a")
                            f.write(f"Completed training for window={window}, unit={unit}, dropout={dropout}, dense_units={dense_unit}, batch_size={batch}, epochs={epoch}, training_error_rate={train_loss}, training_accuracy={train_accuracy}, testing_error_rate={test_loss}, testing_accuracy={test_accuracy}, MASE: {mase}\n")
                            f.close()


    max_index = np.argmin(dictionary_of_training['MASE'])

    # Print the values corresponding to that index
    f = open("ML/model_performances_grid_search.txt", "a")
    f.write("Best parameters based on lowest MASE:\n")
    f.write(f"window_size: {dictionary_of_training['window'][max_index]}\n")
    f.write(f"lstm_unit: {dictionary_of_training['lstm_unit'][max_index]}\n")
    f.write(f"lstm_dropout: {dictionary_of_training['lstm_dropout'][max_index]}\n")
    f.write(f"dense_units: {dictionary_of_training['dense_units'][max_index]}\n")
    f.write(f"batch_size: {dictionary_of_training['batch_size'][max_index]}\n")
    f.write(f"epochs: {dictionary_of_training['epochs'][max_index]}\n")
    f.write(f"training_error_rate: {dictionary_of_training['training_error_rate'][max_index]}\n")
    f.write(f"training_accuracy: {dictionary_of_training['training_accuracy'][max_index]}\n")
    f.write(f"testing_error_rate: {dictionary_of_training['testing_error_rate'][max_index]}\n")
    f.write(f"testing_accuracy: {dictionary_of_training['testing_accuracy'][max_index]}\n")
    f.write(f"MASE: {dictionary_of_training['MASE'][max_index]}\n")

    model = dictionary_of_training['model'][max_index]
    
    # Get X and y from testing set
    X_test, y_test = get_x_y(testing_set, window, label_index, feature_count)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    y_pred = model.predict(X_test)
    
    full_test_set = np.zeros((len(y_test), feature_count))
    full_test_set[:, label_index] = y_test
    y_test_inverse = scaler.inverse_transform(full_test_set)[:, label_index]

    full_pred_set = np.zeros((len(y_pred), feature_count))
    full_pred_set[:, label_index] = y_pred[:, 0]  # Ensure y_pred is 2D
    y_pred_inverse = scaler.inverse_transform(full_pred_set)[:, label_index]

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

    model.save_weights(f'weights/{symbol}_best_model.weights.h5')


if __name__ == "__main__":
#    main("AAPL")
   main("AMZN")
#    main("GOOGL")
#    main("MSFT")
#    main("NVDA")

   # MAKE SURE TO CHECK THE DATASETS NO NULL VALUES AND THE ENGOUH DATAPOINTS