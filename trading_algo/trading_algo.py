import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from datetime import date
import yfinance as yf
import datetime as date
from fredapi import Fred

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key


def build_model(window_size, feature_count, lstm_units, d, dense_units):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(window_size, feature_count), return_sequences=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dropout(d))
    model.add(Dense(dense_units, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    return model

def predict_aapl():
    # window_size: 60
    # lstm_unit: 64
    # lstm_dropout: 0.2
    # dense_units: 16
    # batch_size: 16
    # epochs: 200
    # training_error_rate: 0.00023464551486540586
    # training_accuracy: 0.006090724840760231
    # testing_error_rate: 2.2286689272732474e-05
    # testing_accuracy: 0.003547884291037917
    # MASE: 1.321271300315857
    model = build_model(60, 9, 64, 0.2, 16)
    model.load_weights('../weights/AAPL_best_model.weights.h5')
    return model

def predict_amzn():
    # window_size: 15
    # lstm_unit: 64
    # lstm_dropout: 0.1
    # dense_units: 16
    # batch_size: 16
    # epochs: 100
    # training_error_rate: 4.6920404201955535e-06
    # training_accuracy: 0.0013190218014642596
    # testing_error_rate: 0.0006488984799943864
    # testing_accuracy: 0.019547905772924423
    # MASE: 1.7842113971710205
    model = build_model(15, 9, 64, 0.1, 16)
    model.load_weights('../weights/AMZN_best_model.weights.h5')
    return model

def predict_googl():
    # window_size: 15
    # lstm_unit: 32
    # lstm_dropout: 0.2
    # dense_units: 16
    # batch_size: 16
    # epochs: 50
    # training_error_rate: 0.00014775595627725124
    # training_accuracy: 0.00714624160900712
    # testing_error_rate: 0.00036536858533509076
    # testing_accuracy: 0.013242176733911037
    # MASE: 1.8139164447784424
    model = build_model(15, 9, 32, 0.2, 16)
    model.load_weights('../weights/GOOGL_best_model.weights.h5')
    return model

def predict_msft():
    # window_size: 15
    # lstm_unit: 64
    # lstm_dropout: 0.1
    # dense_units: 32
    # batch_size: 32
    # epochs: 100
    # training_error_rate: 0.00011800717038568109
    # training_accuracy: 0.006529844831675291
    # testing_error_rate: 0.00017038073565345258
    # testing_accuracy: 0.008951040916144848
    # MASE: 1.142071008682251
    model = build_model(15, 9, 64, 0.1, 32)
    model.load_weights('../weights/MSFT_best_model.weights.h5')
    return np.random.uniform(300, 310)  # Replace with your model's prediction

def predict_nvda():
    # window_size: 15
    # lstm_unit: 32
    # lstm_dropout: 0.3
    # dense_units: 32
    # batch_size: 16
    # epochs: 50
    # training_error_rate: 0.00010817285510711372
    # training_accuracy: 0.004988082218915224
    # testing_error_rate: 0.0010140140075236559
    # testing_accuracy: 0.020503859966993332
    # MASE: 1.3169596195220947
    model = build_model(15, 9, 32, 0.3, 32)
    model.load_weights('../weights/NVDA_best_model.weights.h5')
    return model  # Replace with your model's prediction
    
# 'Close_sp500', 'Close_nasdaq', 
# 'Close_dow_jones','Close_tech_sector',
# 'open', 'high', 'low', 
# 'close', 'volume'

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

def get_data(api_key_path, stock_symbol, days):
    api_key = load_api_key(api_key_path)
    
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&datatype=csv&apikey={api_key}'
    try:
        data = pd.read_csv(url)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp').reset_index(drop=True)  # Ensure chronological order
        
        # Filter data for the past `days` days
        end_date = data['timestamp'].iloc[-1]
        start_date = end_date - date.timedelta(days=days)
        filtered_data = data[data['timestamp'] >= start_date]
        
        print(f"Data retrieved successfully for {stock_symbol}.")
        return filtered_data
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        return None


def load_data_to_csv(api_key_path, stock_symbol):
    df = get_data(api_key_path, stock_symbol)
    if df is not None and not df.empty:
        today = date.today()
        csv_filename = f"data/{stock_symbol}/{stock_symbol}_{today}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")



def main():
    today = date.today()
    api_key = load_api_key("./keys/fred_api.txt")
    # You need an API key from FRED
    fred = Fred(api_key=api_key)

    # Download data for S&P 500
    end_date = today
    start_date = end_date - date.timedelta(days=60)
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    nasdaq = yf.download('^IXIC', start=start_date, end=end_date)
    dow_jones = yf.download('^DJI', start=start_date, end=end_date)
    tech_sector = yf.download('XLK', start=start_date, end=end_date)

    sp500.reset_index(inplace=True)
    nasdaq.reset_index(inplace=True)
    dow_jones.reset_index(inplace=True)
    tech_sector.reset_index(inplace=True)

    

    AAPL_df = get_data('keys/alphavantage_api_key.txt', 'AAPL', 60)
    MSFT_df = get_data('keys/alphavantage_api_key.txt', 'MSFT', 15)
    GOOGL_df = get_data('keys/alphavantage_api_key.txt', 'GOOGL', 15)
    AMZN_df = get_data('keys/alphavantage_api_key.txt', 'AMZN', 15)
    NVDA_df = get_data('keys/alphavantage_api_key.txt', 'NVDA', 15)

    api_key = load_api_key("./keys/fred_api.txt")
    # You need an API key from FRED
    fred = Fred(api_key=api_key)

    # Download data for S&P 500
    sp500 = yf.download('^GSPC', start='1999-01-01', end=today)
    nasdaq = yf.download('^IXIC', start='1999-01-01', end=today)
    dow_jones = yf.download('^DJI', start='1999-01-01', end=today)

    # Download sector data
    tech_sector = yf.download('XLK', start='1999-01-01', end=today)

    sp500.reset_index(inplace=True)
    nasdaq.reset_index(inplace=True)
    dow_jones.reset_index(inplace=True)
    tech_sector.reset_index(inplace=True)

    sp500 = sp500[['Date', 'Close']]
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500 = sp500[sp500['Date'] >= '1999-01-01']
    nasdaq = nasdaq[['Date', 'Open', 'Close']]
    nasdaq['Date'] = pd.to_datetime(nasdaq['Date'])
    nasdaq = nasdaq[nasdaq['Date'] >= '1999-01-01']
    dow_jones = dow_jones[['Date', 'Open', 'Close']]
    dow_jones['Date'] = pd.to_datetime(dow_jones['Date'])
    dow_jones = dow_jones[dow_jones['Date'] >= '1999-01-01']
    tech_sector = tech_sector[['Date', 'Open', 'Close']]
    tech_sector['Date'] = pd.to_datetime(tech_sector['Date'])
    tech_sector = tech_sector[tech_sector['Date'] >= '1999-01-01']

    sp500 = sp500.set_axis(['Date', 'Close_sp500'], axis=1)
    nasdaq = nasdaq.set_axis(['Date', 'Close_nasdaq',], axis=1)
    dow_jones = dow_jones.set_axis(['Date', 'Close_dow_jones'], axis=1)
    tech_sector = tech_sector.set_axis(['Date', 'Close_tech_sector'], axis=1)

    all_dates = pd.concat([
        sp500['Date'],
        nasdaq['Date'],
        dow_jones['Date'],
        tech_sector['Date']
    ]).drop_duplicates().sort_values().reset_index(drop=True)

    # Create a DataFrame to start merging
    market_df = pd.DataFrame({'Date': all_dates})

    # Merge with market data
    market_df = pd.merge(market_df, sp500, on='Date', how='left')
    market_df = pd.merge(market_df, nasdaq, on='Date', how='left')
    market_df = pd.merge(market_df, dow_jones, on='Date', how='left')
    market_df = pd.merge(market_df, tech_sector, on='Date', how='left')

    market_df = market_df[market_df['Date'] >= '1999-01-04']

    market_df = market_df.dropna(how='any')

    # 'Close_sp500', 'Close_nasdaq', 
    # 'Close_dow_jones','Close_tech_sector',

if __name__ == "__main__":
    main()