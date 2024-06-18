import pandas as pd
from datetime import date

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

def get_data(api_key_path, stock_symbol):
    api_key = load_api_key(api_key_path)
    
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&datatype=csv&apikey={api_key}'
    try:
        data = pd.read_csv(url)
        data = data.iloc[::-1]  # Reverse the data for chronological order
        print(f"Data retrieved successfully for {stock_symbol}.")
        return data
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        return None

def load_data_to_csv(api_key_path, stock_symbol):
    df = get_data(api_key_path, stock_symbol)
    if df is not None and not df.empty:
        today = date.today()
        csv_filename = f"data/{stock_symbol}_{today}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")

if __name__ == '__main__':
    # test code:
    load_data_to_csv('alphavantage_api_key.txt', 'AAPL')
    load_data_to_csv('alphavantage_api_key.txt', 'MSFT')
    load_data_to_csv('alphavantage_api_key.txt', 'GOOGL')
    load_data_to_csv('alphavantage_api_key.txt', 'AMZN')
    load_data_to_csv('alphavantage_api_key.txt', 'NVDA')

