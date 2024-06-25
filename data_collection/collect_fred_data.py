from fredapi import Fred
from datetime import date
import yfinance as yf

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

if __name__ == "__main__":
    today = date.today()

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

    api_key = load_api_key("keys/fred_api.txt")
    # You need an API key from FRED
    fred = Fred(api_key=api_key)

    # Interest Rates
    interest_rates = fred.get_series('DGS10', start='1999-01-01', end=today)
    # Inflation Rates
    inflation_rates = fred.get_series('CPIAUCSL', start='1999-01-01', end=today)
    # GDP
    gdp = fred.get_series('GDP', start='1999-01-01', end=today)
    # Unemployment Rates
    unemployment_rates = fred.get_series('UNRATE', start='1999-01-01', end=today)
    # Consumer Confidence Index (CCI)
    cci = fred.get_series('UMCSENT', start='1999-01-01', end=today)

    interest_rates = interest_rates.reset_index()
    interest_rates.columns = ['Date', 'Value']
    interest_rates

    inflation_rates = inflation_rates.reset_index()
    inflation_rates.columns = ['Date', 'Value']
    inflation_rates

    gdp = gdp.reset_index()
    gdp.columns = ['Date', 'Value']
    gdp

    unemployment_rates = unemployment_rates.reset_index()
    unemployment_rates.columns = ['Date', 'Value']
    unemployment_rates

    cci = cci.reset_index()
    cci.columns = ['Date', 'Value']
    cci

    print(interest_rates.isnull().sum())          # 695 nan
    print(inflation_rates.isnull().sum())         # 0 nan
    print(gdp.isnull().sum())                     # 4 nan
    print(unemployment_rates.isnull().sum())      # 0 nan
    print(cci.isnull().sum())                     # 210 nan

    interest_rates = interest_rates[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    inflation_rates = inflation_rates[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    gdp = gdp[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    unemployment_rates = unemployment_rates[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    cci = cci[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    sp500 = sp500[['Date_sp500', 'Open_sp500', 'High_sp500', 'Low_sp500', 'Close_sp500', 'Volume_sp500']]

    df = sp500.merge(nasdaq, how='inner', on='Date', suffixes=('', '_nasdaq'))
    df = df.merge(dow_jones, how='inner', on='Date', suffixes=('', '_dow_jones'))

    # Merge with macroeconomic data
    df = df.merge(interest_rates, how='inner', on='Date', suffixes=('', '_interest_rate'))
    df = df.merge(inflation_rates, how='inner', on='Date', suffixes=('', '_inflation_rate'))
    df = df.merge(gdp, how='inner', on='Date', suffixes=('', '_gdp'))
    df = df.merge(unemployment_rates, how='inner', on='Date', suffixes=('', '_unemployment_rate'))
    df = df.merge(cci, how='inner', on='Date', suffixes=('', '_cci'))

    print(df.columns)