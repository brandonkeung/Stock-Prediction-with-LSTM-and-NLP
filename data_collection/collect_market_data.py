from fredapi import Fred
from datetime import date
import yfinance as yf
import pandas as pd

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

if __name__ == "__main__":
    today = date.today()
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

    # Interest Rates
    interest_rates = fred.get_series('DGS10', start='1999-01-01', end=today)
    interest_rates = interest_rates.reset_index()
    interest_rates.columns = ['Date', 'Value_interest']
    # Ensure 'Date' column is in datetime format
    interest_rates['Date'] = pd.to_datetime(interest_rates['Date'])
    # Filter the DataFrame to include only rows from '1999-01-04' onwards
    interest_rates = interest_rates[interest_rates['Date'] >= '1999-01-01']
    interest_rates.iloc[0, interest_rates.columns.get_loc('Value_interest')] = 4.69

    # Inflation Rates
    inflation_rates = fred.get_series('CPIAUCSL', start='1999-01-01', end=today)
    inflation_rates = inflation_rates.reset_index()
    inflation_rates.columns = ['Date', 'Value_inflation']
    # Ensure 'Date' column is in datetime format
    inflation_rates['Date'] = pd.to_datetime(inflation_rates['Date'])
    # Filter the DataFrame to include only rows from '1999-01-04' onwards
    inflation_rates = inflation_rates[inflation_rates['Date'] >= '1999-01-01']

    # GDP
    gdp = fred.get_series('GDP', start='1999-01-01', end=today)
    gdp = gdp.reset_index()
    gdp.columns = ['Date', 'Value_gdp']
    # Ensure 'Date' column is in datetime format
    gdp['Date'] = pd.to_datetime(gdp['Date'])
    # Filter the DataFrame to include only rows from '1999-01-04' onwards
    gdp = gdp[gdp['Date'] >= '1999-01-01']


    # Unemployment Rates
    unemployment_rates = fred.get_series('UNRATE', start='1999-01-01', end=today)
    unemployment_rates = unemployment_rates.reset_index()
    unemployment_rates.columns = ['Date', 'Value_unemployment']
    # Ensure 'Date' column is in datetime format
    unemployment_rates['Date'] = pd.to_datetime(unemployment_rates['Date'])
    # Filter the DataFrame to include only rows from '1999-01-04' onwards
    unemployment_rates = unemployment_rates[unemployment_rates['Date'] >= '1999-01-01']


    # Consumer Confidence Index (CCI)
    cci = fred.get_series('UMCSENT', start='1999-01-01', end=today)
    cci = cci.reset_index()
    cci.columns = ['Date', 'Value_cci']
    # Ensure 'Date' column is in datetime format
    cci['Date'] = pd.to_datetime(cci['Date'])
    # Filter the DataFrame to include only rows from '1999-01-04' onwards
    cci = cci[cci['Date'] >= '1999-01-01']

    sp500 = sp500[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500 = sp500[sp500['Date'] >= '1999-01-01']
    nasdaq = nasdaq[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    nasdaq['Date'] = pd.to_datetime(nasdaq['Date'])
    nasdaq = nasdaq[nasdaq['Date'] >= '1999-01-01']
    dow_jones = dow_jones[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    dow_jones['Date'] = pd.to_datetime(dow_jones['Date'])
    dow_jones = dow_jones[dow_jones['Date'] >= '1999-01-01']
    tech_sector = tech_sector[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    tech_sector['Date'] = pd.to_datetime(tech_sector['Date'])
    tech_sector = tech_sector[tech_sector['Date'] >= '1999-01-01']

    sp500 = sp500.set_axis(['Date', 'Open_sp500', 'High_sp500', 'Low_sp500', 'Close_sp500', 'Volume_sp500'], axis=1)
    nasdaq = nasdaq.set_axis(['Date', 'Open_nasdaq', 'High_nasdaq', 'Low_nasdaq', 'Close_nasdaq', 'Volume_nasdaq'], axis=1)
    dow_jones = dow_jones.set_axis(['Date', 'Open_dow_jones', 'High_dow_jones', 'Low_dow_jones', 'Close_dow_jones', 'Volume_dow_jones'], axis=1)
    tech_sector = tech_sector.set_axis(['Date', 'Open_tech_sector', 'High_tech_sector', 'Low_tech_sector', 'Close_tech_sector', 'Volume_tech_sector'], axis=1)

    all_dates = pd.concat([
        interest_rates['Date'],
        inflation_rates['Date'],
        gdp['Date'],
        unemployment_rates['Date'],
        cci['Date'],
        sp500['Date'],
        nasdaq['Date'],
        dow_jones['Date'],
        tech_sector['Date']
    ]).drop_duplicates().sort_values().reset_index(drop=True)

    # Create a DataFrame to start merging
    df_merged = pd.DataFrame({'Date': all_dates})

    # Merge with economic indicators
    df_merged = pd.merge(df_merged, interest_rates, on='Date', how='left')
    df_merged = pd.merge(df_merged, inflation_rates, on='Date', how='left')
    df_merged = pd.merge(df_merged, gdp, on='Date', how='left')
    df_merged = pd.merge(df_merged, unemployment_rates, on='Date', how='left')
    df_merged = pd.merge(df_merged, cci, on='Date', how='left')

    # Merge with market data
    df_merged = pd.merge(df_merged, sp500, on='Date', how='left')
    df_merged = pd.merge(df_merged, nasdaq, on='Date', how='left')
    df_merged = pd.merge(df_merged, dow_jones, on='Date', how='left')
    df_merged = pd.merge(df_merged, tech_sector, on='Date', how='left')

    columns_to_fill = ['Value_inflation', 'Value_gdp', 'Value_unemployment', 'Value_cci']
    df_merged[columns_to_fill] = df_merged[columns_to_fill].fillna(method='ffill')

    print(df_merged)

    df_merged = df_merged[df_merged['Date'] >= '1999-01-04']

    df_merged = df_merged.dropna(how='any')

    print(df_merged)

    print(df_merged.isnull().sum())

    df_merged.to_csv(f"./data/market_data/market_data_{today}.csv")
