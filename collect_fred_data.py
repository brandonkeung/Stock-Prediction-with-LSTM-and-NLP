from fredapi import Fred
from datetime import date

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

if __name__ == "__main__":
    today = date.today()
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

    print(interest_rates)
    print(inflation_rates)
    print(gdp)
    print(unemployment_rates)
    print(cci)