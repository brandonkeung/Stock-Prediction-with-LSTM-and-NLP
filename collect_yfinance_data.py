import yfinance as yf
from datetime import date

if __name__ == "__main__":
    # Returns the current local date
    today = date.today()
    # Download data for S&P 500
    sp500 = yf.download('^GSPC', start='1999-01-01', end=today)
    nasdaq = yf.download('^IXIC', start='1999-01-01', end=today)
    dow_jones = yf.download('^DJI', start='1999-01-01', end=today)

    # Download sector data
    tech_sector = yf.download('XLK', start='1999-01-01', end=today)

    # Download specific industry data
    industry_data = yf.download('AAPL', start='1999-01-01', end=today)
    print(sp500)
    print(nasdaq)
    print(dow_jones)
    print(tech_sector)
    print(industry_data)

