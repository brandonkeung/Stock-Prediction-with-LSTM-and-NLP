import yfinance as yf
import pandas as pd
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

    sp500.reset_index(inplace=True)
    nasdaq.reset_index(inplace=True)
    dow_jones.reset_index(inplace=True)
    tech_sector.reset_index(inplace=True)

    print(sp500.isnull().sum())
    print(nasdaq.isnull().sum())
    print(dow_jones.isnull().sum())
    print(tech_sector.isnull().sum())


    if sp500 is not None and not sp500.empty:
        today = date.today()
        csv_filename = f"data/market_data/sp500_{today}.csv"
        sp500.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")

    if nasdaq is not None and not nasdaq.empty:
        today = date.today()
        csv_filename = f"data/market_data/nasdaq_{today}.csv"
        nasdaq.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")

    if dow_jones is not None and not dow_jones.empty:
        today = date.today()
        csv_filename = f"data/market_data/dow_jones_{today}.csv"
        dow_jones.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")

    if tech_sector is not None and not tech_sector.empty:
        today = date.today()
        csv_filename = f"data/market_data/tech_sector_{today}.csv"
        tech_sector.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")
    else:
        print("No data to save.")

    

    

