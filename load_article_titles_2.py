import pandas as pd
from datetime import date
import requests

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

# def load_articles_to_csv(stock_symbol): 
#     df = None   # delete
#     # df = get_data(api_key_path, stock_symbol)
#     if df is not None and not df.empty:
#         today = date.today()
#         csv_filename = f"data/{stock_symbol}_{today}.csv"
#         df.to_csv(csv_filename, index=False)
#         print(f"Data saved to {csv_filename}")
#     else:
#         print("No data to save.")

def get_articles(key_filepath, secret_filepath, symbol, start_date):
    # Load the API key and secret from the provided file paths
    API_KEY = load_api_key(key_filepath)
    API_SECRET = load_api_key(secret_filepath)

    # Define the endpoint for Alpaca news
    BASE_URL = 'https://data.alpaca.markets'
    NEWS_ENDPOINT = f'{BASE_URL}/v1beta1/news'

    # Initialize an empty list to hold the article data
    articles = []
    current_date = start_date

    # while True:
    # Set the parameters for the request
    params = {
        'symbols': f'{symbol}',  # Specify the stock symbol
        'limit': 50,  # Number of articles to retrieve
        'start': current_date  # Start date for the news articles
    }

    # Set the headers for the request
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }

    # Make the request to the Alpaca API
    response = requests.get(NEWS_ENDPOINT, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        news_data = response.json()
        if not news_data['news']:
            # Break the loop if there are no more articles
            # break
            pass

        # Extract the headline and date for each article
        print(news_data['news'][0]['created_at'].split('T')[0])

        for article in news_data['news']:
            headline = article['headline']
            date = article['created_at'].split('T')[0]  # Extract the date part
            articles.append({'headline': headline, 'date': date})

        # Update the current date to the date of the last article retrieved
        current_date = news_data['news'][-1]['created_at']
    else:
        print(f'Failed to retrieve news articles: {response.status_code}')
        print(response.json())
        # break
    
    # Create a DataFrame from the list of articles
    df = pd.DataFrame(articles)
    
    return df

if __name__ == '__main__':
    # Define the start date
    start_date = '2024-05-01T00:00:00Z'  # Starting point for the news articles

    # Get the articles
    df = get_articles('alpaca_key.txt', 'alpaca_secret_key.txt', 'AAPL', start_date)

    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv('test_articles.csv', index=False)

    # load_articles_to_csv('AAPL')