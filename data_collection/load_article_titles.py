import pandas as pd
from datetime import date, datetime, timedelta
import requests
import tqdm

def load_api_key(filepath):
    """Load and return the API key from a file."""
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes any leading/trailing whitespace
    return api_key

def get_articles(key_filepath, secret_filepath, symbol, start_date):
    # Load the API key and secret from the provided file paths
    API_KEY = load_api_key(key_filepath)
    API_SECRET = load_api_key(secret_filepath)

    # Define the endpoint for Alpaca news
    BASE_URL = 'https://data.alpaca.markets'
    NEWS_ENDPOINT = f'{BASE_URL}/v1beta1/news'

    # Initialize an empty list to hold the article data
    articles = []
    end_date = datetime.utcnow().isoformat() + 'Z'  # Current date and time in ISO format with Z suffix

    while True:
        # Set the parameters for the request
        params = {
            'symbols': f'{symbol}',  # Specify the stock symbol
            'limit': 50,  # Number of articles to retrieve
            'start': start_date,  # Start date for the news articles
            'end': end_date  # End date for the news articles
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
                break
            
            print(news_data['news'][0]['created_at'].split('T')[0])

            # Extract the headline and date for each article
            for article in news_data['news']:
                headline = article['headline']
                date = article['created_at'].split('T')[0]  # Extract the date part
                articles.append({'headline': headline, 'date': date})

            # Update the end date to one day before the date of the oldest article retrieved
            oldest_article_date = news_data['news'][-1]['created_at']
            end_date = (datetime.fromisoformat(oldest_article_date.rstrip('Z')) - timedelta(days=1)).isoformat() + 'Z'
        else:
            print(f'Failed to retrieve news articles: {response.status_code}')
            print(response.json())
            break
    
    # Create a DataFrame from the list of articles
    df = pd.DataFrame(articles)
    
    return df


if __name__ == '__main__':
    # Define the start date
    start_date = '1999-11-01T00:00:00Z'  # Starting point for the news articles

    symbol = 'AAPL'
    # Get the articles
    alpaca_df = get_articles('alpaca_key.txt', 'alpaca_secret_key.txt', symbol, start_date)

    print(alpaca_df)

    # Save the DataFrame to a CSV file
    alpaca_df.to_csv(f"data/{symbol}_articles.csv", index=False)