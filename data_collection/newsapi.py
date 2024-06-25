import pandas as pd
from datetime import date, datetime, timedelta
import requests
import tqdm

# NOT IN USE NEWSAPI ONLY HAS DATA 1 MONTH IN THE PAST


def fetch_news(api_key, query, from_date, to_date):
    """Fetch news articles from the News API based on a query and date range."""
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'apiKey': api_key,
        'sortBy': 'publishedAt',  # Sort by publication date
        'language': 'en'  # Assuming you want articles in English
    }
    response = requests.get(url, params=params)
    return response.json()

def news_api(key_filepath, query, start_date, end_date):
    api_key = load_api_key(key_filepath)
    news_articles = fetch_news(api_key, query, start_date, end_date)
    
    data = []
    print(news_articles)
    for article in tqdm(news_articles['articles']):
        if article['title'] != "[Removed]":
            date_only = article['publishedAt'].split('T')[0]
            data.append({'headline': article['title'], 'date': date_only})

    df = pd.DataFrame(data, columns=['headline', 'date'])

    return df

if __name__=='__main__':
    news_df = news_api('news_api_key.txt', 'Apple', '1999-11-01', '2015-01-01')

    # print(f"the data: {news_df}")
    news_df.to_csv('alpaca_articles.csv', index=False)