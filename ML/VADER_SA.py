import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to get sentiments
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def main(symbol):
    df=pd.read_csv(f"data/{symbol}_articles.csv")
    print("read successfully")
    # Apply sentiment analysis
    df['sentiments'] = df['headline'].apply(get_sentiment)
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

    data_aggregated = []

    # Group by 'date'
    grouped = df.groupby('date')

    print("calculating sentiments")
    for date, group in grouped:
        # Initialize sentiment sums
        pos_sum = neu_sum = neg_sum = compound_sum = 0
        headlines = []

        # Loop through each article in the group
        for _, article in group.iterrows():
            sentiment = get_sentiment(article['headline'])
            pos_sum += sentiment['pos']
            neu_sum += sentiment['neu']
            neg_sum += sentiment['neg']
            compound_sum += sentiment['compound']
            headlines.append(article['headline'])

        # Calculate average sentiments for the group
        n = len(group)
        avg_pos = pos_sum / n
        avg_neu = neu_sum / n
        avg_neg = neg_sum / n
        avg_compound = compound_sum / n

        data_aggregated.append({
            "date": date,
            "article headlines": headlines,
            "pos": avg_pos,
            "neu": avg_neu,
            "neg": avg_neg,
            "compound": avg_compound
        })

    df_aggregated = pd.DataFrame(data_aggregated, columns=['date', 'article headlines', 'pos', 'neu', 'neg', 'compound'])

    df_aggregated['number of articles'] = df_aggregated['article headlines'].apply(lambda x: len(x))
    print("outputting csv")
    df_aggregated.to_csv(f'data/{symbol}_sent.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main('AAPL')