import pandas as pd
from datetime import date, datetime, timedelta
import requests
import tqdm
from GoogleNews import GoogleNews






if __name__ == '__main__':
    googlenews = GoogleNews()
    googlenews.enableException(True)
    googlenews = GoogleNews(lang='en', region='US')
    googlenews = GoogleNews(start='11/01/1999',end='12/01/1999')
    googlenews.search('APPLE')        # google news section
    #googlenews.get_news('APPLE')        # news.google.com
    data = googlenews.results(sort=True)
    print(data)
    googlenews.clear()