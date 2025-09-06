import pandas as pd
from textblob import TextBlob

class Analyzer:
    @staticmethod
    def get_sentiment(text):
        '''
        - Analyzes the sentiment of the given text using TextBlob.
        - Returns 'positive', 'negative', or 'neutral' based on polarity.
        '''
        if not text or not isinstance(text, str):
            return "neutral"
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def calc_soe(row):
        '''
        - Calculates the Strength of Engagement (SOE) for a video based on likes, comments, favorites, and views.
        - SOE is defined as (likes + comments + favorites) / views.
        - Returns SOE as a float; returns 0.0 if views are zero or undefined.
        '''
        total = (row["video_likeCount"] or 0) + (row["video_commentCount"] or 0) + (row["video_favCount"] or 0)
        views = row["video_viewCount"] or 0
        return float(total) / float(views) if views > 0 else 0.0

    @staticmethod
    def minmax_0_100(g: pd.Series) -> pd.Series:
        '''
        - Applies min-max normalization to a pandas Series.
        - Scales values to a range of 0 to 100.
        '''
        gmin, gmax = g.min(), g.max()
        if pd.isna(gmin) or pd.isna(gmax) or gmin == gmax:
            return pd.Series(50.0, index=g.index)
        return 100 * (g - gmin) / (gmax - gmin)
