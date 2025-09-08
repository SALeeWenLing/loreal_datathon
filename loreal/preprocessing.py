import re
import ast
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    @staticmethod
    def clean_comment(text):
        '''Cleans the input comment text by removing non-alphabetic characters and tokenizing.'''
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)

    @staticmethod
    def extract_topic_names(topic_categories):
        '''Extracts and cleans topic names from Wikipedia URLs or lists.'''
        if topic_categories is None or (isinstance(topic_categories, float) and pd.isna(topic_categories)):
            return ["Other"]
        if isinstance(topic_categories, list):
            urls = topic_categories
        else:
            try:
                parsed = ast.literal_eval(str(topic_categories))
                urls = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                urls = [str(topic_categories)]
        topics = []
        for url in urls if isinstance(urls, list) else []:
            if "/wiki/" in str(url):
                t = str(url).split("/wiki/")[-1].replace("_", " ")
                t = re.sub(r"\([^)]*\)", "", t).strip()
                if t:
                    topics.append(t)
        print("Topics extracted:", topics)
        return topics or ["Other"]

