import sys
import os

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.file_io import save_pickle, load_pickle, load_csv, load_numpy_array

import sys
import os
import pandas as pd
import numpy as np
import re
from utils.file_io import save_pickle, load_pickle
from config import DATA_PATH, PROCESSED_DATA_PATH

# Ensure project root is added to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)





import spacy
from textblob import TextBlob
from collections import Counter

class SentimentAnalyzer:
    """
    Performs sentiment analysis using TextBlob and extracts key aspects using spaCy.
    """
    @staticmethod
    def assignSentimentlabel(sentiment_score):
        """
        Assigns a sentiment label based on sentiment score.
        """
        if sentiment_score > 0.6:
            return "Positive"
        elif sentiment_score > 0.4:
            return "Neutral"
        else:
            return "Negative"

    def __init__(self):
        """
        Initializes spaCy NLP model and defines common skincare aspects.
        """
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy NLP model
        self.common_aspects = ['hydration', 'texture', 'price', 'smell', 'effectiveness', 'moisturizer', 'cleanser']

    def analyze_review(self, review):
        """
        Performs sentiment analysis and extracts key aspects from a review.
        """
        if not isinstance(review, str) or review.strip() == "":
            return {
                'sentiment_score': 0,
                'subjectivity': 0,
                'sentiment_label': "Neutral",
                'key_aspects': []
            }

        blob = TextBlob(review)
        doc = self.nlp(review.lower())

        aspects = [token.text for token in doc if token.text in self.common_aspects or token.pos_ in ["NOUN", "ADJ"]]
        aspect_count = Counter(aspects).most_common(3)

        sentiment_score = blob.sentiment.polarity

        # Classify sentiment
        if sentiment_score > 0:
            sentiment_label = "Positive"
        elif sentiment_score < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"


        return {
            'sentiment_score': sentiment_score,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': sentiment_label,
            'key_aspects': aspect_count
        }


def load_and_preprocess_data():
    """
    Loads skincare dataset, performs cleaning, sentiment analysis, and saves processed data.
    """
    df = pd.read_csv(DATA_PATH)

    # Ensure necessary columns exist
    required_columns = ['Brand', 'Label', 'Category', 'Price', 'Ingredients', 'Skin_Type', 'Benefits', 'Reviews']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Initialize SentimentAnalyzer
    analyzer = SentimentAnalyzer()

    # This should print sentiment analysis results

    df['Review_Analysis'] = df['Reviews'].apply(
    lambda reviews: [analyzer.analyze_review(review.strip()) for review in re.split(r'[|,]', reviews)] if isinstance(reviews, str) else []
    )

    
    # Calculate average sentiment scores
    df['Avg_Sentiment'] = df['Review_Analysis'].apply(
    lambda analyses: np.mean([analysis['sentiment_score'] for analysis in analyses]) if analyses else 0.0
    )

    df['Avg_Subjectivity'] = df['Review_Analysis'].apply(
    lambda analyses: np.mean([analysis['subjectivity'] for analysis in analyses]) if analyses else 0.0
    )

    
    df['sentiment_label'] = df['Avg_Sentiment'].apply(SentimentAnalyzer.assignSentimentlabel)

    df['combined_features'] = df.apply(
        lambda x: f"{x['Brand'].lower()} {x['Category'].lower()} {x['Ingredients'].lower()} {x['Skin_Type'].lower()} {x['Benefits'].lower()}",
        axis=1
    )


    # Save preprocessed data
    save_pickle(df, PROCESSED_DATA_PATH)
    
    # Save first 50 rows to CSV for quick inspection
    df.head(50).to_csv("output.csv", index=False)

    print("Data preprocessing completed! Processed data saved.")



#if __name__ == "__main__":
 #  load_and_preprocess_data()
