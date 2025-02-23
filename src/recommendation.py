import sys
import os

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.file_io import save_pickle, load_pickle, load_csv, load_numpy_array
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_io import load_pickle, save_pickle,save_numpy_array
from config import PROCESSED_DATA_PATH, TFIDF_VECTORIZER_PATH, SIMILARITY_MATRIX_PATH, FEATURE_WEIGHT, SENTIMENT_WEIGHT

def create_similarity_matrix():
    """
    Create an enhanced similarity matrix with TF-IDF and sentiment weighting.
    """
    df = load_pickle(PROCESSED_DATA_PATH)
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    save_pickle(tfidf, TFIDF_VECTORIZER_PATH)

    feature_similarity = cosine_similarity(tfidf_matrix)

    # Sentiment Similarity
    sentiment_features = np.column_stack((df['Avg_Sentiment'], df['Avg_Subjectivity']))
    sentiment_similarity = cosine_similarity(sentiment_features)

    # Weighted Combination
    combined_similarity = (FEATURE_WEIGHT * feature_similarity) + (SENTIMENT_WEIGHT * sentiment_similarity)

    # Save similarity matrix using NumPy
    save_numpy_array(combined_similarity, SIMILARITY_MATRIX_PATH)


    
    # Save similarity matrix
def load_similarity_matrix():
    """
    Load the saved similarity matrix.
    """
    return load_numpy_array(SIMILARITY_MATRIX_PATH)


def get_recommendations(product_idx, n_recommendations=5):
    """
    Returns top recommendations for a given product.
    """
    df = load_pickle(PROCESSED_DATA_PATH)
    similarity_matrix = load_similarity_matrix()
    print(similarity_matrix.shape)  # Should be (num_products, num_products)
    print(similarity_matrix[product_idx])  # Should be a list/array of similarity scores

    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    product_indices = [i[0] for i in sim_scores]
    print(df.iloc[product_indices])
    recommendations = df.iloc[product_indices][['Brand', 'Label', 'Price', 'Category', 'Skin_Type', 'Benefits', 'Avg_Sentiment','sentiment_label']]
    recommendations['Similarity Score'] = [i[1] for i in sim_scores]

    return recommendations

#if __name__ == "__main__":
   #create_similarity_matrix()