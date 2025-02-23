import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from collections import Counter

class SentimentAnalyzer:
    def __init__(self):
        self.common_aspects = ['hydration', 'texture', 'price', 'smell', 'effectiveness']

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
    
    def analyze_review(self, review):
        """
        Perform detailed sentiment analysis on a single review using TextBlob
        """
        # TextBlob analysis
        blob = TextBlob(review)
        
        # Extract key aspects mentioned
        words = review.lower().split()
        aspects = [word for word in words if word in self.common_aspects]
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
            'sentiment_score': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': sentiment_label,
            'key_aspects': aspect_count
        }

def create_sample_dataset():
    """
    Create a sample skincare dataset with reviews
    Returns: pandas DataFrame with sample product data including reviews
    """
    DATA_PATH = "C:/Users/Admin/Desktop/ProductRecom/data/skincare_products_reduced.csv"
    df = pd.read_csv(DATA_PATH)
    
    
    # Calculate sentiment scores for reviews
    analyzer = SentimentAnalyzer()
    import re

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


    return df

@st.cache_data
def load_data():
    """
    Load and preprocess the skincare products dataset
    Returns: pandas DataFrame with preprocessed product data including sentiment analysis
    """
    df = create_sample_dataset()
    return df

def create_similarity_matrix(data, feature_weight=0.7, sentiment_weight=0.3):
    """
    Create enhanced similarity matrix with adjustable weights
    """
    # Feature-based similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    feature_similarity = cosine_similarity(tfidf_matrix)
    
    # Sentiment similarity
    sentiment_features = np.column_stack((
        data['Avg_Sentiment'],
        data['Avg_Subjectivity']
    ))
    sentiment_similarity = cosine_similarity(sentiment_features)
    
    # Combine similarities with weights
    combined_similarity = (
        feature_weight * feature_similarity +
        sentiment_weight * sentiment_similarity
    )
    
    return combined_similarity

def get_recommendations(product_idx, similarity_matrix, data, n_recommendations=5):
    """
    Get product recommendations based on enhanced similarity
    """
    print(similarity_matrix.shape)  # Should be (num_products, num_products)
    print(similarity_matrix[product_idx])  # Should be a list/array of similarity scores

    sim_scores = list(enumerate(similarity_matrix[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
    product_indices = [i[0] for i in sim_scores]
    print(data.iloc[product_indices])
    recommendations = data.iloc[product_indices][['Brand', 'Label', 'Price', 'Category', 'Skin_Type', 'Benefits', 'Avg_Sentiment','sentiment_label']]
    recommendations['Similarity Score'] = [i[1] for i in sim_scores]
    return recommendations

def main():
    st.set_page_config(page_title="Skincare Product Recommender", layout="wide")
    
    st.title("🧴 Skincare Product Recommendation System")
    st.markdown("""
    This enhanced system recommends products based on both product features and user sentiment analysis.
    Select a product you like, and we'll find similar products with positive user experiences!
    """)
    
    # Load data
    try:
        data = load_data()
        similarity_matrix = create_similarity_matrix(data, feature_weight = 0.7, sentiment_weight = 0.3)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar filters
    st.sidebar.header("Filter Products")
    
    # Brand filter
    brands = ['All'] + sorted(data['Brand'].unique().tolist())
    selected_brand = st.sidebar.selectbox("Select Brand", brands)
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(data['Price'].min()),
        max_value=float(data['Price'].max()),
        value=(0.0, float(data['Price'].max()))
    )
    
    # Category filter
    categories = ['All'] + sorted(data['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Apply filters
    filtered_data = data.copy()
    if selected_brand != 'All':
        filtered_data = filtered_data[filtered_data['Brand'] == selected_brand]
    filtered_data = filtered_data[
        (filtered_data['Price'] >= price_range[0]) &
        (filtered_data['Price'] <= price_range[1])
    ]
    if selected_category != 'All':
        filtered_data = filtered_data[filtered_data['Category'] == selected_category]
    
    # Product selection
    st.subheader("Select a Product")
    if len(filtered_data) == 0:
        st.warning("No products found with the selected filters.")
        return
    
    selected_product = st.selectbox(
        "Choose a product you like:",
        filtered_data['Label'].tolist(),
        index=0
    )
    
    # Display selected product details
    product_idx = data[data['Label'] == selected_product].index[0]
    print(f"Type of product_idx: {type(product_idx)}")
    print(f"Value of product_idx: {product_idx}")
    selected_product_details = data.iloc[product_idx]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Product Details")
        st.write(f"**Brand:** {selected_product_details['Brand']}")
        st.write(f"**Price:** ${selected_product_details['Price']:.2f}")
        st.write(f"**Category:** {selected_product_details['Category']}")
        st.write(f"**Average Sentiment:** {selected_product_details['Avg_Sentiment']:.2f}")
        
        # Display reviews
        with st.expander("View Reviews"):
            for idx, review_analysis in enumerate(selected_product_details['Review_Analysis'], 1):
                st.write(f"Review {idx}:")
                st.write(f"- Sentiment: {review_analysis['sentiment_score']:.2f}")
                st.write(f"- Subjectivity: {review_analysis['subjectivity']:.2f}")
                if review_analysis['key_aspects']:
                    st.write("- Key aspects:", 
                            ", ".join([f"{aspect[0]} ({aspect[1]})" for aspect in review_analysis['key_aspects']]))
                st.markdown("---")
    
    # Get and display recommendations
    if st.button("Get Recommendations"):
        st.subheader("Recommended Products")
        recommendations = get_recommendations(product_idx, similarity_matrix, data)
        
        for i, row in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{row['Label']}** by {row['Brand']}")
                    st.write(f"Category: {row['Category']}")
                    st.write(f"Skin Type: {row['Skin_Type']}")
                    st.write(f"Benefits: {row['Benefits']}")
                with col2:
                    st.write(f"Price: ${row['Price']:.2f}")
                    st.write(f"Similarity Score: {row['Similarity Score']:.2%}")
                    st.write(f"Sentiment Score: {row['Avg_Sentiment']:.2f}")
                    st.write(f"Sentiment: {row['sentiment_label']}")

                st.markdown("---")

if __name__ == "__main__":
    main()