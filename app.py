import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from utils.file_io import load_pickle, load_numpy_array
from utils.logging_utils import log_info, log_error
from src.data_preprocessing import load_and_preprocess_data
from src.recommendation import create_similarity_matrix , get_recommendations

# Configure Streamlit App
st.set_page_config(page_title="Skincare Product Recommender", layout="wide")

st.title("🧴 Skincare Product Recommendation System")
st.markdown("""
This system recommends skincare products based on product features and user sentiment analysis.
Select a product, and we'll suggest similar ones with positive reviews!  
""")

# Load Data
try:
    log_info("Loading preprocessed data and models...")
    data = load_pickle("data/processed_data.pkl")
    print("Columns in recommendations DataFrame:", data.columns)  
    tfidf_vectorizer = load_pickle("models/tfidf_vectorizer.pkl")
    #sentiment_model = load_pickle("models/sentiment_model.pkl")
    similarity_matrix = load_numpy_array("models/similarity_matrix.npy")
    log_info("Data and models loaded successfully.")
except Exception as e:
    st.error(f"Error loading data/models: {str(e)}")
    log_error(f"Error loading data/models: {str(e)}")
    st.stop()


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
    (filtered_data['Price'] >= price_range[0]) & (filtered_data['Price'] <= price_range[1])
]
if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['Category'] == selected_category]

# Product selection
st.subheader("Select a Product")
if len(filtered_data) == 0:
    st.warning("No products found with the selected filters.")
    st.stop()

selected_product = st.selectbox("Choose a product you like:", filtered_data['Label'].tolist(), index=0)

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
            st.write(f"- Sentiment Score: {review_analysis['sentiment_score']:.2f}")
            st.write(f"- Subjectivity: {review_analysis['subjectivity']:.2f}")
            st.write(f"- Sentiment Type: {review_analysis['sentiment_label']}")
            if review_analysis['key_aspects']:
                st.write("- Key Aspects:", 
                         ", ".join([f"{aspect[0]} ({aspect[1]})" for aspect in review_analysis['key_aspects']]))
            st.markdown("---")

# Get and display recommendations
if st.button("Get Recommendations"):
    st.subheader("Recommended Products")

    recommendations = get_recommendations(product_idx, n_recommendations=5)

    # Filter recommendations for only Positive and Neutral sentiment
    filtered_recommendations = recommendations[
        recommendations["sentiment_label"].isin(["Positive", "Neutral"])
    ]

    if filtered_recommendations.empty:
        st.write("No recommendations with Positive or Neutral sentiment.")
    else:
        for i, (_, row) in enumerate(filtered_recommendations.iterrows(), start=1):
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"### **{i}. {row['Label']}** by {row['Brand']}")
                    st.write(f"Category: {row['Category']}")
                    st.write(f"Skin Type: {row['Skin_Type']}")
                    st.write(f"Benefits: {row['Benefits']}")
                with col2:
                    st.write(f"Price: ${row['Price']:.2f}")
                    st.write(f"Similarity Score: {row['Similarity Score']:.2%}")
                    st.write(f"Sentiment Score: {row['Avg_Sentiment']:.2f}")
                    st.write(f"Sentiment: {row['sentiment_label']}")
            st.markdown("---")

