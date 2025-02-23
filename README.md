![screenshot](https://github.com/user-attachments/assets/9646a105-c791-4727-b9ab-49322c471a57)

# 🧔 Skincare Product Recommendation System

## Overview
The **Skincare Product Recommendation System** is a machine learning-based application that suggests skincare products based on user preferences, product features, and sentiment analysis of customer reviews. It uses **TF-IDF, cosine similarity**, and **TextBlob sentiment analysis** to provide personalized recommendations.

## Features
✅ **Content-Based Filtering** using **TF-IDF** and **cosine similarity**  
✅ **Sentiment Analysis** of reviews using **TextBlob**  
✅ **Product Filters** (Brand, Price Range, Category)  
✅ **User-Friendly UI** with **Streamlit**  
✅ **Dynamic Recommendations** based on product selection  

## Dataset
The dataset contains skincare product details, including:
- **Brand, Label, Category, Price**
- **Ingredients, Skin Type, Benefits**
- **Customer Reviews & Sentiment Analysis**

## Installation
### 🔹 Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### 🔹 Clone the Repository
```bash
git clone https://github.com/SiddiqueSahb/SkinCare_Product_Recommendation_System.git
cd SkinCare_Product_Recommendation_System
```

### 🔹 Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Run the Streamlit App
```bash
streamlit run app.py
```

## Project Structure
```
📚 SkinCare_Product_Recommendation_System
├── 📚 data
│   ├── skincare_products_reduced.csv
├── 📚 src
│   ├── data_preprocessing.py
│   ├── recommendation.py
├── 📚 utils
│   ├── file_io.py
│   ├── logging_utils.py
├── app.py                              # Streamlit UI
├── data_preprocessing.py               # Data processing & sentiment analysis
├── requirements.txt                    # Dependencies
├── README.md                           # Project Documentation
```

## How It Works
1. The user selects a skincare product.
2. The system extracts **text-based features** and **sentiment scores**.
3. **TF-IDF + Cosine Similarity** is used to find similar products.
4. Recommendations are **filtered** to show only products with **positive or neutral sentiment**.
5. Results are displayed with **product details, sentiment analysis, and similarity scores**.

## Technologies Used
- **Python** 🐍
- **Pandas, NumPy** for data manipulation
- **TextBlob** for sentiment analysis
- **Scikit-learn** for TF-IDF and cosine similarity
- **Streamlit** for the web interface

## To-Do / Future Improvements
- 🔹 Add more **advanced NLP models** (e.g., BERT for sentiment analysis)
- 🔹 Implement **hybrid filtering** (content + collaborative)
- 🔹 Improve UI with more **interactive visualizations**

---
🌟 **Enjoy your skincare recommendations!** 🧔


