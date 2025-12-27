"""
Real-Time Sentiment Analysis System
Module 1 & 2: Data Exploration and Preprocessing

Project: Reddit Sentiment Analysis for Company Monitoring
Data Modality: Textual Data (Reddit Comments/Posts)
Dataset: Reddit Sentiment Dataset (Kaggle) or Sample Data

    Developed by:
- Tanush Mehra | 102318065 | 3R12
- Saiyam Gupta | 102318060 | 3R11
- Tashu Sonker | 102303914 | 3C64

Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib

# Setup directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_or_create_dataset():
    """
    Load the Reddit sentiment dataset from Kaggle or create sample data.
    Kaggle dataset: https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
    Contains 37K+ Reddit comments with sentiment labels (-1, 0, 1)
    """
    
    kaggle_path = 'data/Reddit_Data.csv'
    
    if os.path.exists(kaggle_path):
        print("Loading Reddit Sentiment Dataset from Kaggle...")
        df = pd.read_csv(kaggle_path)
        
        if 'clean_comment' in df.columns:
            df = df.rename(columns={'clean_comment': 'text', 'category': 'sentiment'})
        
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        if df['sentiment'].dtype in ['int64', 'float64']:
            df['sentiment'] = df['sentiment'].map(sentiment_map)
        
        print(f"Loaded {len(df)} samples")
        return df
    
    print("Creating sample dataset for demonstration...")
    print("Note: For full dataset, download from Kaggle link above")
    
    positive_samples = [
        "This product exceeded my expectations highly recommend it",
        "Amazing experience with customer service very helpful staff",
        "Best purchase I made this year totally worth the price",
        "Love the new features the update is fantastic",
        "Great quality and fast shipping will buy again",
        "Impressed by the attention to detail in this product",
        "Finally found something that actually works as advertised",
        "Customer support resolved my issue within minutes",
        "The new version is so much better than before",
        "Excellent value for money very satisfied customer",
        "This company really cares about their customers",
        "Smooth experience from start to finish",
        "Would definitely recommend to friends and family",
        "Quality is outstanding exceeded all expectations",
        "Best decision I made should have bought it sooner",
        "Absolutely love this cannot imagine life without it",
        "Perfect for what I needed works flawlessly",
        "Happy with my purchase exceeded expectations",
        "Top notch quality will be a returning customer",
        "Brilliant product that actually delivers on promises",
    ]
    
    negative_samples = [
        "Terrible quality broke after one week of use",
        "Complete waste of money avoid this product",
        "Customer service was unhelpful and rude",
        "Shipping took forever and item arrived damaged",
        "Does not work as advertised very disappointed",
        "Worst purchase I ever made requesting refund",
        "Poor build quality cheap materials used",
        "The product stopped working after two days",
        "Misleading description nothing like the pictures",
        "Save your money and buy something else",
        "Horrible experience with this company",
        "Product quality has declined significantly",
        "Not worth the price overpriced garbage",
        "Installation was a nightmare confusing instructions",
        "Regret buying this complete disappointment",
        "Avoid at all costs total scam",
        "Broke immediately upon first use",
        "Terrible support refused to help me",
        "Product arrived defective had to return it",
        "Never buying from this brand again",
    ]
    
    neutral_samples = [
        "It works okay nothing special about it",
        "Average product does what it says",
        "Received the item looks as expected",
        "Normal delivery time standard packaging",
        "Product is fine for the price point",
        "Not bad but not great either",
        "Gets the job done no complaints",
        "Exactly what I expected nothing more",
        "Standard quality meets basic needs",
        "It is what it is functional product",
        "Decent option if you need something basic",
        "Works as intended no issues so far",
        "Middle of the road product acceptable",
        "Neither impressed nor disappointed",
        "Fair enough for everyday use",
        "Basic functionality nothing fancy",
        "Does the job adequately enough",
        "Acceptable for the price range",
        "Standard product nothing to complain about",
        "Ordinary product with ordinary results",
    ]
    
    data = []
    suffixes = ["", " Really recommend.", " Honestly.", " My opinion.", " Worth noting."]
    
    for text in positive_samples:
        for suffix in suffixes[:3]:
            data.append({'text': text + suffix, 'sentiment': 'Positive'})
    
    for text in negative_samples:
        for suffix in suffixes[:3]:
            data.append({'text': text + suffix, 'sentiment': 'Negative'})
    
    for text in neutral_samples:
        for suffix in suffixes[:3]:
            data.append({'text': text + suffix, 'sentiment': 'Neutral'})
    
    df = pd.DataFrame(data)
    np.random.seed(42)
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Created sample dataset with {len(df)} samples")
    return df


def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the text dataset.
    Module 1 Requirements:
    - Word frequency extraction
    - Stopword identification  
    - Sample sentence analysis
    - Text length distribution
    """
    
    print("\n" + "=" * 60)
    print("MODULE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Dataset Overview
    print("\n[1] DATASET OVERVIEW")
    print("-" * 40)
    print(f"    Total samples: {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Missing values: {df.isnull().sum().sum()}")
    
    # Class Distribution
    print("\n[2] SENTIMENT DISTRIBUTION")
    print("-" * 40)
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / len(df) * 100
        print(f"    {sentiment}: {count} samples ({pct:.1f}%)")
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'Positive': '#27ae60', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values,
                  color=[colors.get(s, '#7f8c8d') for s in sentiment_counts.index])
    ax.set_title('Sentiment Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sentiment Category')
    ax.set_ylabel('Count')
    for bar, count in zip(bars, sentiment_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/sentiment_distribution.png', dpi=150)
    plt.close()
    print("    Saved: outputs/sentiment_distribution.png")
    
    # Text Length Analysis
    print("\n[3] TEXT LENGTH ANALYSIS")
    print("-" * 40)
    df['char_count'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    print(f"    Average length: {df['char_count'].mean():.1f} characters")
    print(f"    Average words: {df['word_count'].mean():.1f} words")
    print(f"    Min length: {df['char_count'].min()} chars")
    print(f"    Max length: {df['char_count'].max()} chars")
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df['char_count'], bins=25, color='#3498db', edgecolor='white')
    axes[0].set_title('Character Count Distribution', fontweight='bold')
    axes[0].set_xlabel('Characters')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(df['word_count'], bins=15, color='#9b59b6', edgecolor='white')
    axes[1].set_title('Word Count Distribution', fontweight='bold')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('outputs/text_length_distribution.png', dpi=150)
    plt.close()
    print("    Saved: outputs/text_length_distribution.png")
    
    # Word Frequency
    print("\n[4] WORD FREQUENCY ANALYSIS")
    print("-" * 40)
    all_text = ' '.join(df['text'].astype(str).tolist()).lower()
    words = re.findall(r'\b[a-z]+\b', all_text)
    word_freq = Counter(words)
    top_words = word_freq.most_common(15)
    
    print("    Top 10 frequent words:")
    for word, count in top_words[:10]:
        print(f"        {word}: {count}")
    
    words_list, counts_list = zip(*top_words)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(words_list)), counts_list, color='#1abc9c')
    ax.set_yticks(range(len(words_list)))
    ax.set_yticklabels(words_list)
    ax.invert_yaxis()
    ax.set_title('Top 15 Frequent Words', fontweight='bold')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('outputs/word_frequency.png', dpi=150)
    plt.close()
    print("    Saved: outputs/word_frequency.png")
    
    # Stopword Analysis
    print("\n[5] STOPWORD ANALYSIS")
    print("-" * 40)
    stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'was', 'i', 'and', 'of',
                 'in', 'for', 'on', 'with', 'this', 'that', 'be', 'are', 'as',
                 'my', 'so', 'or', 'if', 'from', 'just', 'very', 'what', 'all'}
    
    stopword_count = sum(count for word, count in word_freq.items() if word in stopwords)
    total_words = len(words)
    pct = stopword_count / total_words * 100
    
    print(f"    Total words: {total_words}")
    print(f"    Stopwords: {stopword_count} ({pct:.1f}%)")
    
    # Sample sentences
    print("\n[6] SAMPLE SENTENCES")
    print("-" * 40)
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sample = df[df['sentiment'] == sentiment]['text'].iloc[0]
        print(f"    [{sentiment}] \"{sample[:60]}...\"")
    
    return df, word_freq


def preprocess_text(text):
    """Clean text: lowercase, remove punctuation and extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataset(df):
    """
    Preprocess the dataset for machine learning.
    Module 2 Requirements:
    - Removal of punctuation, stopwords
    - Lowercasing, cleaning
    - Tokenization
    - TF-IDF / Bag-of-Words conversion
    """
    
    print("\n" + "=" * 60)
    print("MODULE 2: TEXT PREPROCESSING")
    print("=" * 60)
    
    stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'was', 'i', 'and', 'of',
                 'in', 'for', 'on', 'with', 'this', 'that', 'be', 'are', 'as',
                 'at', 'have', 'has', 'had', 'but', 'not', 'you', 'they', 'we',
                 'my', 'so', 'or', 'if', 'from', 'just', 'very', 'what', 'all',
                 'can', 'will', 'been', 'would', 'could', 'do', 'does', 'did'}
    
    # Step 1: Clean text
    print("\n[1] CLEANING TEXT")
    print("-" * 40)
    print("    Applying lowercase and punctuation removal...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    print("    Done")
    
    # Step 2: Remove stopwords
    print("\n[2] REMOVING STOPWORDS")
    print("-" * 40)
    df['text_processed'] = df['text_clean'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stopwords])
    )
    
    idx = df.index[0]
    print(f"    Before: \"{df.loc[idx, 'text'][:50]}...\"")
    print(f"    After:  \"{df.loc[idx, 'text_processed'][:50]}...\"")
    
    # Step 3: TF-IDF Vectorization
    print("\n[3] TF-IDF VECTORIZATION")
    print("-" * 40)
    
    tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['text_processed'])
    
    print(f"    Matrix shape: {tfidf_matrix.shape}")
    print(f"    Features: {len(tfidf.get_feature_names_out())}")
    
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("    Saved: models/tfidf_vectorizer.pkl")
    
    # Step 4: Bag-of-Words
    print("\n[4] BAG-OF-WORDS VECTORIZATION")
    print("-" * 40)
    
    bow = CountVectorizer(max_features=500, min_df=2, max_df=0.95)
    bow_matrix = bow.fit_transform(df['text_processed'])
    
    print(f"    Matrix shape: {bow_matrix.shape}")
    
    joblib.dump(bow, 'models/bow_vectorizer.pkl')
    print("    Saved: models/bow_vectorizer.pkl")
    
    # Save preprocessed data
    df.to_csv('data/preprocessed_reddit_data.csv', index=False)
    print("\n    Saved: data/preprocessed_reddit_data.csv")
    
    return df, tfidf, tfidf_matrix


def main():
    print("\n" + "=" * 60)
    print("REAL-TIME SENTIMENT ANALYSIS SYSTEM")
    print("Modules 1 & 2: Exploration and Preprocessing")
    print("=" * 60)
    
    df = load_or_create_dataset()
    df, word_freq = perform_eda(df)
    df, tfidf, tfidf_matrix = preprocess_dataset(df)
    
    print("\n" + "=" * 60)
    print("MODULES 1 & 2 COMPLETE")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  - outputs/sentiment_distribution.png")
    print("  - outputs/text_length_distribution.png")
    print("  - outputs/word_frequency.png")
    print("  - models/tfidf_vectorizer.pkl")
    print("  - models/bow_vectorizer.pkl")
    print("  - data/preprocessed_reddit_data.csv")
    print("\nNext: Run module_3_4_model_evaluation.py")


if __name__ == "__main__":
    main()
