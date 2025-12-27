"""
Real-Time Sentiment Analysis System
Live Reddit Data Analyzer

This module fetches real Reddit posts about any company using Reddit's
free JSON API (no authentication required) and analyzes public sentiment.

How it works:
    Reddit allows adding .json to any URL to get data:
    - https://www.reddit.com/r/stocks/search.json?q=google
    - https://www.reddit.com/r/investing/hot.json

No API key or credentials needed.

    Developed by:
- Tanush Mehra | 102318065 | 3R12
- Saiyam Gupta | 102318060 | 3R11
- Tashu Sonker | 102303914 | 3C64
Date: December 2025
"""

import requests
import pandas as pd
import time
import json
import re
import os
from datetime import datetime
from collections import Counter

# Try to load ML model
try:
    import joblib
    from textblob import TextBlob
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
REQUEST_DELAY = 1.5  # seconds between requests


def fetch_reddit_posts(company_name, subreddit="stocks", limit=25):
    """
    Fetch Reddit posts using the free JSON API.
    
    Just append .json to any Reddit URL:
    - Search: reddit.com/r/stocks/search.json?q=google
    - Feed: reddit.com/r/stocks/hot.json
    """
    
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    
    params = {
        'q': company_name,
        'restrict_sr': 1,
        'limit': limit,
        'sort': 'relevance',
        't': 'month'
    }
    
    try:
        print(f"    Fetching from r/{subreddit}...", end=" ")
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            posts = []
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                posts.append({
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'comments': post_data.get('num_comments', 0),
                    'subreddit': post_data.get('subreddit', ''),
                    'author': post_data.get('author', ''),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}"
                })
            
            print(f"Found {len(posts)} posts")
            return posts
        
        elif response.status_code == 429:
            print("Rate limited, waiting...")
            time.sleep(5)
            return []
        else:
            print(f"Error: {response.status_code}")
            return []
    
    except Exception as e:
        print(f"Error: {str(e)[:30]}")
        return []


def fetch_company_data(company_name, subreddits=None):
    """Fetch data about a company from multiple subreddits."""
    
    if subreddits is None:
        subreddits = ['stocks', 'investing', 'technology']
    
    all_posts = []
    
    print(f"\nSearching for '{company_name}' on Reddit...")
    print("-" * 40)
    
    for subreddit in subreddits:
        posts = fetch_reddit_posts(company_name, subreddit, limit=25)
        all_posts.extend(posts)
        time.sleep(REQUEST_DELAY)
    
    print("-" * 40)
    print(f"Total posts collected: {len(all_posts)}")
    
    return all_posts


def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob."""
    
    if not text or len(text.strip()) == 0:
        return "Neutral", 0.0
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity


def analyze_sentiment_ml(text, model, vectorizer):
    """Analyze sentiment using trained ML model."""
    
    if not text or len(text.strip()) == 0:
        return "Neutral", 0.0
    
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    try:
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        
        try:
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
        except:
            confidence = 1.0
        
        return prediction, confidence
    except:
        return analyze_sentiment_textblob(text)


def analyze_posts(posts, use_ml=False):
    """Analyze sentiment of all posts."""
    
    model = None
    vectorizer = None
    
    if use_ml:
        try:
            model = joblib.load('models/sentiment_model.pkl')
            vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            print("Using ML model for analysis")
        except:
            print("Using TextBlob for analysis")
    
    results = []
    
    for post in posts:
        full_text = f"{post['title']} {post['text']}"
        
        if model and vectorizer:
            sentiment, score = analyze_sentiment_ml(full_text, model, vectorizer)
        else:
            sentiment, score = analyze_sentiment_textblob(full_text)
        
        results.append({
            'title': post['title'][:80],
            'subreddit': post['subreddit'],
            'score': post['score'],
            'comments': post['comments'],
            'sentiment': sentiment,
            'polarity': score
        })
    
    return pd.DataFrame(results)


def generate_report(df, company_name):
    """Generate sentiment report for a company."""
    
    print("\n" + "=" * 60)
    print(f"SENTIMENT REPORT: {company_name.upper()}")
    print("=" * 60)
    
    if df.empty:
        print("No data found for this company")
        return
    
    total = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    
    print(f"\nOverall Sentiment ({total} posts)")
    print("-" * 40)
    
    for sentiment, count in sentiment_counts.items():
        pct = (count / total) * 100
        bar = "#" * int(pct / 3)
        symbol = "+" if sentiment == "Positive" else ("-" if sentiment == "Negative" else "o")
        print(f"  [{symbol}] {sentiment:10}: {count:3} posts ({pct:5.1f}%) {bar}")
    
    # Net sentiment
    positive = sentiment_counts.get('Positive', 0) / total * 100
    negative = sentiment_counts.get('Negative', 0) / total * 100
    net = positive - negative
    
    print(f"\nNet Sentiment: {net:+.1f}%")
    
    if net > 15:
        print("Assessment: BULLISH - Strong positive sentiment")
    elif net > 5:
        print("Assessment: SLIGHTLY BULLISH")
    elif net > -5:
        print("Assessment: NEUTRAL - Mixed sentiment")
    elif net > -15:
        print("Assessment: SLIGHTLY BEARISH")
    else:
        print("Assessment: BEARISH - Strong negative sentiment")
    
    # By subreddit
    print(f"\nBy Subreddit")
    print("-" * 40)
    
    for subreddit in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == subreddit]
        sub_positive = (sub_df['sentiment'] == 'Positive').sum() / len(sub_df) * 100
        print(f"  r/{subreddit:15}: {len(sub_df):3} posts, {sub_positive:.0f}% positive")
    
    # Top posts
    print(f"\nTop Engaging Posts")
    print("-" * 40)
    
    df['engagement'] = df['score'] + df['comments'] * 2
    top = df.nlargest(5, 'engagement')
    
    for _, row in top.iterrows():
        symbol = "+" if row['sentiment'] == "Positive" else ("-" if row['sentiment'] == "Negative" else "o")
        print(f"  [{symbol}] ({row['score']:5} pts) r/{row['subreddit']}")
        print(f"      {row['title'][:55]}...")
    
    return {
        'total': total,
        'positive_pct': positive,
        'negative_pct': negative,
        'net_sentiment': net
    }


def main():
    print("\n" + "=" * 60)
    print("LIVE REDDIT SENTIMENT ANALYZER")
    print("=" * 60)
    
    print("""
    This tool fetches real Reddit data using the free JSON API.
    
    How it works:
    - Add .json to any Reddit URL to get JSON data
    - Example: reddit.com/r/stocks/search.json?q=google
    - No API key or login required
    
    Usage:
    - Modify 'company' variable below
    - Run this script
    - View sentiment analysis results
    """)
    
    # Configuration
    company = "Google"  # Change this to analyze different companies
    subreddits = ['stocks', 'investing', 'technology']
    
    print(f"\nAnalyzing: {company}")
    print("-" * 40)
    
    # Fetch data
    posts = fetch_company_data(company, subreddits)
    
    if posts:
        # Analyze
        df = analyze_posts(posts, use_ml=True)
        
        # Report
        report = generate_report(df, company)
        
        # Save
        os.makedirs('outputs', exist_ok=True)
        df.to_csv(f'outputs/{company.lower()}_sentiment.csv', index=False)
        print(f"\nSaved: outputs/{company.lower()}_sentiment.csv")
        
        with open(f'outputs/{company.lower()}_posts.json', 'w') as f:
            json.dump(posts, f, indent=2)
        print(f"Saved: outputs/{company.lower()}_posts.json")
    
    else:
        print("\nNo posts found. Try again later or check your connection.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nTo analyze other companies:")
    print("  1. Open this file")
    print("  2. Change: company = 'Tesla'")
    print("  3. Run again")


if __name__ == "__main__":
    main()
