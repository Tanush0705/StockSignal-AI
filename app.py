"""
Reddit Stock Sentiment Analyzer - Flask Web Application
Real-time sentiment analysis for smart investment decisions

Developed by:
- Tanush Mehra | 102318065 | 3R12
- Saiyam Gupta | 102318060 | 3R11
- Tashu Sonker | 102303914 | 3C64
"""

from flask import Flask, render_template, request, jsonify
import requests
import random
import time
from textblob import TextBlob
from datetime import datetime, timedelta

app = Flask(__name__)

# Sample data for when Reddit API is blocked
SAMPLE_POSTS = {
    'tesla': [
        {"title": "Tesla's new Model Y refresh looks incredible", "score": 2847, "comments": 423},
        {"title": "Just got my Tesla - best decision ever!", "score": 1923, "comments": 287},
        {"title": "Tesla FSD beta is getting really good", "score": 3201, "comments": 512},
        {"title": "Cybertruck deliveries ramping up nicely", "score": 1567, "comments": 234},
        {"title": "Tesla earnings beat expectations again", "score": 4521, "comments": 678},
        {"title": "Supercharger network expansion is impressive", "score": 892, "comments": 145},
        {"title": "Model 3 Highland - detailed review after 1 month", "score": 2134, "comments": 321},
        {"title": "Tesla stock analysis - bullish outlook for 2025", "score": 1876, "comments": 298},
    ],
    'apple': [
        {"title": "iPhone 16 Pro camera is absolutely stunning", "score": 5621, "comments": 734},
        {"title": "Apple Vision Pro - worth every penny", "score": 2341, "comments": 456},
        {"title": "M3 MacBook Pro performance is insane", "score": 3892, "comments": 521},
        {"title": "Apple's services revenue keeps growing", "score": 1234, "comments": 189},
        {"title": "AirPods Pro 2 with USB-C are perfect", "score": 2156, "comments": 312},
        {"title": "Apple Watch saved my life - heart monitoring", "score": 8934, "comments": 1023},
        {"title": "WWDC 2025 predictions - AI everywhere", "score": 1567, "comments": 234},
        {"title": "Apple stock hitting new all-time highs", "score": 3421, "comments": 478},
    ],
    'google': [
        {"title": "Gemini AI is actually impressive now", "score": 3421, "comments": 567},
        {"title": "Pixel 9 Pro camera beats iPhone in low light", "score": 2876, "comments": 423},
        {"title": "Google Cloud revenue growth accelerating", "score": 1543, "comments": 234},
        {"title": "YouTube Premium is worth it for the family plan", "score": 1987, "comments": 298},
        {"title": "Google Maps new AI features are game-changing", "score": 2341, "comments": 345},
        {"title": "Waymo robotaxi expansion going well", "score": 1654, "comments": 267},
        {"title": "Android 15 features I'm excited about", "score": 2789, "comments": 401},
        {"title": "Google Workspace productivity improvements", "score": 1123, "comments": 178},
    ],
    'microsoft': [
        {"title": "Copilot in Windows 11 is surprisingly useful", "score": 2341, "comments": 345},
        {"title": "Azure growth continues to dominate cloud", "score": 1876, "comments": 267},
        {"title": "Xbox Game Pass value is unbeatable", "score": 4532, "comments": 623},
        {"title": "Microsoft 365 Copilot review - productivity boost", "score": 1987, "comments": 298},
        {"title": "Surface Laptop Studio 2 is a beast", "score": 1543, "comments": 212},
        {"title": "GitHub Copilot makes coding 10x faster", "score": 3876, "comments": 534},
        {"title": "Teams getting so much better lately", "score": 1234, "comments": 189},
        {"title": "Microsoft stock dividend increase announced", "score": 2654, "comments": 378},
    ],
    'amazon': [
        {"title": "AWS re:Invent announcements are massive", "score": 2987, "comments": 423},
        {"title": "Prime delivery getting even faster somehow", "score": 1876, "comments": 267},
        {"title": "Alexa with AI is actually useful now", "score": 1543, "comments": 234},
        {"title": "Amazon stock split speculation", "score": 3421, "comments": 512},
        {"title": "AWS Bedrock making AI accessible", "score": 1654, "comments": 245},
        {"title": "Ring doorbell camera quality improved", "score": 987, "comments": 156},
        {"title": "Amazon Fresh expansion nationwide", "score": 1234, "comments": 189},
        {"title": "Kindle Scribe review - paper replacement", "score": 2156, "comments": 312},
    ],
    'nvidia': [
        {"title": "RTX 5090 benchmarks are INSANE", "score": 8934, "comments": 1234},
        {"title": "NVIDIA stock to $1000? Analysts say yes", "score": 5621, "comments": 823},
        {"title": "AI chip demand shows no signs of slowing", "score": 4532, "comments": 678},
        {"title": "CUDA dominance continues in AI/ML", "score": 2341, "comments": 345},
        {"title": "Jensen Huang keynote highlights", "score": 3876, "comments": 534},
        {"title": "NVIDIA earnings crush expectations again", "score": 6789, "comments": 923},
        {"title": "GeForce NOW cloud gaming improvements", "score": 1543, "comments": 223},
        {"title": "NVIDIA AI enterprise solutions growing fast", "score": 2654, "comments": 378},
    ],
    'meta': [
        {"title": "Quest 3 is the VR headset to buy", "score": 3421, "comments": 478},
        {"title": "Instagram Reels algorithm finally good", "score": 1987, "comments": 298},
        {"title": "Meta AI assistant is actually helpful", "score": 1654, "comments": 234},
        {"title": "WhatsApp channels feature is useful", "score": 1234, "comments": 189},
        {"title": "Reality Labs losses decreasing", "score": 2156, "comments": 312},
        {"title": "Threads growing faster than expected", "score": 2876, "comments": 401},
        {"title": "Meta stock recovery impressive", "score": 3654, "comments": 512},
        {"title": "Llama 3 open source AI model released", "score": 4532, "comments": 634},
    ]
}

def generate_dynamic_posts(company):
    """Generate realistic posts for companies not in sample data"""
    templates = [
        f"{company} stock analysis - looking bullish for Q1",
        f"Just invested in {company} - here's my thesis",
        f"{company} earnings preview - what to expect",
        f"Why I'm long on {company} for 2025",
        f"{company} vs competitors - detailed comparison",
        f"Breaking: {company} announces new product line",
        f"{company} CEO interview highlights",
        f"Technical analysis: {company} chart patterns",
        f"{company} dividend increase speculation",
        f"Insider buying at {company} - bullish signal?",
    ]
    
    posts = []
    for template in random.sample(templates, min(8, len(templates))):
        posts.append({
            "title": template,
            "score": random.randint(500, 5000),
            "comments": random.randint(50, 500)
        })
    return posts

def fetch_reddit_posts(company, limit=25):
    """Fetch posts from Reddit or use sample data"""
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']
    all_posts = []
    
    # Try Reddit API first
    for subreddit in subreddits[:2]:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': company,
                'sort': 'relevance',
                'limit': limit,
                't': 'week'
            }
            headers = {'User-Agent': 'SentimentAnalyzer/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                for post in posts:
                    post_data = post.get('data', {})
                    permalink = post_data.get('permalink', '')
                    reddit_url = f"https://www.reddit.com{permalink}" if permalink else ''
                    all_posts.append({
                        'title': post_data.get('title', ''),
                        'score': post_data.get('score', 0),
                        'comments': post_data.get('num_comments', 0),
                        'subreddit': post_data.get('subreddit', ''),
                        'created': post_data.get('created_utc', 0),
                        'url': reddit_url
                    })
        except:
            continue
    
    # Use sample data if Reddit fails
    if not all_posts:
        company_lower = company.lower()
        if company_lower in SAMPLE_POSTS:
            return SAMPLE_POSTS[company_lower], True
        else:
            return generate_dynamic_posts(company), True
    
    return all_posts, False

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def get_recommendation(avg_sentiment, positive_pct):
    """Generate investment recommendation"""
    if avg_sentiment > 0.15 and positive_pct > 60:
        return {
            'action': 'STRONG BUY',
            'color': '#00ff88',
            'description': 'Very positive sentiment detected. Market is bullish.',
            'confidence': min(95, int(50 + positive_pct * 0.5))
        }
    elif avg_sentiment > 0.05 and positive_pct > 45:
        return {
            'action': 'BUY',
            'color': '#00d4ff',
            'description': 'Positive sentiment detected. Consider investing.',
            'confidence': min(85, int(40 + positive_pct * 0.5))
        }
    elif avg_sentiment > -0.05:
        return {
            'action': 'HOLD',
            'color': '#ffd700',
            'description': 'Mixed sentiment. Monitor closely before acting.',
            'confidence': min(75, int(50 + abs(avg_sentiment) * 100))
        }
    elif avg_sentiment > -0.15:
        return {
            'action': 'SELL',
            'color': '#ff6b6b',
            'description': 'Negative sentiment detected. Consider reducing position.',
            'confidence': min(80, int(40 + (100 - positive_pct) * 0.4))
        }
    else:
        return {
            'action': 'STRONG SELL',
            'color': '#ff4444',
            'description': 'Very negative sentiment. High risk detected.',
            'confidence': min(90, int(50 + (100 - positive_pct) * 0.4))
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get('company', '').strip()
    
    if not company:
        return jsonify({'error': 'Please enter a company name'}), 400
    
    # Fetch posts
    posts, is_sample = fetch_reddit_posts(company)
    
    if not posts:
        return jsonify({'error': 'No posts found for this company'}), 404
    
    # Analyze sentiment
    sentiments = []
    analyzed_posts = []
    
    for post in posts[:15]:
        title = post.get('title', '')
        sentiment = analyze_sentiment(title)
        sentiments.append(sentiment)
        
        if sentiment > 0.05:
            label = 'Positive'
            color = '#00ff88'
        elif sentiment < -0.05:
            label = 'Negative'
            color = '#ff6b6b'
        else:
            label = 'Neutral'
            color = '#ffd700'
        
        analyzed_posts.append({
            'title': title[:100] + '...' if len(title) > 100 else title,
            'score': post.get('score', 0),
            'comments': post.get('comments', 0),
            'sentiment': round(sentiment, 3),
            'label': label,
            'color': color,
            'url': post.get('url', '')
        })
    
    # Calculate metrics
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    positive_count = sum(1 for s in sentiments if s > 0.05)
    negative_count = sum(1 for s in sentiments if s < -0.05)
    neutral_count = len(sentiments) - positive_count - negative_count
    
    positive_pct = (positive_count / len(sentiments) * 100) if sentiments else 0
    negative_pct = (negative_count / len(sentiments) * 100) if sentiments else 0
    neutral_pct = (neutral_count / len(sentiments) * 100) if sentiments else 0
    
    # Get recommendation
    recommendation = get_recommendation(avg_sentiment, positive_pct)
    
    return jsonify({
        'company': company.upper(),
        'posts_analyzed': len(analyzed_posts),
        'avg_sentiment': round(avg_sentiment, 3),
        'positive_pct': round(positive_pct, 1),
        'negative_pct': round(negative_pct, 1),
        'neutral_pct': round(neutral_pct, 1),
        'recommendation': recommendation,
        'posts': analyzed_posts,
        'is_sample': is_sample,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
