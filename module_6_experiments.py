"""
Real-Time Sentiment Analysis System
Module 6: AI Exploration Experiments

This module tests the trained model with various experimental inputs
to understand its behavior, limitations, and edge cases.

Experiments:
1. Unexpected inputs the model was not trained on
2. Real-world personal inputs
3. Modification effects on predictions
4. Sarcasm detection challenge
5. Edge cases and stress testing

Developed by:
- Tanush Mehra | 102318065 | 3R12
- Saiyam Gupta | 102318060 | 3R11
- Tashu Sonker | 102303914 | 3C64

Date: December 2025
"""

import os
import joblib
import re

# Try to import TextBlob for comparison
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


def load_model():
    """Load the trained model and vectorizer."""
    
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("ML model loaded successfully")
        return model, vectorizer, True
    
    print("ML model not found. Using TextBlob for analysis.")
    return None, None, False


def preprocess_text(text):
    """Clean text for prediction."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text, model, vectorizer, use_ml=True):
    """Predict sentiment for given text."""
    
    clean_text = preprocess_text(text)
    
    if not clean_text:
        return "Neutral", 0.0
    
    if use_ml and model and vectorizer:
        features = vectorizer.transform([clean_text])
        sentiment = model.predict(features)[0]
        
        try:
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
        except:
            confidence = 1.0
        
        return sentiment, confidence
    
    elif TEXTBLOB_AVAILABLE:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    
    return "Neutral", 0.0


def experiment_1_unexpected_inputs(model, vectorizer, use_ml):
    """Test the model with inputs it was not trained on."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: UNEXPECTED INPUTS")
    print("=" * 60)
    print("\nTesting model with unfamiliar input types:\n")
    
    test_cases = [
        ("Technical jargon", "The API endpoint returned a 500 error"),
        ("Technical jargon", "Successfully deployed to production server"),
        ("Mixed language", "This is muy bueno I love it"),
        ("Mixed language", "C'est terrible vraiment bad"),
        ("Numbers only", "10 out of 10 would recommend"),
        ("Numbers only", "0 stars if I could absolutely 0"),
        ("Random text", "asdfghjkl qwerty zxcvbnm"),
        ("Random text", "xyz abc 123 456"),
        ("Scientific", "The mitochondria is the powerhouse of the cell"),
        ("Scientific", "Water boils at 100 degrees Celsius"),
        ("Minimal", "..."),
        ("Minimal", "hmm okay"),
    ]
    
    results = {}
    
    for category, text in test_cases:
        sentiment, _ = predict(text, model, vectorizer, use_ml)
        symbol = "+" if sentiment == "Positive" else ("-" if sentiment == "Negative" else "o")
        
        print(f"  [{symbol}] [{category:15}] {sentiment:10} \"{text[:40]}...\"")
        
        if category not in results:
            results[category] = []
        results[category].append(sentiment)
    
    print("\nObservations:")
    print("-" * 40)
    print("  - Technical text: Model relies on sentiment words if present")
    print("  - Random/gibberish: Defaults to neutral (no sentiment indicators)")
    print("  - Scientific facts: Usually classified as neutral")
    print("  - Mixed language: English sentiment words still recognized")


def experiment_2_realworld_inputs(model, vectorizer, use_ml):
    """Test with everyday real-world inputs."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: REAL-WORLD PERSONAL INPUTS")
    print("=" * 60)
    print("\nTesting with everyday sentences:\n")
    
    test_cases = [
        ("I just finished my assignment and I am so relieved", "Positive"),
        ("The cafeteria food today was actually pretty good", "Positive"),
        ("Cannot believe I failed that easy test", "Negative"),
        ("My friend helped me with my project so grateful", "Positive"),
        ("Traffic was terrible this morning wasted so much time", "Negative"),
        ("Just chilling at home nothing much happening", "Neutral"),
        ("Got a new phone today super excited", "Positive"),
        ("The lecture was so boring I almost fell asleep", "Negative"),
    ]
    
    correct = 0
    
    for text, expected in test_cases:
        sentiment, _ = predict(text, model, vectorizer, use_ml)
        
        match = sentiment == expected
        if match:
            correct += 1
            status = "OK"
        else:
            status = "X"
        
        symbol = "+" if sentiment == "Positive" else ("-" if sentiment == "Negative" else "o")
        print(f"  [{symbol}] {sentiment:10} \"{text[:45]}...\" [{status}]")
        
        if not match:
            print(f"       Expected: {expected}")
    
    accuracy = correct / len(test_cases) * 100
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({accuracy:.0f}%)")
    
    print("\nObservations:")
    print("-" * 40)
    print("  - Clear emotions (excited, grateful): Usually detected correctly")
    print("  - Casual language: May be harder to classify")
    print("  - Context-dependent: Performance varies with phrasing")


def experiment_3_modification_effects(model, vectorizer, use_ml):
    """Test how small modifications affect predictions."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: MODIFICATION EFFECTS")
    print("=" * 60)
    print("\nTesting how changes affect predictions:\n")
    
    base_sentences = [
        ("This is great", [
            "This is great!",
            "This is GREAT",
            "This is great...",
            "This is greaaaat",
            "This is not great",
        ]),
        ("I hate this product", [
            "I hate this product!",
            "I HATE this product",
            "I kinda hate this product",
            "I dont hate this product",
            "I haaate this product",
        ]),
        ("The movie was okay", [
            "The movie was okay I guess",
            "The movie was just okay",
            "The movie was more than okay",
            "The movie was OK",
            "The movie was meh",
        ]),
    ]
    
    for base, modifications in base_sentences:
        base_sentiment, _ = predict(base, model, vectorizer, use_ml)
        print(f"\nOriginal: \"{base}\" -> {base_sentiment}")
        print("-" * 40)
        
        for mod in modifications:
            mod_sentiment, _ = predict(mod, model, vectorizer, use_ml)
            changed = "CHANGED" if mod_sentiment != base_sentiment else ""
            print(f"  \"{mod[:35]}...\" -> {mod_sentiment} {changed}")
    
    print("\nObservations:")
    print("-" * 40)
    print("  - Capitalization: Usually no effect (text is lowercased)")
    print("  - Punctuation: Minimal effect (removed in preprocessing)")
    print("  - Negation: Hard for simple models to handle correctly")
    print("  - Stretched words: May not match vocabulary")


def experiment_4_sarcasm_challenge(model, vectorizer, use_ml):
    """Test sarcasm detection - a known challenge for sentiment models."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: SARCASM DETECTION")
    print("=" * 60)
    print("\nTesting sarcastic statements (positive words, negative meaning):\n")
    
    sarcastic_cases = [
        ("Oh great another meeting just what I needed", "Negative"),
        ("Wow thanks for nothing", "Negative"),
        ("Sure that is exactly what I wanted perfect", "Negative"),
        ("What a surprise it is broken again", "Negative"),
        ("I just love waiting in long lines", "Negative"),
        ("Best day ever NOT", "Negative"),
        ("Yeah right like that is going to work", "Negative"),
        ("Oh wonderful more homework", "Negative"),
    ]
    
    correct = 0
    
    for text, expected in sarcastic_cases:
        sentiment, _ = predict(text, model, vectorizer, use_ml)
        
        match = sentiment == expected
        if match:
            correct += 1
            status = "OK"
        else:
            status = "X"
        
        symbol = "+" if sentiment == "Positive" else ("-" if sentiment == "Negative" else "o")
        print(f"  [{symbol}] {sentiment:10} \"{text[:45]}...\" [{status}]")
    
    accuracy = correct / len(sarcastic_cases) * 100
    print(f"\nSarcasm Detection: {correct}/{len(sarcastic_cases)} ({accuracy:.0f}%)")
    
    print("\nWhy sarcasm is difficult:")
    print("-" * 40)
    print("  1. Words vs Intent: Positive words used with negative meaning")
    print("  2. Context Required: Needs understanding beyond word patterns")
    print("  3. Cultural Variation: Sarcasm patterns differ across cultures")
    print("  4. Possible Improvements: Transformer models (BERT, GPT) handle better")


def experiment_5_edge_cases(model, vectorizer, use_ml):
    """Stress test with edge cases."""
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: EDGE CASES")
    print("=" * 60)
    print("\nTesting unusual and edge case inputs:\n")
    
    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "     "),
        ("Single word", "good"),
        ("Single word", "bad"),
        ("Single word", "maybe"),
        ("Only emojis", "😊😊😊"),
        ("Only punctuation", "!!!!????"),
        ("Mixed sentiment", "good bad happy sad love hate"),
        ("Negation", "not good"),
        ("Double negation", "not bad at all"),
        ("Triple negation", "I am not unhappy"),
        ("Internet slang", "lol lmao rofl"),
        ("Very long text", "this is a " + "very " * 50 + "long sentence"),
    ]
    
    for label, text in edge_cases:
        if not text.strip():
            print(f"  [?] [{label:20}] (empty input)")
            continue
        
        sentiment, _ = predict(text, model, vectorizer, use_ml)
        symbol = "+" if sentiment == "Positive" else ("-" if sentiment == "Negative" else "o")
        
        display_text = text[:30] + "..." if len(text) > 30 else text
        print(f"  [{symbol}] [{label:20}] {sentiment:10} \"{display_text}\"")
    
    print("\nObservations:")
    print("-" * 40)
    print("  - Empty/whitespace: Should return neutral or handle gracefully")
    print("  - Single words: Classification based on that word alone")
    print("  - Mixed sentiment: Model averages or picks dominant")
    print("  - Negation: Remains a challenge for bag-of-words models")


def print_summary():
    """Print summary of all experiments."""
    
    print("\n" + "=" * 60)
    print("MODULE 6: EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("""
    EXPERIMENT 1: Unexpected Inputs
    - Model handles unfamiliar text by defaulting to neutral
    - Technical jargon classification depends on sentiment words
    - Gibberish and random text classified as neutral
    
    EXPERIMENT 2: Real-World Inputs
    - Clear sentiment expressions detected accurately
    - Casual and informal language may be misclassified
    - Performance varies with phrasing and context
    
    EXPERIMENT 3: Modification Effects
    - Preprocessing normalizes case and punctuation
    - Negation handling is a known weakness
    - Stretched words may not match vocabulary
    
    EXPERIMENT 4: Sarcasm Detection
    - Traditional ML models struggle with sarcasm
    - Word-level analysis misses intended meaning
    - Transformer models recommended for improvement
    
    EXPERIMENT 5: Edge Cases
    - Empty inputs handled gracefully
    - Single words classified by word sentiment
    - Long text and mixed sentiment are challenging
    
    KEY TAKEAWAYS:
    - No model is perfect; understanding limitations is important
    - Preprocessing choices significantly impact results
    - Real-world text is messy and diverse
    - Continuous testing improves model reliability
    """)


def main():
    print("\n" + "=" * 60)
    print("REAL-TIME SENTIMENT ANALYSIS SYSTEM")
    print("Module 6: AI Exploration Experiments")
    print("=" * 60)
    
    model, vectorizer, model_loaded = load_model()
    
    experiment_1_unexpected_inputs(model, vectorizer, model_loaded)
    experiment_2_realworld_inputs(model, vectorizer, model_loaded)
    experiment_3_modification_effects(model, vectorizer, model_loaded)
    experiment_4_sarcasm_challenge(model, vectorizer, model_loaded)
    experiment_5_edge_cases(model, vectorizer, model_loaded)
    
    print_summary()
    
    print("\n" + "=" * 60)
    print("MODULE 6 COMPLETE")
    print("=" * 60)
    print("\nAll experiments completed successfully.")
    print("\nProject Files:")
    print("  - module_1_2_eda_preprocessing.py")
    print("  - module_3_4_model_evaluation.py")
    print("  - module_5_dashboard.py")
    print("  - module_6_experiments.py")
    print("  - live_reddit_analyzer.py")


if __name__ == "__main__":
    main()
