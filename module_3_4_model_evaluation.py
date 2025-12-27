"""
Real-Time Sentiment Analysis System
Module 3 & 4: Model Building and Evaluation

This module trains multiple ML models on preprocessed text data
and evaluates their performance using standard metrics.

Feature Extraction: TF-IDF with n-grams (1,2)
Models: Logistic Regression, SVM, Decision Tree, Random Forest, Naive Bayes

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
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    """Load preprocessed data from Module 1 & 2."""
    
    data_path = 'data/preprocessed_reddit_data.csv'
    
    if not os.path.exists(data_path):
        print("Error: Preprocessed data not found.")
        print("Please run module_1_2_eda_preprocessing.py first.")
        return None
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text_processed', 'sentiment'])
    
    print(f"Loaded {len(df)} samples")
    print(f"Sentiment distribution:")
    for sent, count in df['sentiment'].value_counts().items():
        print(f"    {sent}: {count}")
    
    return df


def prepare_features(df):
    """Create TF-IDF features for modeling."""
    
    print("\nCreating TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    X = tfidf.fit_transform(df['text_processed'])
    y = df['sentiment']
    
    print(f"Feature matrix: {X.shape}")
    print(f"Features extracted: {len(tfidf.get_feature_names_out())}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, tfidf


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple ML models and compare performance.
    Module 3: Feature Extraction and Model Building
    """
    
    print("\n" + "=" * 60)
    print("MODULE 3: MODEL TRAINING")
    print("=" * 60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining models...\n")
    
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        trained_models[name] = model
        
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    
    print("\n" + "-" * 60)
    print("MODEL COMPARISON:")
    print("-" * 60)
    print(results_df.to_string(index=False))
    
    # Save comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(results_df))
    width = 0.2
    
    ax.bar([i - 1.5*width for i in x], results_df['Accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar([i - 0.5*width for i in x], results_df['Precision'], width, label='Precision', color='#2ecc71')
    ax.bar([i + 0.5*width for i in x], results_df['Recall'], width, label='Recall', color='#e74c3c')
    ax.bar([i + 1.5*width for i in x], results_df['F1-Score'], width, label='F1-Score', color='#9b59b6')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150)
    plt.close()
    print("\nSaved: outputs/model_comparison.png")
    
    return results_df, trained_models


def evaluate_best_model(trained_models, results_df, X_train, X_test, y_train, y_test):
    """
    Evaluate the best performing model in detail.
    Module 4: Model Evaluation and Performance Analysis
    """
    
    print("\n" + "=" * 60)
    print("MODULE 4: MODEL EVALUATION")
    print("=" * 60)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_f1 = results_df.iloc[0]['F1-Score']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {best_f1:.4f}")
    
    # Classification Report
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: outputs/confusion_matrix.png")
    
    # Training vs Test Performance
    print("\n" + "-" * 60)
    print("TRAINING vs TEST PERFORMANCE")
    print("-" * 60)
    
    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Training F1:       {train_f1:.4f}")
    print(f"  Test F1:           {test_f1:.4f}")
    
    if train_acc - test_acc > 0.1:
        print("\n  Warning: Model may be overfitting")
    elif test_acc > train_acc:
        print("\n  Note: Test accuracy higher than training (unusual)")
    else:
        print("\n  Model appears well-balanced")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Accuracy', 'F1-Score']
    train_scores = [train_acc, train_f1]
    test_scores = [test_acc, test_f1]
    
    x = range(len(metrics))
    ax.bar([i - 0.15 for i in x], train_scores, 0.3, label='Training', color='#3498db')
    ax.bar([i + 0.15 for i in x], test_scores, 0.3, label='Testing', color='#e74c3c')
    ax.set_ylabel('Score')
    ax.set_title('Training vs Test Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('outputs/training_vs_test.png', dpi=150)
    plt.close()
    print("Saved: outputs/training_vs_test.png")
    
    return best_model, best_model_name


def test_on_unseen_data(model, vectorizer):
    """Test the model on new, unseen examples."""
    
    print("\n" + "-" * 60)
    print("TESTING ON UNSEEN DATA")
    print("-" * 60)
    
    test_sentences = [
        ("This is the best thing I have ever seen", "Positive"),
        ("Absolutely terrible waste of time", "Negative"),
        ("It is okay nothing special", "Neutral"),
        ("I love this so much amazing work", "Positive"),
        ("Very disappointing experience overall", "Negative"),
        ("Just an average product works fine", "Neutral"),
        ("Brilliant exceeded all my expectations", "Positive"),
        ("Horrible quality do not recommend", "Negative"),
    ]
    
    correct = 0
    print("\nPredictions:")
    
    for text, expected in test_sentences:
        features = vectorizer.transform([text.lower()])
        prediction = model.predict(features)[0]
        
        match = "OK" if prediction == expected else "X"
        symbol = "+" if prediction == "Positive" else ("-" if prediction == "Negative" else "o")
        
        print(f"  [{symbol}] {prediction:10} \"{text[:45]}...\" [{match}]")
        
        if prediction == expected:
            correct += 1
    
    print(f"\nAccuracy on unseen data: {correct}/{len(test_sentences)} ({correct/len(test_sentences)*100:.0f}%)")


def save_model(model, vectorizer, model_name):
    """Save the trained model and vectorizer."""
    
    print("\n" + "-" * 60)
    print("SAVING MODEL")
    print("-" * 60)
    
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print(f"  Saved: models/sentiment_model.pkl ({model_name})")
    print("  Saved: models/tfidf_vectorizer.pkl")


def main():
    print("\n" + "=" * 60)
    print("REAL-TIME SENTIMENT ANALYSIS SYSTEM")
    print("Modules 3 & 4: Model Building and Evaluation")
    print("=" * 60)
    
    df = load_data()
    if df is None:
        return
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df)
    
    results_df, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    best_model, best_model_name = evaluate_best_model(
        trained_models, results_df, X_train, X_test, y_train, y_test
    )
    
    test_on_unseen_data(best_model, vectorizer)
    
    save_model(best_model, vectorizer, best_model_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("MODULES 3 & 4 COMPLETE")
    print("=" * 60)
    print("\nModule 3 (Model Building):")
    print("  - TF-IDF features extracted")
    print("  - 5 ML models trained and compared")
    print(f"  - Best model: {best_model_name}")
    print("\nModule 4 (Evaluation):")
    print("  - Classification report generated")
    print("  - Confusion matrix created")
    print("  - Training vs test comparison done")
    print("  - Tested on unseen data")
    print("\nOutputs:")
    print("  - outputs/model_comparison.png")
    print("  - outputs/confusion_matrix.png")
    print("  - outputs/training_vs_test.png")
    print("  - models/sentiment_model.pkl")
    print("\nNext: Run module_5_dashboard.py")


if __name__ == "__main__":
    main()
