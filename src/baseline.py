"""
Baseline model using TF-IDF + Logistic Regression for Bloom's Taxonomy classification.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import joblib


class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr'
        )
        self.label_names = ['BT1', 'BT2', 'BT3', 'BT4', 'BT5', 'BT6']
    
    def load_data(self, data_dir):
        """Load train/val/test splits."""
        splits = {}
        for split in ['train', 'val', 'test']:
            df = pd.read_csv(f"{data_dir}/{split}.csv")
            splits[split] = {
                'text': df['text'].values,
                'labels': df['labels'].values
            }
        return splits
    
    def train(self, train_data, val_data):
        """Train the baseline model."""
        print("Training TF-IDF vectorizer...")
        X_train = self.vectorizer.fit_transform(train_data['text'])
        y_train = train_data['labels']
        
        print("Training Logistic Regression...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        X_val = self.vectorizer.transform(val_data['text'])
        y_val = val_data['labels']
        
        val_predictions = self.classifier.predict(X_val)
        val_metrics = self.calculate_metrics(y_val, val_predictions)
        
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        return val_metrics
    
    def predict(self, texts):
        """Make predictions on new texts."""
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        return predictions, probabilities
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'per_class_precision': per_class_metrics[0],
            'per_class_recall': per_class_metrics[1],
            'per_class_f1': per_class_metrics[2],
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Baseline Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_path}")
    
    def save_model(self, output_dir):
        """Save the trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, f"{output_dir}/tfidf_vectorizer.pkl")
        
        # Save classifier
        joblib.dump(self.classifier, f"{output_dir}/logistic_regression.pkl")
        
        print(f"Baseline model saved to {output_dir}")
    
    def load_model(self, model_dir):
        """Load a trained model."""
        self.vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
        self.classifier = joblib.load(f"{model_dir}/logistic_regression.pkl")
        print("Baseline model loaded successfully")


def main():
    """Main training pipeline for baseline model."""
    # Configuration
    data_dir = "artifacts/data"
    output_dir = "artifacts/models/baseline"
    results_dir = "artifacts/results"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model
    model = BaselineModel()
    
    # Load data
    print("Loading data...")
    splits = model.load_data(data_dir)
    
    # Train model
    print("Training baseline model...")
    val_metrics = model.train(splits['train'], splits['val'])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions, test_probabilities = model.predict(splits['test']['text'])
    test_metrics = model.calculate_metrics(splits['test']['labels'], test_predictions)
    
    # Print results
    print("\n" + "="*50)
    print("BASELINE MODEL RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    print("\nPer-class F1 scores:")
    for i, (label, f1) in enumerate(zip(model.label_names, test_metrics['per_class_f1'])):
        print(f"{label}: {f1:.4f}")
    
    # Plot confusion matrix
    model.plot_confusion_matrix(
        splits['test']['labels'], 
        test_predictions,
        f"{results_dir}/baseline_confusion_matrix.png"
    )
    
    # Save model
    model.save_model(output_dir)
    
    # Save metrics
    with open(f"{results_dir}/baseline_metrics.pkl", 'wb') as f:
        pickle.dump(test_metrics, f)
    
    print(f"\nResults saved to {results_dir}")
    print("Baseline model training completed!")


if __name__ == "__main__":
    main()