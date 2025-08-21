"""
BERT-based model training for Bloom's Taxonomy classification.
Uses HuggingFace Transformers with early stopping on validation F1.
"""

import pandas as pd
import numpy as np
import torch
import os
import pickle
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class BERTModel:
    def __init__(self, model_name="bert-base-uncased", num_labels=6):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        self.label_names = ['BT1', 'BT2', 'BT3', 'BT4', 'BT5', 'BT6']
    
    def load_data(self, data_dir):
        """Load and tokenize data."""
        splits = {}
        
        for split in ['train', 'val', 'test']:
            df = pd.read_csv(f"{data_dir}/{split}.csv")
            
            # Create HuggingFace dataset
            dataset = Dataset.from_dict({
                'text': df['text'].tolist(),
                'labels': df['labels'].tolist()
            })
            
            # Tokenize
            dataset = dataset.map(
                lambda x: self.tokenizer(
                    x['text'], 
                    truncation=True, 
                    padding='max_length', 
                    max_length=512
                ),
                batched=True
            )
            
            splits[split] = dataset
        
        return splits
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir):
        """Train the BERT model."""
        print("Initializing BERT model...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        
        # Training arguments - compatible with multiple transformers versions
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",  # Updated parameter name
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb logging
            seed=42,
            warmup_steps=500,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Training completed!")
        return train_result
    
    def evaluate(self, test_dataset):
        """Evaluate the model on test set."""
        print("Evaluating on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'per_class_precision': per_class_metrics[0],
            'per_class_recall': per_class_metrics[1],
            'per_class_f1': per_class_metrics[2],
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('BERT Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_path}")
    
    def predict(self, texts):
        """Make predictions on new texts."""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        probabilities = predictions.cpu().numpy()
        
        return predicted_classes, probabilities


def main():
    """Main training pipeline for BERT model."""
    # Configuration
    data_dir = "artifacts/data"
    output_dir = "artifacts/models/bert"
    results_dir = "artifacts/results"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = BERTModel()
    
    # Load data
    print("Loading and tokenizing data...")
    datasets = model.load_data(data_dir)
    
    # Train model
    print("Training BERT model...")
    train_result = model.train(
        datasets['train'], 
        datasets['val'], 
        output_dir
    )
    
    # Evaluate on test set
    test_results = model.evaluate(datasets['test'])
    
    # Print results
    print("\n" + "="*50)
    print("BERT MODEL RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision (macro): {test_results['precision_macro']:.4f}")
    print(f"Test Recall (macro): {test_results['recall_macro']:.4f}")
    print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
    
    print("\nPer-class F1 scores:")
    for i, (label, f1) in enumerate(zip(model.label_names, test_results['per_class_f1'])):
        print(f"{label}: {f1:.4f}")
    
    # Plot confusion matrix
    model.plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions'],
        f"{results_dir}/bert_confusion_matrix.png"
    )
    
    # Save results
    with open(f"{results_dir}/bert_metrics.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    
    print(f"\nResults saved to {results_dir}")
    print("BERT model training completed!")


if __name__ == "__main__":
    main()