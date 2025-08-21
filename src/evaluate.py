"""
Model evaluation and comparison module.
Compares baseline vs BERT performance and generates comparison reports.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class ModelEvaluator:
    def __init__(self, results_dir="artifacts/results"):
        self.results_dir = Path(results_dir)
        self.label_names = ['BT1', 'BT2', 'BT3', 'BT4', 'BT5', 'BT6']
    
    def load_results(self):
        """Load results from both models."""
        try:
            # Load baseline results
            with open(self.results_dir / "baseline_metrics.pkl", 'rb') as f:
                baseline_results = pickle.load(f)
            
            # Load BERT results
            with open(self.results_dir / "bert_metrics.pkl", 'rb') as f:
                bert_results = pickle.load(f)
            
            return baseline_results, bert_results
            
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print("Make sure both baseline.py and train_bert.py have been run.")
            return None, None
    
    def create_comparison_table(self, baseline_results, bert_results):
        """Create a comparison table of model performance."""
        comparison_data = {
            'Metric': [
                'Accuracy', 
                'Precision (Macro)', 
                'Recall (Macro)', 
                'F1-Score (Macro)'
            ],
            'Baseline (TF-IDF + LogReg)': [
                f"{baseline_results['accuracy']:.4f}",
                f"{baseline_results['precision_macro']:.4f}",
                f"{baseline_results['recall_macro']:.4f}",
                f"{baseline_results['f1_macro']:.4f}"
            ],
            'BERT': [
                f"{bert_results['accuracy']:.4f}",
                f"{bert_results['precision_macro']:.4f}",
                f"{bert_results['recall_macro']:.4f}",
                f"{bert_results['f1_macro']:.4f}"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def create_per_class_comparison(self, baseline_results, bert_results):
        """Create per-class F1 score comparison."""
        per_class_data = {
            'Class': self.label_names,
            'Baseline F1': baseline_results['per_class_f1'],
            'BERT F1': bert_results['per_class_f1']
        }
        
        df = pd.DataFrame(per_class_data)
        df['Improvement'] = df['BERT F1'] - df['Baseline F1']
        return df
    
    def plot_model_comparison(self, baseline_results, bert_results, output_path):
        """Create a comprehensive comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        baseline_scores = [
            baseline_results['accuracy'],
            baseline_results['precision_macro'],
            baseline_results['recall_macro'],
            baseline_results['f1_macro']
        ]
        bert_scores = [
            bert_results['accuracy'],
            bert_results['precision_macro'],
            bert_results['recall_macro'],
            bert_results['f1_macro']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x + width/2, bert_scores, width, label='BERT', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Model Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # Per-class F1 comparison
        x_classes = np.arange(len(self.label_names))
        axes[0, 1].bar(x_classes - width/2, baseline_results['per_class_f1'], 
                      width, label='Baseline', alpha=0.8)
        axes[0, 1].bar(x_classes + width/2, bert_results['per_class_f1'], 
                      width, label='BERT', alpha=0.8)
        axes[0, 1].set_xlabel('Bloom\'s Taxonomy Level')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('Per-Class F1-Score Comparison')
        axes[0, 1].set_xticks(x_classes)
        axes[0, 1].set_xticklabels(self.label_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # Baseline confusion matrix
        sns.heatmap(
            baseline_results['confusion_matrix'], 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Baseline - Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted Label')
        axes[1, 0].set_ylabel('True Label')
        
        # BERT confusion matrix
        sns.heatmap(
            bert_results['confusion_matrix'], 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('BERT - Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {output_path}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("Loading model results...")
        baseline_results, bert_results = self.load_results()
        
        if baseline_results is None or bert_results is None:
            return
        
        print("\n" + "="*70)
        print("BLOOM'S TAXONOMY CLASSIFICATION - MODEL COMPARISON REPORT")
        print("="*70)
        
        # Overall comparison table
        comparison_df = self.create_comparison_table(baseline_results, bert_results)
        print("\n1. OVERALL PERFORMANCE COMPARISON")
        print("-" * 40)
        print(comparison_df.to_string(index=False))
        
        # Per-class comparison
        per_class_df = self.create_per_class_comparison(baseline_results, bert_results)
        print("\n2. PER-CLASS F1-SCORE COMPARISON")
        print("-" * 40)
        print(per_class_df.to_string(index=False))
        
        # Best performing classes
        print("\n3. ANALYSIS")
        print("-" * 40)
        best_baseline_class = per_class_df.loc[per_class_df['Baseline F1'].idxmax(), 'Class']
        best_bert_class = per_class_df.loc[per_class_df['BERT F1'].idxmax(), 'Class']
        most_improved = per_class_df.loc[per_class_df['Improvement'].idxmax(), 'Class']
        
        print(f"Best Baseline Performance: {best_baseline_class} "
              f"(F1: {per_class_df.loc[per_class_df['Class'] == best_baseline_class, 'Baseline F1'].iloc[0]:.4f})")
        print(f"Best BERT Performance: {best_bert_class} "
              f"(F1: {per_class_df.loc[per_class_df['Class'] == best_bert_class, 'BERT F1'].iloc[0]:.4f})")
        print(f"Most Improved Class: {most_improved} "
              f"(+{per_class_df.loc[per_class_df['Class'] == most_improved, 'Improvement'].iloc[0]:.4f})")
        
        # Overall improvement
        f1_improvement = bert_results['f1_macro'] - baseline_results['f1_macro']
        accuracy_improvement = bert_results['accuracy'] - baseline_results['accuracy']
        
        print(f"\nOverall F1 Improvement: {f1_improvement:+.4f}")
        print(f"Overall Accuracy Improvement: {accuracy_improvement:+.4f}")
        
        # Generate plots
        self.plot_model_comparison(
            baseline_results, 
            bert_results, 
            self.results_dir / "model_comparison.png"
        )
        
        # Save comparison tables
        comparison_df.to_csv(self.results_dir / "model_comparison.csv", index=False)
        per_class_df.to_csv(self.results_dir / "per_class_comparison.csv", index=False)
        
        print(f"\nDetailed results saved to {self.results_dir}")
        print("="*70)


def main():
    """Main evaluation pipeline."""
    evaluator = ModelEvaluator()
    evaluator.generate_report()


if __name__ == "__main__":
    main()