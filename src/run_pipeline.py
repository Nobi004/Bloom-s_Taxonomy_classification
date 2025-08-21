#!/usr/bin/env python3
"""
Complete pipeline runner for Bloom's Taxonomy Classification project.
Runs all steps: data preparation, baseline training, BERT training, and evaluation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=False)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f}s")
    else:
        print(f"❌ {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def check_requirements():
    """Check if required directories and files exist."""
    required_dirs = [
        "src/",
        "data/",
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"Creating directory: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check for dataset
    data_file = Path("data/blooms.csv")
    if not data_file.exists():
        print("⚠️  WARNING: data/blooms.csv not found!")
        print("Please place your dataset in data/blooms.csv")
        print("Expected format: CSV with 'text' and 'label' columns")
        print("Labels should be BT1, BT2, BT3, BT4, BT5, BT6")
        
        # Create sample dataset for demo
        create_sample = input("Create a small sample dataset for testing? (y/n): ")
        if create_sample.lower() == 'y':
            create_sample_dataset()
        else:
            sys.exit(1)


def create_sample_dataset():
    """Create a small sample dataset for testing."""
    import pandas as pd
    
    sample_data = [
        {"text": "Define photosynthesis", "label": "BT1"},
        {"text": "List the parts of a cell", "label": "BT1"},
        {"text": "Explain how photosynthesis works", "label": "BT2"},
        {"text": "Describe the cell cycle", "label": "BT2"},
        {"text": "Calculate the area of a triangle", "label": "BT3"},
        {"text": "Apply Newton's laws to solve this problem", "label": "BT3"},
        {"text": "Compare and contrast mitosis and meiosis", "label": "BT4"},
        {"text": "Analyze the causes of World War II", "label": "BT4"},
        {"text": "Evaluate the effectiveness of this policy", "label": "BT5"},
        {"text": "Judge the validity of this argument", "label": "BT5"},
        {"text": "Design an experiment to test this hypothesis", "label": "BT6"},
        {"text": "Create a new solution to this problem", "label": "BT6"},
    ] * 20  # Duplicate to have more samples
    
    df = pd.DataFrame(sample_data)
    df.to_csv("data/blooms.csv", index=False)
    print("✅ Created sample dataset with 240 examples")


def main():
    """Run the complete pipeline."""
    print("🧠 BLOOM'S TAXONOMY CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Check requirements
    print("Checking requirements...")
    check_requirements()
    
    # Create output directories
    output_dirs = [
        "artifacts/data",
        "artifacts/models/baseline",
        "artifacts/models/bert",
        "artifacts/results"
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Pipeline steps
    steps = [
        ("python src/prepare_data.py", "Data Preparation"),
        ("python src/baseline.py", "Baseline Model Training"),
        ("python src/train_bert.py", "BERT Model Training"),
        ("python src/evaluate.py", "Model Evaluation")
    ]
    
    total_start_time = time.time()
    
    try:
        for command, description in steps:
            run_command(command, description)
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"\nResults saved in:")
        print(f"  - artifacts/models/baseline/ (Baseline model)")
        print(f"  - artifacts/models/bert/ (BERT model)")
        print(f"  - artifacts/results/ (Evaluation results)")
        
        print(f"\nTo run the Streamlit demo:")
        print(f"  streamlit run src/app.py")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()