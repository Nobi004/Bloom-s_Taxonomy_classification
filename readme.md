# 🧠 Bloom's Taxonomy Classification Project

A complete machine learning pipeline for classifying educational questions and tasks into Bloom's Taxonomy levels (BT1-BT6) using both traditional ML and transformer models.

## 📋 Overview

This project implements a comprehensive solution for educational content classification:

- **BT1 (Remember)**: Recall facts and basic concepts
- **BT2 (Understand)**: Explain ideas or concepts  
- **BT3 (Apply)**: Use information in new situations
- **BT4 (Analyze)**: Draw connections among ideas
- **BT5 (Evaluate)**: Justify a stand or decision
- **BT6 (Create)**: Produce new or original work

## 🏗️ Project Structure

```
repo/
├── requirements.txt          # Python dependencies
├── run_pipeline.py          # Complete pipeline runner
├── README.md               # This file
├── src/
│   ├── prepare_data.py     # Data preprocessing
│   ├── baseline.py         # TF-IDF + Logistic Regression
│   ├── train_bert.py       # BERT fine-tuning
│   ├── evaluate.py         # Model comparison
│   └── app.py             # Streamlit demo
├── data/
│   └── blooms.csv         # Dataset (user-provided)
└── artifacts/
    ├── data/              # Processed data splits
    ├── models/            # Trained models
    └── results/           # Evaluation results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset in `data/blooms.csv` with the following format:

```csv
text,label
"What is photosynthesis?",BT1
"Explain how photosynthesis works",BT2
"Apply the concept of photosynthesis to design an experiment",BT3
...
```

**Requirements:**
- CSV format with `text` and `label` columns
- Labels must be: BT1, BT2, BT3, BT4, BT5, BT6
- Minimum ~100 examples per class recommended

### 3. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will automatically:
1. Process and split your data (80/10/10 train/val/test)
2. Train baseline TF-IDF + Logistic Regression model
3. Fine-tune BERT model with early stopping
4. Generate comprehensive evaluation comparison

### 4. Launch Demo

```bash
streamlit run src/app.py
```

## 📊 Models

### Baseline Model
- **Vectorization**: TF-IDF (1-2 grams, max 10k features)
- **Classifier**: Logistic Regression with L2 regularization
- **Fast training**: ~1-5 minutes depending on dataset size

### BERT Model  
- **Base Model**: `bert-base-uncased`
- **Fine-tuning**: 5 epochs with early stopping on validation F1
- **Optimization**: AdamW with learning rate 2e-5
- **Training time**: ~10-60 minutes depending on dataset size and hardware

## 🔧 Manual Usage

If you prefer running components individually:

```bash
# 1. Data preparation
python src/prepare_data.py

# 2. Train baseline
python src/baseline.py

# 3. Train BERT (requires GPU for reasonable speed)
python src/train_bert.py

# 4. Generate comparison
python src/evaluate.py

# 5. Run demo
streamlit run src/app.py
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro-averaged F1 (handles class imbalance)
- **Per-class metrics**: Precision, recall, F1 for each BT level
- **Confusion matrices**: Visual analysis of classification patterns
- **Comparison plots**: Side-by-side model performance

## 🖥️ Streamlit Demo Features

### Single Prediction
- Enter any educational question
- Get predicted Bloom level + confidence
- View probability distribution across all levels
- Choose between Baseline and BERT models

### Batch Prediction
- Upload CSV file with questions
- Process multiple questions at once
- Download results with predictions and confidence scores
- View summary statistics and distribution plots

## ⚙️ Configuration

### Model Hyperparameters

**Baseline (baseline.py)**:
```python
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)
```

**BERT (train_bert.py)**:
```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    early_stopping_patience=3
)
```

### Data Splits
- Train: 80%
- Validation: 10% 
- Test: 10%
- Stratified sampling ensures balanced class distribution

## 🔍 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage if needed
export CUDA_VISIBLE_DEVICES=""
```

**2. Memory Issues**
- Reduce batch size in `train_bert.py`
- Use gradient checkpointing for BERT training
- Process data in smaller chunks

**3. Model Loading Issues**
- Ensure models are trained before running demo
- Check file paths in `artifacts/models/`

**4. Dataset Format**
```python
# Verify your CSV format
import pandas as pd
df = pd.read_csv('data/blooms.csv')
print(df.head())
print(df['label'].value_counts())
```

## 📝 Sample Dataset Creation

If you don't have a dataset, the pipeline can create a small sample for testing:

```bash
python run_pipeline.py
# Choose 'y' when prompted to create sample dataset
```

## 🎯 Performance Tips

### For Better Results:
1. **Data Quality**: Clean, well-labeled examples
2. **Data Size**: 500+ examples per class recommended
3. **Hardware**: GPU significantly speeds up BERT training
4. **Hypertuning**: Adjust learning rates and batch sizes

### Expected Performance:
- **Baseline**: 60-80% accuracy (depending on dataset quality)
- **BERT**: 70-90% accuracy (usually 5-15% improvement over baseline)

## 📚 Dependencies

Key libraries:
- `transformers`: HuggingFace transformers for BERT
- `torch`: PyTorch for deep learning
- `scikit-learn`: Traditional ML algorithms
- `streamlit`: Web interface
- `pandas`: Data manipulation
- `matplotlib/plotly`: Visualization

## 🤝 Contributing

To extend this project:

1. **Add new models**: Implement in `src/` following existing patterns
2. **Enhance preprocessing**: Modify `prepare_data.py`
3. **Improve demo**: Add features to `app.py`
4. **Add evaluation metrics**: Extend `evaluate.py`

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your dataset format matches requirements
3. Ensure all dependencies are installed correctly

---

**Happy classifying! 🎓📊**