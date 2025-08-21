"""
Streamlit demo application for Bloom's Taxonomy classification.
Provides single prediction and batch prediction functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import io


class BloomsClassifier:
    def __init__(self):
        self.label_names = ['BT1', 'BT2', 'BT3', 'BT4', 'BT5', 'BT6']
        self.label_descriptions = {
            'BT1': 'Remember - Recall facts and basic concepts',
            'BT2': 'Understand - Explain ideas or concepts',
            'BT3': 'Apply - Use information in new situations',
            'BT4': 'Analyze - Draw connections among ideas',
            'BT5': 'Evaluate - Justify a stand or decision',
            'BT6': 'Create - Produce new or original work'
        }
        self.baseline_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.vectorizer = None
    
    @st.cache_resource
    def load_baseline_model(_self):
        """Load the baseline TF-IDF + LogReg model."""
        try:
            model_dir = Path("artifacts/models/baseline")
            vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
            classifier = joblib.load(model_dir / "logistic_regression.pkl")
            return vectorizer, classifier
        except Exception as e:
            st.error(f"Error loading baseline model: {e}")
            return None, None
    
    @st.cache_resource
    def load_bert_model(_self):
        """Load the fine-tuned BERT model."""
        try:
            model_dir = Path("artifacts/models/bert")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading BERT model: {e}")
            return None, None
    
    def predict_baseline(self, text):
        """Make prediction using baseline model."""
        if self.vectorizer is None or self.baseline_model is None:
            return None, None
        
        X = self.vectorizer.transform([text])
        prediction = self.baseline_model.predict(X)[0]
        probabilities = self.baseline_model.predict_proba(X)[0]
        
        return prediction, probabilities
    
    def predict_bert(self, text):
        """Make prediction using BERT model."""
        if self.bert_tokenizer is None or self.bert_model is None:
            return None, None
        
        inputs = self.bert_tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_class = torch.argmax(predictions, dim=1).item()
        probabilities = predictions[0].numpy()
        
        return predicted_class, probabilities
    
    def create_probability_chart(self, probabilities, model_name):
        """Create a probability distribution chart."""
        df = pd.DataFrame({
            'Level': self.label_names,
            'Probability': probabilities,
            'Description': [self.label_descriptions[level] for level in self.label_names]
        })
        
        fig = px.bar(
            df, 
            x='Level', 
            y='Probability',
            hover_data=['Description'],
            title=f'{model_name} - Probability Distribution',
            color='Probability',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Bloom's Taxonomy Level",
            yaxis_title="Probability",
            showlegend=False
        )
        
        return fig
    
    def process_batch(self, df, model_type):
        """Process batch predictions."""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, text in enumerate(df['text']):
            status_text.text(f'Processing {i+1}/{len(df)} texts...')
            progress_bar.progress((i + 1) / len(df))
            
            if model_type == "Baseline":
                pred, probs = self.predict_baseline(text)
            else:
                pred, probs = self.predict_bert(text)
            
            if pred is not None:
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'predicted_level': self.label_names[pred],
                    'confidence': probs[pred],
                    'description': self.label_descriptions[self.label_names[pred]]
                })
            else:
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'predicted_level': 'Error',
                    'confidence': 0.0,
                    'description': 'Prediction failed'
                })
        
        status_text.text('Processing complete!')
        progress_bar.progress(1.0)
        
        return pd.DataFrame(results)


def main():
    st.set_page_config(
        page_title="Bloom's Taxonomy Classifier",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Bloom's Taxonomy Question Classifier")
    st.markdown("Classify educational questions into Bloom's Taxonomy levels using AI models")
    
    # Initialize classifier
    classifier = BloomsClassifier()
    
    # Sidebar for model selection and info
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Baseline (TF-IDF + LogReg)", "BERT (Fine-tuned)"]
    )
    
    # Load models based on selection
    if "Baseline" in model_choice:
        if classifier.vectorizer is None:
            with st.spinner("Loading baseline model..."):
                classifier.vectorizer, classifier.baseline_model = classifier.load_baseline_model()
    else:
        if classifier.bert_tokenizer is None:
            with st.spinner("Loading BERT model..."):
                classifier.bert_tokenizer, classifier.bert_model = classifier.load_bert_model()
    
    # Sidebar info
    st.sidebar.header("Bloom's Taxonomy Levels")
    for level, desc in classifier.label_descriptions.items():
        st.sidebar.write(f"**{level}**: {desc}")
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.header("Single Question Classification")
        
        # Text input
        user_input = st.text_area(
            "Enter an educational question or task:",
            height=150,
            placeholder="Example: What are the key differences between mitosis and meiosis?"
        )
        
        if st.button("Classify Question", type="primary"):
            if user_input.strip():
                with st.spinner("Classifying..."):
                    if "Baseline" in model_choice:
                        prediction, probabilities = classifier.predict_baseline(user_input)
                    else:
                        prediction, probabilities = classifier.predict_bert(user_input)
                
                if prediction is not None:
                    predicted_level = classifier.label_names[prediction]
                    confidence = probabilities[prediction]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Predicted Level: {predicted_level}**")
                        st.write(f"**Description:** {classifier.label_descriptions[predicted_level]}")
                        st.write(f"**Confidence:** {confidence:.2%}")
                    
                    with col2:
                        # Probability chart
                        fig = classifier.create_probability_chart(probabilities, model_choice)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show all probabilities
                    st.subheader("Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Level': classifier.label_names,
                        'Probability': [f"{p:.2%}" for p in probabilities],
                        'Description': [classifier.label_descriptions[level] for level in classifier.label_names]
                    })
                    st.dataframe(prob_df, use_container_width=True)
                    
                else:
                    st.error("Failed to make prediction. Please check if the model is loaded correctly.")
            else:
                st.warning("Please enter a question to classify.")
    
    with tab2:
        st.header("Batch Question Classification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with questions",
            type=['csv'],
            help="CSV should have a 'text' column containing the questions"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column with the questions.")
                else:
                    st.write(f"Loaded {len(df)} questions")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Process button
                    if st.button("Process Batch", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            results_df = classifier.process_batch(df, model_choice.split()[0])
                        
                        st.success(f"Processed {len(results_df)} questions!")
                        
                        # Show results
                        st.subheader("Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_str = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_str,
                            file_name="bloom_classification_results.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            level_counts = results_df['predicted_level'].value_counts()
                            fig = px.pie(
                                values=level_counts.values,
                                names=level_counts.index,
                                title="Distribution of Predicted Levels"
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.2%}")
                            
                            high_conf = (results_df['confidence'] > 0.8).sum()
                            st.metric("High Confidence Predictions (>80%)", 
                                    f"{high_conf} / {len(results_df)}")
                            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, scikit-learn, and HuggingFace Transformers")


if __name__ == "__main__":
    main()