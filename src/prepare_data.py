import pandas as pd
import numpy as np  
import re 
import os     
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class BloomsDataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_mapping = {
            'BT1': 0, 'BT2': 1, 'BT3': 2, 'BT4': 3, 'BT5': 4, 'BT6': 5
        }
        self.reverse_mapping = {v: k for k ,v in self.label_mapping.items()}

    def clean_text(self,text):
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-]', '', text)

        return text 
        
    def encode_labels(self,labels):
        """Encode BT1-BT6 labels to 0-5."""
        encoded = []
        for label in labels:
            if label in self.label_mapping:
                encoded.append(self.label_mapping[label])
            else:
                print(f"Warning: Unknown label {label}, mapping to BT1")
                encoded.append(0)
        return np.array(encoded)
    
    def decode_labels(self,encoded_labels):
        return [self.reverse_mapping[label] for lable in encoded_labels]
    
    def load_and_preprocess(self,csv_path):
        print(f"Loading data from {csv_path}")

        df = pd.read_csv(csv_path)

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        print(f"Loaded {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        # Clean text 
        df['cleaned_text'] = df['text'].apply(self.clean_text)

        df = df[df['cleaned_text'] != ""].reset_index(drop=True)
        print(f"After cleaning: {len(df)} samples")

        df['encoded_label'] = self.encode_labels(df['label'])

        return df          
    
    




