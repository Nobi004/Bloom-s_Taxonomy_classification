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
    
    def create_splits(self,df,test_size=0.1,val_size=0.1,random_state=42):
        """Create stratified train/val/test splits."""
        X = df['cleaned_text'].values
        y = df['encoded_label'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples") 
        print(f"Test: {len(X_test)} samples")
        
        return {
            'train': {'text': X_train, 'labels': y_train},
            'val': {'text': X_val, 'labels': y_val},
            'test': {'text': X_test, 'labels': y_test}
        }
    
    def save_splits(self, splits, output_dir):
        """Save train/val/test splits to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            split_df = pd.DataFrame({
                'text': split_data['text'],
                'labels': split_data['labels']
            })
            split_df.to_csv(f"{output_dir}/{split_name}.csv", index=False)
        
        # Save label encoder
        with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Splits saved to {output_dir}")

def main():
    """Main preprocessing pipeline"""
    data_path = "data/blooms.csv"
    output_dir = "artifacts/data"

    # Initialize preprocessor 
    preprocessor = BloomsDataPreprocessor()
    
    # Load and preprocess data 
    df = preprocessor.load_and_preprocess(data_path)
    splits = preprocessor.create_splits(df)
    preprocessor.save_splits(splits,output_dir)

    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()
    




