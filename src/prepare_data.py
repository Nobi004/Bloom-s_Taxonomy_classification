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
        

