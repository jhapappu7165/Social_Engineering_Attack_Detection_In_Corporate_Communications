import torch #a PyTorch library (build & train neural networks)
import torch.nn as nn # module for neural network layers + loss functions
from torch.utils.data import DataLoader, Dataset #wrap emails into Dataset, then DataLoader serves batches during training.
from transformers import ( #Hugging Face's Transformers (framework) library 
    BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
)
import pandas as pd #for data manipulation and analysis
import numpy as np #for numerical arrays and math
import time #for timing operations
import sys #for interpreter/system settings
import os #for operating system operations
from sklearn.metrics import accuracy_score, classification_report, confusion_matric, precision_score, recall_score, f1_score


''' Need to reach the baseline folder to import the commonBase.py file '''
current_file = __file__ #BERTcode.py
absolute_path = os.path.abspath(current_file) #/home/pappu/Research2/Social-Engineering-Attack-Detection-in-Corporate-Communications-/phishing/models/deepLearning/BERT/BERTcode.py
current_dir = os.path.dirname(absolute_path) #BERT/ folder
deepLearning_dir = os.path.dirname(current_dir) #deepLearning/ folder
models_dir = os.path.dirname(deepLearning_dir) #models/ folder
baseline_dir = os.path.join(models_dir, 'baseline') #models/baseline/ folder

if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from commonBase import (
    datasetCombining, preprocessing, splitting, predictions, results, savingModel, crossValidation, learningCurve, additionalMetrics, featureImportance, errorAnalysis, comprehensiveResults
)


''' BERT-specific Functions and Classes '''
class EmailDataset(Dataset): #custom dataset so PyTorch can fetch examples: email text,label,tokenized input

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts #list/series of email strings
        self.labels = labels #list/series of binary labels (0 or 1)
        self.tokenizer = tokenizer #Stores tokenizer so dataset can tokenize each text on-demand
        self.max_length = max_length #max token length rule for BERT inputs
    
    def __len__(self):
        return len(self.texts) #dataset size
    
    def __getitem__(self, idx): #returns one training example at index idx. DataLoader repeatedly calls this to build batches
        text = str(self.texts[idx]) #convert email text to string
        label = self.labels[idx] #get label for this example (0 or 1)

        encoding = self.tokenizer( #converts text into model-ready inputs: split text into subword tokens, convert tokens into integer IDs, create attention mask.
            text, 
            truncation=True, #if text is longer than max_length, cut it.
            padding='max_length', #pad shorter texts up to max_length.
            max_length=self.max_length, #max length of tokenized input.
            return_tensors='pt' #return PyTorch tensors, not Python lists.
        )

        return {
            'input_ids': encoding['input_ids'].flatten(), #flatten to 1D tensor
            'attention_mask': encoding['attention_mask'].flatten(), #tells BERT which tokens are real(1) vs padding(0)
            'labels': torch.tensor(label, dtype=torch.long) #convert label to long tensor
        }
