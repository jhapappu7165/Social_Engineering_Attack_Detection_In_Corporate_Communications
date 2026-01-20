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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


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


''' GET THE CODE BELOW REVIEWED AND EDITED '''

def trainBertModel(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, device='cpu'):
    print('\n', '\n', "*** Training BERT Model ***", '\n')

    optimizer = AdamW(model.parameters(), lr=learning_rate) #Creates AdamW optimizer
    total_steps = len(train_loader) * num_epochs #number of batches per epoch times total epochs (how many steps the training will take total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    model = model.to(device) #Places the model on GPU (if available) or CPU. Must match where tensors are placed for computation.
    training_loss_history = []
    validation_loss_history = [] #store loss values
    
    for epoch in range(num_epochs): #loop over each epoch
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model.train() # Set model to training mode to enable training-specific behavior
        total_train_loss = 0 #loss across all batches
        train_correct = 0 #number of correct predictions
        train_total = 0 #total number of samples/examples
        
        for batch in train_loader: #Loop through each batch in the training DataLoader
            input_ids = batch['input_ids'].to(device) #Token IDs for the email text
            attention_mask = batch['attention_mask'].to(device) #1=real token, 0=padding
            labels = batch['labels'].to(device) #0=safe, 1=phishing
            
            optimizer.zero_grad() #Clear previous gradients
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss #cross-entropy loss for this batch
            logits = outputs.logits
            
            total_train_loss += loss.item()
            
            # Track accuracy
            predictions = torch.argmax(logits, dim=1) #argmax returns index 0 or 1
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        training_loss_history.append(avg_train_loss)
        
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_val_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        validation_loss_history.append(avg_val_loss)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("\nTraining completed successfully")
    return training_loss_history, validation_loss_history
