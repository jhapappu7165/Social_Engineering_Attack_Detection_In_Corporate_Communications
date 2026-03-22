''' 
CPU in Use, not GPU.
A change made to reduce speed: epochs to 2 (from 3)
'''

import torch #a PyTorch library (build & train neural networks)
import torch.nn as nn # module for neural network layers + loss functions
from torch.utils.data import DataLoader, Dataset #wrap emails into Dataset, then DataLoader serves batches during training.
from torch.optim import AdamW #AdamW optimizer (moved from transformers to torch.optim in newer versions)
from transformers import ( #Hugging Face's Transformers (framework) library 
    BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
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
    datasetsCombining, preprocessing, splitting, predictions, results, savingModel, crossValidation, learningCurve, additionalMetrics, featureImportance, errorAnalysis, comprehensiveResults
)


''' BERT-specific Functions and Classes '''
class EmailDataset(Dataset): #custom dataset so PyTorch can fetch examples: email text, label, tokenized input

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



def prepareBERTData(X_train, X_val, X_test, y_train, y_val, y_test, model_name='bert-base-uncased'): #default: 'bert-base-uncased'
    """
    Prepares data for BERT model.
    Creates tokenizer and DataLoaders for train/val/test sets.
    
    Parameters:
    - model_name: BERT model variant ('bert-base-uncased', 'bert-large-uncased', etc.)
    """
    print('\n', '\n', "*** Preparing Data for BERT ***", '\n')
    
    # Initialize BERT tokenizer for the specified BERT model (Converts text to token IDs that BERT expects)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"Using BERT model: {model_name}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    #Creates PyTorch Dataset objects using EmailDataset class. Each dataset wraps texts, labels, and the tokenizer.
    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    val_dataset = EmailDataset(X_val, y_val, tokenizer) 
    test_dataset = EmailDataset(X_test, y_test, tokenizer)
    '''
    EmailDataset: "How to process one email"
    DataLoader: "How to batch and iterate"
    '''
    # Create data loaders
    batch_size = 16  # Adjust based on GPU memory (16, 32, or 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle randomizes the order of the examples.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #number of batches per split in the train/validation/test loaders
    print(f"Train batches: {len(train_loader)}") 
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return tokenizer, train_loader, val_loader, test_loader

    '''
    Input: Raw text emails (X_train, X_val, X_test) + Labels (y_train, y_val, y_test)
    ↓
    Step 1: Load BERT tokenizer (converts text → token IDs)
    ↓
    Step 2: Create EmailDataset objects (wraps text + labels + tokenizer)
    ↓
    Step 3: Create DataLoaders (batches the datasets for efficient training)
    ↓
    Output: tokenizer + train_loader + val_loader + test_loader
    '''



def trainingBERT(model, train_loader, val_loader, device, num_epochs=2, learning_rate=2e-5):
    """
    Trains BERT model on phishing email dataset.
    
    Parameters:
    - model: BERT model (BertForSequenceClassification)
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - device: 'cuda' or 'cpu'
    - num_epochs: Number of training epochs (default: 3)
    - learning_rate: Learning rate (default: 2e-5, standard for BERT)
    """
    print('\n', '\n', "*** Training BERT Model ***", '\n')
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_accuracy = 0
    training_history = {'train_loss': [], 'val_accuracy': []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # Progress update
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_accuracy = evaluateBERT(model, val_loader, device)
        training_history['val_accuracy'].append(val_accuracy)
        
        print(f'\nAverage Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f'New best validation accuracy! Saving model...')
            # Model will be saved after training
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f'\nTraining completed in {training_time/60:.2f} minutes')
    print(f'Best validation accuracy: {best_val_accuracy:.4f}')
    
    return model, training_time, training_history


def evaluateBERT(model, data_loader, device):
    """
    Evaluates BERT model on a dataset.
    Returns accuracy.
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def predictBERT(model, test_loader, device):
    """
    Makes predictions using trained BERT model.
    Returns predictions and probabilities.
    """
    print('\n', '\n', "*** Making Predictions with BERT ***", '\n')
    
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    testing_time = end_time - start_time
    
    # Get probabilities for positive class (phishing = 1)
    y_pred_proba = np.array(probabilities)[:, 1]
    
    print(f"Testing time: {testing_time:.2f} seconds ({testing_time/60:.2f} minutes)")
    print("Predictions made successfully")
    
    return np.array(predictions), y_pred_proba, np.array(true_labels), testing_time


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Step 1: Load and combine datasets (from commonBase)
    df = datasetsCombining()
    
    # Step 2: Preprocess data (from commonBase)
    X, y = preprocessing(df)
    
    # Step 3: Split into train/validation/test (from commonBase)
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(X, y)
    
    # Step 4: Prepare data for BERT
    model_name = 'bert-base-uncased'  # Options: 'bert-base-uncased', 'bert-large-uncased', 'distilbert-base-uncased'
    tokenizer, train_loader, val_loader, test_loader = prepareBERTData(
        X_train, X_val, X_test, y_train, y_val, y_test, model_name=model_name
    )
    
    # Step 5: Initialize BERT model
    print('\n', '\n', "*** Initializing BERT Model ***", '\n')
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification: Safe (0) vs Phishing (1)
        output_attentions=False,
        output_hidden_states=False
    )
    print(f"Model initialized: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 6: Train BERT model
    num_epochs = 2  # BERT typically needs 2-4 epochs
    learning_rate = 2e-5  # Standard learning rate for BERT fine-tuning
    model, training_time, training_history = trainingBERT(
        model, train_loader, val_loader, device, 
        num_epochs=num_epochs, learning_rate=learning_rate
    )
    
    # Step 7: Make predictions
    y_pred, y_pred_proba, y_test_array, testing_time = predictBERT(model, test_loader, device)
    
    # Step 8: Evaluate model (from commonBase)
    results(y_test_array, y_pred)
    
    # Step 9: Additional metrics (from commonBase)
    # Note: We need to create a wrapper for BERT to use with commonBase functions
    # For now, calculate metrics directly
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_auc = roc_auc_score(y_test_array, y_pred_proba)
    avg_precision = average_precision_score(y_test_array, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Step 10: Save model and tokenizer
    print('\n', '\n', "*** Saving BERT Model and Tokenizer ***", '\n')
    model_save_path = 'models/deepLearning/BERT/bert_model'
    tokenizer_save_path = 'models/deepLearning/BERT/bert_tokenizer'
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Tokenizer saved to: {tokenizer_save_path}")
    
    # Step 11: Save training history
    import json
    with open('models/deepLearning/BERT/bert_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)
    print("Training history saved")
    
    print('\n', '\n', "*** BERT TRAINING COMPLETE ***", '\n')
