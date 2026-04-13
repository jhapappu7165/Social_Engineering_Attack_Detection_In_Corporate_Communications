import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm


df1 = pd.read_csv('phishing/datasets/dataset2/CEAS_08.csv')
df2 = pd.read_csv('phishing/datasets/dataset2/Enron.csv')
df3 = pd.read_csv('phishing/datasets/dataset2/Ling.csv')
df4 = pd.read_csv('phishing/datasets/dataset2/Nazario.csv')
df5 = pd.read_csv('phishing/datasets/dataset2/Nigerian_Fraud.csv')
df6 = pd.read_csv('phishing/datasets/dataset2/phishing_email.csv')
df7 = pd.read_csv('phishing/datasets/dataset2/SpamAssasin.csv')

index = 0
for df in [df1, df2, df3, df4, df5, df6, df7]:
    index += 1
    print('\n', "=" * 80, '\n')

    print("Dataset: ", index, "of 7", '\n')
    print(df.shape) 

    cols = df.columns
    print(cols)
    for col in cols:
        print(col, ":", df[col].head())

    print(df.count())
    print(df['label'].value_counts()) 
    print(df.head())
    print(df.tail())


parts = []
for df in [df1, df2, df3, df4, df5, df6, df7]:
    if 'text_combined' in df.columns:
        text = df['text_combined'].astype(str)
    elif {'sender', 'receiver', 'date'}.issubset(df.columns):
        text = (
            df['sender'].fillna('').astype(str) + ' '
            + df['receiver'].fillna('').astype(str) + ' '
            + df['date'].fillna('').astype(str) + ' '
            + df['subject'].fillna('').astype(str) + ' '
            + df['body'].fillna('').astype(str)
        )
    else:
        text = df['subject'].fillna('').astype(str) + ' ' + df['body'].fillna('').astype(str)
    
    parts.append(pd.DataFrame({'text': text.str.strip(), 'label': df['label']}))

combined = pd.concat(parts, ignore_index=True)
X = combined[['text']]
y = combined['label'].values

texts = X['text'].astype(str).tolist()
y = np.asarray(y).astype(int)

x_train, x_temp, y_train, y_temp = train_test_split(texts, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
x_train = list(x_train) + list(x_val)
y_train = np.concatenate([y_train, y_val])

model_name = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


tokenizer = BertTokenizer.from_pretrained(model_name)
batch_size = 16
train_loader = DataLoader(EmailDataset(x_train, y_train, tokenizer), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(EmailDataset(x_test, y_test, tokenizer), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

train_pbar = tqdm(
    total=len(train_loader) * epochs,
    desc='Training',
    unit='batch',
    ncols=100,
)
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device),
        )
        loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_pbar.update(1)
        train_pbar.set_postfix(epoch=f'{epoch + 1}/{epochs}', loss=f'{loss.item():.4f}')
train_pbar.close()

save_dir = os.path.join(os.path.dirname(__file__), 'bert_finetuned')
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

model.eval()
y_pred = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Eval (test)', unit='batch', ncols=100):
        logits = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
        ).logits
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())

print('\nClassification report (test):\n', classification_report(y_test, y_pred, digits=4))
print('Confusion matrix (test):\n', confusion_matrix(y_test, y_pred))
print('Saved:', save_dir)


'''
1. sender, receiver, date, subject, body => OUTPUT: label [1: 21842, 0: 17312]
2. subject, body ==> OUTPUT: label [0: 15791, 1: 13976]
3. subject, body ==> OUTPUT: label [0: 2401, 1: 458]
4. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 1565]
5. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 3332]
6. text_combined ==> OUTPUT: label [0: 39595, 1: 42891]
7. sender, receiver, date, subject, body ==> OUTPUT: label [0: 4091, 1: 1718]
'''