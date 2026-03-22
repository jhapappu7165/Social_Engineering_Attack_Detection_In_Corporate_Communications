import pandas as pd
import numpy as np

df = pd.read_csv('phishing/datasets/dataset2/Nigerian_Fraud.csv')

#print(df.head())
print("Shape: ", df.shape) #(3332, 7)
print("Columns: ", df.columns) #Index(['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label']
print("Label distribution: ", df['urls'].value_counts()) # 0: 2183, 1: 1149
print("Label distribution: ", df['label'].value_counts()) # 0: 0, 1: 3332
print("Data Types: ", df.dtypes) # sender: object, receiver: object, date: object, subject: object, body: object, urls: int64, label: int64
print("Missing values: ", df.isnull().sum()) # sender: 331, receiver: 1324, date: 482, subject: 39, body: 0, urls: 0, label: 0

print(df.iloc[0])
# sender         MR. JAMES NGOLA. <james_ngola2002@maktoob.com>
# receiver                                 webmaster@aclweb.org
# date                          Thu, 31 Oct 2002 02:38:20 +0000
# subject            URGENT BUSINESS ASSISTANCE AND PARTNERSHIP
# body        FROM:MR. JAMES NGOLA.\nCONFIDENTIAL TEL: 233-2...
# urls                                                        0
# label                                                       1


df['text'] = df['sender'].fillna('') + ' ' + df['receiver'].fillna('') + ' ' + df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['text'] = df['text'].str.strip()
print(df['text'].iloc[0])


df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
print(df[['text_length', 'word_count']].describe()) #describe() gives a detailed description
