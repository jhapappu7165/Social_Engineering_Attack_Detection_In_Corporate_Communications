import pandas as pd

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


'''
1. sender, receiver, date, subject, body => OUTPUT: label [1: 21842, 0: 17312]
2. subject, body ==> OUTPUT: label [0: 15791, 1: 13976]
3. subject, body ==> OUTPUT: label [0: 2401, 1: 458]
4. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 1565]
5. sender, receiver, date, subject, body, urls ==> OUTPUT: label [0: 0, 1: 3332]
6. text_combined ==> OUTPUT: label [0: 39595, 1: 42891]
7. sender, receiver, date, subject, body ==> OUTPUT: label [0: 4091, 1: 1718]
'''