''' 0: safe, 1: phishing '''

import pandas as pd
import numpy as np
import re
from collections import Counter

df = pd.read_csv('datasets/dataset2/Ling.csv')

print("*** DATASET 2 ANALYSIS OF LING.CSV ***")
print(df['subject'].head())

print('Names of columns: ', df.columns) #subject, body, label
print(f"Names of columns:, {list(df.columns)}")

print("Shape of dataset: ", df.shape) #(2859, 3)
print("Total number of emails:", len(df)) #2859

unique_subject = df['subject'].nunique()
duplicate_subject = df['subject'].duplicated().sum()

unique_body = df['body'].nunique()
duplicate_body = df['body'].duplicated().sum()

label_counts = df['label'].value_counts() #labels: risk, nan, safe

print("Count of LABELS is: ", label_counts)

print('count of unique and duplicate subjects is: ', unique_subject, duplicate_subject)
print('count of unique and duplicate bodies is: ', unique_body, duplicate_body)

#Note: value_counts() return all unique values (so good in categorical data, not in texts)
#nunique() return number of unique values (just one number)
#duplicated() gives a boolean series (True/False), so sum() required


df['sub_len'] = df['subject'].str.len() #creates a new column
df['body_len'] = df['body'].str.len()
# No need for label_len as it is categorical data (risk, nan, safe)

print("Average length of subjects is: ", round(df['sub_len'].mean(), 2))
print("Average length of bodies is: ", round(df['body_len'].mean(), 2))

print("Median length of subjects is: ", round(df['sub_len'].median(), 2))
print("Median length of bodies is: ", round(df['body_len'].median(), 2))

# Label counts
count0 = count1 = countX = 0
for label in df['label']:
    if label == 0:
        count0 += 1
    elif label == 1:
        count1 += 1
    else:
        countX += 1

print("zero_count: ", count0, "one_count: ", count1, "other_count: ", countX)
print(df['label'].nunique())
# CONCLUSION: ONLY TWO LABELS (0 and 1) [zero_count:  2401 one_count:  458 other_count:  0]