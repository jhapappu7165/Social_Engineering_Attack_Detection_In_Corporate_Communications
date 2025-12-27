import pandas as pd
import numpy as np
import re
from collections import Counter

df = pd.read_csv('datasets/dataset2/phishing_email.csv')

print("* Dataset 3 Analysis of Phishing vs Safe Emails *")

print(df.head()) 
print("Names of columns: ", df.columns) #2 columns: text_combined, label
print("Length of dataset: ", len(df)) #82486 emails total
print("Shape of dataset: ", df.shape) 

print(df['label'].value_counts()) #1(42891) and 0(39595)
print(df['label'].nunique()) #2 unique labels: 1 and 0

unique_text = df['text_combined'].nunique()
duplicate_text = df['text_combined'].duplicated().sum()

print('count of unique and duplicate texts is: ', unique_text, duplicate_text)
# 82078 unique texts and 408 duplicate texts

df['text_len'] = df['text_combined'].str.len() #new column called text_len

print("Average length of texts is: ", round(df['text_len'].mean(), 2))
print("Median length of texts is: ", round(df['text_len'].median(), 2))

print("Minimum length of texts is: ", df['text_len'].min()) #1
print("Maximum length of texts is: ", df['text_len'].max()) #4279526

count0 = count1 = countX = 0
for label in df['label']:
    if label == 0:
        count0 += 1
    elif label == 1:
        count1 += 1
    else:
        countX += 1

print("zero_count: ", count0, "one_count: ", count1, "other_count: ", countX)
# zero_count:  39595 one_count:  42891 other_count:  0