import pandas as pd
import numpy as np 
import re
from collections import Counter 

df = pd.read_csv("phishing/datasets/Ds1:Phishing_Email.csv", index_col=0) #index_col=0 tells pandas to use the first column (column 0) as the DataFrame's index instead of creating a new numeric index.

# print(df.head()) # first 5 rows
# print(df.iloc[0]['Email Text']) #return email text of first row

# print(df.tail()) # last 5 rows
# print(df.iloc[18645]['Email Text'])
# print(df.iloc[-5]['Email Text']) # 18645th and -5th row are same

# print(len(df)) #length of dataset
# print(len(df.columns)) #length of columns
# print(df.columns) #names of columns

# print(df['Email Type'].value_counts()) #count of each type of email
# print('\n'*2)
# print(df['Email Type'].value_counts()*100/len(df))#percentage of each type of email
