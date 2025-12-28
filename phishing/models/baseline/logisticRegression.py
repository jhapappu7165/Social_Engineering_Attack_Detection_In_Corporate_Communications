import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

print('\n', '\n', "Datasets Loading and Combining ***", '\n')

print("*** For Dataset 1 ***")
df1 = pd.read_csv("datasets/dataset1/dataset1_modified.csv")
df1['text'] = df1['Email Text']
df1 = df1[['text', 'label']]
print("Number of emails in Dataset 1: ", len(df1))

print("*** For Dataset 2 ***")
df2 = pd.read_csv("datasets/dataset2/Ling.csv")
df2['text'] = df2['subject'].fillna('') + ' ' + df2['body'].fillna('') 
df2 = df2[['text', 'label']]
print("Number of emails in Dataset 2: ", len(df2))

print("*** For Dataset 3 ***")
df3 = pd.read_csv("datasets/dataset2/phishing_email.csv")
df3['text'] = df3['text_combined']
df3 = df3[['text', 'label']]
print("Number of emails in Dataset 3: ", len(df3))

df = pd.concat([df1, df2, df3])
print("Number of emails in combined dataset: ", len(df)) #103995 
print("Number of values in each type of label: ", df['label'].value_counts()) #0: 53318 and 1: 50677


print('\n', '\n', "*** Preprocessing the combined dataset ***", '\n')

df = df.dropna(subset=['text', 'label']) # drop rows with null values in text or label columns
print("AFTER DROPPING NULL VALUES: Number of emails in combined dataset: ", len(df)) # 103979 (16 dropped)
print("AFTER DROPPING NULL VALUES: Number of values in each type of label: ", df['label'].value_counts()) # 0(53318) & 1(50661)

df['text'] = df['text'].astype(str) #convert text to string type

df = df[df['text'].str.len() > 0] # drop rows with empty text
print('\n', "AFTER DROPPING EMPTY TEXT: Number of emails in combined dataset: ", len(df), " (0 dropped)") # 103979 (0 dropped)
print("AFTER DROPPING EMPTY TEXT: Number of values in each type of label: ", df['label'].value_counts()) # 0(53318) & 1(50661)

''' Nothing got dropped due to empty text '''

print('\n', '\n', "*** Extracting X and y values ***", '\n')
X = df['text'].values
y = df['label'].values

print("X shape: ", X.shape) # (103979,)
print("y shape: ", y.shape) # (103979,)
print("y distribution: ", np.bincount(y)) # [53318 50661] (0: 53318 and 1: 50661)


print('\n', '\n', "*** Splitting the dataset into training and testing sets ***", '\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) 

print("Training set size: ", len(X_train)) # 83183
print("Testing set size: ", len(X_test)) # 20796
print("y distribution in training set", np.bincount(y_train)) # [42654 40529] (0: 42654 and 1: 40529)
print("y distribution in testing set", np.bincount(y_test)) # [10664 10132] (0: 10664 and 1: 10132)



print('\n', '\n', "*** Vectorizing the text data ***", '\n')

vectorizer = TfidfVectorizer(max_features = 10000, ngram_range = (1, 2), min_df = 2, max_df = 0.95, stop_words = 'english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Train features shape: ", X_train_tfidf.shape) # (83183, 10000)
print("Test features shape: ", X_test_tfidf.shape) # (20796, 10000)

''' Converted all 83,183 training emails into a numeric matrix with 10,000 features (words). 
Each email becomes a vector of length 10,000, where each position represents a word (or token).'''



print('\n', '\n', "*** Training the Logistic Regression model ***", '\n')

lr_model = LogisticRegression(random_state = 42, C = 1.0, max_iter = 1000, penalty = 'l2', solver = 'lbfgs')

start_time = time.time() 
lr_model.fit(X_train_tfidf, y_train)
end_time = time.time()

training_time = end_time - start_time
print("Training time: ", round(training_time, 2), " seconds AND ", round(training_time/60, 2), " minutes")
print("Model trained successfully")


print('\n', '\n', "*** Making predictions on the testing set ***", '\n')

start_time = time.time()
y_pred = lr_model.predict(X_test_tfidf)
end_time = time.time()

testing_time = end_time - start_time
print("Testing time: ", round(testing_time, 2), " seconds AND ", round(testing_time/60, 2), " minutes")
print("Predictions made successfully")


print('\n', '\n', "*** Evaluating the model on the testing set ***", '\n')

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", round(accuracy, 4))
print("Precision: ", round(precision, 4))
print("Recall: ", round(recall, 4))
print("F1 Score: ", round(f1, 4))

print("Classification report: ", classification_report(y_test, y_pred, target_names = ['Safe', 'Phishing']))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred, labels = [0, 1]))
print("Model evaluation completed successfully")

print('\n', '\n', "*** Saving the model and vectorizer ***", '\n')
joblib.dump(lr_model, "models/baseline/logisticRegression.joblib")
joblib.dump(vectorizer, "models/baseline/vectorizer.joblib")
print("Model and vectorizer saved successfully to models/baseline/logisticRegression.joblib and models/baseline/vectorizer.joblib")
