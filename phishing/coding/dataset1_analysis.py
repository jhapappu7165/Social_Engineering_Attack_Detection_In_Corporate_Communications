# ISSUE: pre-defined keywords for phishing emails and safe emails

import pandas as pd
import numpy as np 
import re
from collections import Counter 


# Load dataset
df = pd.read_csv("phishing/datasets/Ds1:Phishing_Email.csv", index_col=0)


print("=" * 80)
print("DATASET 1 ANALYSIS: Ds1:Phishing_Email.csv")
print("=" * 80)
print()


# ============================================================================
# 1. BASIC DATASET INFO
# ============================================================================
print("1. DATASET OVERVIEW")
print(f"Total emails: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Dataset shape: {df.shape}")


# ============================================================================
# 2. EMAIL TYPE DISTRIBUTION
# ============================================================================
print("2. EMAIL TYPE DISTRIBUTION")
counts = df['Email Type'].value_counts()
percentages = df['Email Type'].value_counts(normalize=True) * 100

for email_type in counts.index:
    print(f"{email_type}: {counts[email_type]:6d} emails ({percentages[email_type]:6.2f}%)")


# 3. EMAIL LENGTH ANALYSIS
print("3. EMAIL LENGTH ANALYSIS")
df['text_length'] = df['Email Text'].str.len()
df['word_count'] = df['Email Text'].str.split().str.len()

print(f"Average length (chars): {df['text_length'].mean():.2f}")
print(f"Median length (chars): {df['text_length'].median():.2f}")
print(f"Min length: {df['text_length'].min():.0f}, Max length: {df['text_length'].max():.0f}")

print(f"Average word count: {df['word_count'].mean():.2f}")
print(f"Median word count: {df['word_count'].median():.2f}")
print(f"Min words: {df['word_count'].min():.0f}, Max words: {df['word_count'].max():.0f}")



# By email type
print("Length comparison by type:")
for email_type in df['Email Type'].unique():
    subset = df[df['Email Type'] == email_type]
    print(f"  {email_type}:")
    print(f"    Avg: {subset['text_length'].mean():.0f} chars, {subset['word_count'].mean():.0f} words")
print()



# 4. PHISHING EMAIL KEYWORDS
print("4. PHISHING EMAIL KEYWORD ANALYSIS")
phishing_emails = df[df['Email Type'] == 'Phishing Email']['Email Text'].str.lower()
safe_emails = df[df['Email Type'] == 'Safe Email']['Email Text'].str.lower()

phishing_keywords = ['click', 'free', 'offer', 'money', 'account', 'verify', 'confirm', 
                     'update', 'password', 'urgent', 'limited', 'required', 'loan', 'prize']

print("Top keywords in PHISHING emails:")
keyword_counts = {}
for keyword in phishing_keywords:
    count = phishing_emails.str.contains(keyword, case=False, regex=False).sum()
    if count > 0:
        keyword_counts[keyword] = count

sorted_kw = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
for keyword, count in sorted_kw[:10]:
    pct = (count / len(phishing_emails)) * 100
    print(f"  '{keyword}': {count:4d} ({pct:5.2f}%)")
print()


# ============================================================================
# 5. SAFE EMAIL KEYWORDS
# ============================================================================
print("5. SAFE EMAIL KEYWORD ANALYSIS")
print("-" * 80)
safe_keywords = ['meeting', 'work', 'please', 'thanks', 'regards', 'forward', 'deal',
                 'subject', 'cc', 'attached', 'business', 'request']

print("Top keywords in SAFE emails:")
keyword_counts_safe = {}
for keyword in safe_keywords:
    count = safe_emails.str.contains(keyword, case=False, regex=False).sum()
    if count > 0:
        keyword_counts_safe[keyword] = count

sorted_kw_safe = sorted(keyword_counts_safe.items(), key=lambda x: x[1], reverse=True)
for keyword, count in sorted_kw_safe[:10]:
    pct = (count / len(safe_emails)) * 100
    print(f"  '{keyword}': {count:4d} ({pct:5.2f}%)")
print()


# ============================================================================
# 6. URL/LINK DETECTION
# ============================================================================
print("6. URL AND LINK DETECTION")
print("-" * 80)
df['has_url'] = df['Email Text'].str.contains(r'http|www\.', case=False, na=False, regex=True)
url_count = df['has_url'].sum()
print(f"Total emails with URLs: {url_count} ({url_count/len(df)*100:.2f}%)")

for email_type in df['Email Type'].unique():
    subset = df[df['Email Type'] == email_type]
    urls = subset['has_url'].sum()
    print(f"  {email_type}: {urls} ({urls/len(subset)*100:.2f}%)")
print()


# ============================================================================
# 7. EMAIL ADDRESSES AND PHONE NUMBERS
# ============================================================================
print("7. CONTACT INFORMATION")
print("-" * 80)
df['has_email'] = df['Email Text'].str.contains(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 
                                                  regex=True, na=False)
df['has_phone'] = df['Email Text'].str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', regex=True, na=False)

print(f"Emails containing email addresses: {df['has_email'].sum()}")
print(f"Emails containing phone numbers: {df['has_phone'].sum()}")
print()

# ============================================================================
# 8. SUSPICIOUS PATTERNS
# ============================================================================
print("8. SUSPICIOUS PATTERNS")
print("-" * 80)
patterns = {
    'Dollar amounts ($)': r'\$[\d,]+',
    'Repeated punctuation (!! ??)': r'[!?]{2,}',
    'Urgent language': r'urgent|immediate|now|asap',
}

for pattern_name, pattern in patterns.items():
    total = df['Email Text'].str.contains(pattern, regex=True, case=False).sum()
    phishing = df[df['Email Type'] == 'Phishing Email']['Email Text'].str.contains(pattern, regex=True, case=False).sum()
    safe = df[df['Email Type'] == 'Safe Email']['Email Text'].str.contains(pattern, regex=True, case=False).sum()
    print(f"{pattern_name}:")
    print(f"  Total: {total}, Phishing: {phishing}, Safe: {safe}")
print()

# ============================================================================
# 9. DATA QUALITY
# ============================================================================
print("9. DATA QUALITY CHECKS")
print("-" * 80)
print(f"Null values in 'Email Text': {df['Email Text'].isna().sum()}")
print(f"Null values in 'Email Type': {df['Email Type'].isna().sum()}")
print(f"Empty emails: {(df['Email Text'] == '').sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print()

# ============================================================================
# 10. SAMPLE EMAILS
# ============================================================================
print("10. SAMPLE EMAILS")
print("-" * 80)
print("\nFirst PHISHING email:")
phishing_sample = df[df['Email Type'] == 'Phishing Email'].iloc[0]
print(f"Length: {len(phishing_sample['Email Text'])} chars")
print(f"Text: {phishing_sample['Email Text'][:200]}...")
print()

print("First SAFE email:")
safe_sample = df[df['Email Type'] == 'Safe Email'].iloc[0]
print(f"Length: {len(safe_sample['Email Text'])} chars")
print(f"Text: {safe_sample['Email Text'][:200]}...")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
