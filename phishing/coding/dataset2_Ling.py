import pandas as pd
import numpy as np
import re
from collections import Counter

df = pd.read_csv('Social-Engineering-Attack-Detection-in-Corporate-Communications-/phishing/datasets/dataset2/Ling.csv')

print("*** DATASET 2 ANALYSIS OF LING.CSV ***")
print(df['subject'].head())
