import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
import time
import sys
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matric, precision_score, recall_score, f1_score

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
    datasetCombining, preprocessing, splitting, predictions, results, savingModel, crossValidation, learningCurve, additionalMetrics, featureImportance, errorAnalysis, comprehensiveResults
)


''' BERT-specific Functions and Classes '''
