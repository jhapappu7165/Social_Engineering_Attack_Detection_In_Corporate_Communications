import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os

''' __file__ is for the current file (LRcode.py). os.path.dirname() pushes the folder level up by 1. os.path.abspath() gets the full path from the root.'''
current_dir = os.path.dirname(os.path.abspath(__file__))  # LogisticRegression folder
baseline_dir = os.path.dirname(current_dir)  # baseline folder

if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)


from commonBase import (
    datasetsCombining, preprocessing, splitting, vectorizing,
    predictions, results, savingModel, crossValidation, learningCurve,
    additionalMetrics, featureImportance, errorAnalysis, comprehensiveResults,
    hyperparameterTuning, classWeightAnalysis
)


def training(X_train_tfidf, y_train):
    print('\n', '\n', "*** Training the Logistic Regression model ***", '\n')

    lr_model = LogisticRegression(random_state = 42, C = 1.0, max_iter = 1000, penalty = 'l2', solver = 'lbfgs')

    start_time = time.time() 
    lr_model.fit(X_train_tfidf, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    print("Training time: ", round(training_time, 2), " seconds AND ", round(training_time/60, 2), " minutes")
    print("Model trained successfully")
    
    return lr_model, training_time



def regularizationAnalysis(X_train_tfidf, y_train, X_val_tfidf, y_val):
    '''
    Purpose: Analyzes how different C values affect overfitting.
    - Tests only C values (regularization strength)
    - Uses fixed penalty='l2' and solver='lbfgs'
    - Compares training vs validation accuracy
    '''
    print('\n', '\n', "*** Regularization Analysis ***", '\n')
    
    # Test different C values (regularization strength)
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    train_scores = []
    val_scores = []
    
    for C in C_values:
        model = LogisticRegression(C=C, random_state=42, max_iter=1000, solver='lbfgs', penalty='l2')
        model.fit(X_train_tfidf, y_train)
        
        train_pred = model.predict(X_train_tfidf)
        val_pred = model.predict(X_val_tfidf)
        
        train_scores.append(accuracy_score(y_train, train_pred))
        val_scores.append(accuracy_score(y_val, val_pred))
    
    # Plot regularization analysis
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_scores, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(C_values, val_scores, 'o-', label='Validation Accuracy', linewidth=2)
    plt.xscale('log')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Regularization Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/baseline/LogisticRegression/LRregularization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Regularization analysis plot saved successfully")
    
    # Find best C value
    best_idx = np.argmax(val_scores)
    best_C = C_values[best_idx]
    print("Best C value: {} (Validation Accuracy: {:.4f})".format(best_C, val_scores[best_idx]))
    
    return C_values, train_scores, val_scores, best_C


if __name__ == "__main__":
    # Step 1: Load and combine datasets
    df = datasetsCombining()
    
    # Step 2: Preprocess data
    X, y = preprocessing(df)
    
    # Step 3: Split into train/validation/test
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(X, y)
    
    # Step 4: Vectorize text data
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = vectorizing(X_train, X_val, X_test)
    
    # Step 5: Train model (LR-specific)
    lr_model, training_time = training(X_train_tfidf, y_train)
    
    # Step 6: Cross-validation (from commonBase)
    cv_scores = crossValidation(lr_model, X_train_tfidf, y_train)
    
    # Step 7: Learning curves (from commonBase)
    learningCurve(lr_model, X_train_tfidf, y_train, save_path='models/baseline/LogisticRegression/LRlearningCurve.png')
    
    # Step 8: Hyperparameter tuning (from commonBase, optional - can be commented out if takes too long)
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    # best_model, best_params = hyperparameterTuning(LogisticRegression(), param_grid, X_train_tfidf, y_train)
    # lr_model = best_model  # Use best model for remaining steps
    best_params = None  # Set to None if not using hyperparameter tuning
    
    # Step 9: Make predictions (from commonBase)
    y_pred, y_pred_proba, testing_time = predictions(lr_model, X_test_tfidf, y_test)
    
    # Step 10: Basic results (from commonBase)
    results(y_test, y_pred)
    
    # Step 11: Additional metrics (from commonBase)
    roc_auc, avg_precision, y_pred_proba = additionalMetrics(
        lr_model, X_test_tfidf, y_test, 
        save_path_prefix='models/baseline/LogisticRegression/LR'
    )
    
    # Step 12: Feature importance (from commonBase)
    feature_importance_df = featureImportance(
        lr_model, vectorizer, 
        save_path='models/baseline/LogisticRegression/LRfeature_importance.csv'
    )
    
    # Step 13: Error analysis (from commonBase)
    misclassified_df, fp_indices, fn_indices = errorAnalysis(
        lr_model, X_test, X_test_tfidf, y_test, y_pred, y_pred_proba,
        save_path='models/baseline/LogisticRegression/LRmisclassified_samples.csv'
    )
    
    # Step 14: Regularization analysis (LR-specific)
    C_values, train_scores, val_scores, best_C = regularizationAnalysis(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Step 15: Class weight analysis (from commonBase)
    model_params = {'random_state': 42, 'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'}
    lr_model_balanced, accuracy_balanced = classWeightAnalysis(
        LogisticRegression, model_params, X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    
    # Step 16: Comprehensive results (from commonBase)
    comprehensive_results = comprehensiveResults(
        lr_model, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test,
        y_pred, y_pred_proba, cv_scores, roc_auc, avg_precision,
        training_time, testing_time, 
        model_name='Logistic Regression',
        best_params=best_params,
        save_path='models/baseline/LogisticRegression/LRcomprehensive_results.json'
    )
    
    # Step 17: Save model (from commonBase)
    savingModel(lr_model, vectorizer, 'LR', 'models/baseline/LogisticRegression')
    
    print('\n', '\n', "*** ALL ANALYSIS COMPLETE ***", '\n')