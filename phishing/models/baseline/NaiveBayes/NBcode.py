import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
import sys
import os


current_file = __file__
absolute_path = os.path.abspath(current_file)
current_dir = os.path.dirname(absolute_path)
baseline_dir = os.path.dirname(current_dir)

if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from commonBase import (
    datasetsCombining, preprocessing, splitting, vectorizing, predictions, 
    results, savingModel, crossValidation, learningCurve, additionalMetrics, 
    featureImportance, errorAnalysis, comprehensiveResults, hyperparameterTuning,
    classWeightAnalysis
)



def training(X_train_tfidf, y_train, X_val_tfidf, y_val):
    print('\n', '\n', "*** Training the Naive Bayes model ***", '\n')

    # Use MultinomialNB for text (works with sparse matrices)
    nb_model = MultinomialNB()

    start_time = time.time()
    nb_model.fit(X_train_tfidf, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    print("Training Time: ", round(training_time, 2), " seconds AND ", round(training_time/60, 2), " minutes")
    print("Model trained successfully")

    return nb_model, training_time



def alphaTuning(X_train_tfidf, y_train, X_val_tfidf, y_val):
    '''
    Purpose: Analyzes how different alpha (smoothing) values affect Naive Bayes performance.
    - Alpha is the Laplace smoothing parameter (prevents zero probabilities)
    - Tests different alpha values to find the best regularization
    - Compares training vs validation accuracy
    - Alpha = 0 means no smoothing (can lead to zero probabilities)
    - Alpha > 0 adds small counts to avoid zero probabilities
    '''
    print('\n', '\n', "*** Alpha Tuning for Naive Bayes ***", '\n')
    
    # Test different alpha values (smoothing strength)
    alpha_values = [1e-10, 1e-8, 1e-6, 1e-4, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
    train_scores = []
    val_scores = []
    
    for alpha in alpha_values:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train_tfidf, y_train)
        
        train_pred = model.predict(X_train_tfidf)
        val_pred = model.predict(X_val_tfidf)
        
        train_scores.append(accuracy_score(y_train, train_pred))
        val_scores.append(accuracy_score(y_val, val_pred))
    
    # Plot alpha tuning analysis
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, train_scores, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(alpha_values, val_scores, 'o-', label='Validation Accuracy', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Alpha (Smoothing Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Alpha Tuning for Naive Bayes')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/baseline/NaiveBayes/NBalphatuning.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Alpha tuning plot saved successfully")
    
    # Find best alpha value
    best_idx = np.argmax(val_scores)
    best_alpha = alpha_values[best_idx]
    print("Best alpha value: {} (Validation Accuracy: {:.4f})".format(best_alpha, val_scores[best_idx]))
    
    return alpha_values, train_scores, val_scores, best_alpha



if __name__ == "__main__":
    # Load and preprocess
    df = datasetsCombining()
    X, y = preprocessing(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(X, y)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = vectorizing(X_train, X_val, X_test)

    # Train Naive Bayes
    nb_model, training_time = training(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # Cross-validation and learning curve
    cv_scores = crossValidation(nb_model, X_train_tfidf, y_train)
    learningCurve(nb_model, X_train_tfidf, y_train, save_path='models/baseline/NaiveBayes/NBlearning_curve.png')

    # Optional: tune alpha (smoothing) for MultinomialNB
    alpha_values, train_scores, val_scores, best_alpha = alphaTuning(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # Retrain MultinomialNB with best alpha on training set
    nb_model = MultinomialNB(alpha=best_alpha)
    nb_model.fit(X_train_tfidf, y_train)

    # Predictions and results
    y_pred, y_pred_proba, testing_time = predictions(nb_model, X_test_tfidf, y_test)
    results(y_test, y_pred)

    # Additional metrics (ROC, average precision)
    roc_auc, avg_precision, y_pred_proba = additionalMetrics(
        nb_model, X_test_tfidf, y_test,
        save_path_prefix='models/baseline/NaiveBayes/NB'
    )

    # Feature importance (may not be available for all NB variants)
    try:
        feature_importance_df = featureImportance(
            nb_model, vectorizer,
            save_path='models/baseline/NaiveBayes/NBfeature_importance.csv'
        )
    except Exception as e:
        feature_importance_df = None
        print("Feature importance not available for Naive Bayes:", e)

    # Error analysis
    misclassified_df, fp_indices, fn_indices = errorAnalysis(
        nb_model, X_test, X_test_tfidf, y_test, y_pred, y_pred_proba,
        save_path='models/baseline/NaiveBayes/NBmisclassified_samples.csv'
    )

    # Comprehensive results
    comprehensive_results = comprehensiveResults(
        nb_model, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test,
        y_pred, y_pred_proba, cv_scores, roc_auc, avg_precision,
        training_time, testing_time,
        model_name='Naive Bayes',
        best_params={'alpha': best_alpha},
        save_path='models/baseline/NaiveBayes/NBcomprehensive_results.json'
    )

    # Save model and vectorizer
    savingModel(nb_model, vectorizer, 'NB', 'models/baseline/NaiveBayes')

    print('\n', '\n', "*** ALL NB ANALYSIS COMPLETE ***", '\n')