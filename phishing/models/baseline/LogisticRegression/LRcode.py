import pandas as pd
import numpy as np
import time
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier


def datasetsCombining():
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

    return df


def preprocessing(df):
    print('\n', '\n', "*** Preprocessing the combined dataset ***", '\n')

    df = df.dropna(subset=['text', 'label']) # drop rows with null values in text or label columns
    print("AFTER DROPPING NULL VALUES: Number of emails in combined dataset: ", len(df)) # 103979 (16 dropped)
    print("AFTER DROPPING NULL VALUES: Number of values in each type of label: ", df['label'].value_counts()) # 0(53318) & 1(50661)

    df['text'] = df['text'].astype(str) #convert text to string type

    df = df[df['text'].str.len() > 0] # drop rows with empty text
    print('\n', "AFTER DROPPING EMPTY TEXT: Number of emails in combined dataset: ", len(df), " (0 dropped)") # 103979 (0 dropped)
    print("AFTER DROPPING EMPTY TEXT: Number of values in each type of label: ", df['label'].value_counts()) # 0(53318) & 1(50661)

    ''' Nothing was dropped :=> No empty text emails'''

    print('\n', '\n', "*** Extracting X and y values ***", '\n')
    X = df['text'].values
    y = df['label'].values

    print("X shape: ", X.shape) # (103979,)
    print("y shape: ", y.shape) # (103979,)
    print("y distribution: ", np.bincount(y)) # [53318 50661] (0: 53318 and 1: 50661)

    return X, y

def splitting(X, y):
    print('\n', '\n', "*** Splitting the dataset into training, validation and testing sets ***", '\n')

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Second split: separate train and validation (80% -> 64% train, 16% (20% of 80%) validation) 
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

    print("Training set size: ", len(X_train))
    print("Validation set size: ", len(X_val))
    print("Testing set size: ", len(X_test))
    print("y distribution in training set: ", np.bincount(y_train))
    print("y distribution in validation set: ", np.bincount(y_val))
    print("y distribution in testing set: ", np.bincount(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorizing(X_train, X_val, X_test):
    print('\n', '\n', "*** Vectorizing the text (train, test, validation) data ***", '\n')

    vectorizer = TfidfVectorizer(max_features = 10000, ngram_range = (1, 2), min_df = 2, max_df = 0.95, stop_words = 'english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train features shape: ", X_train_tfidf.shape)
    print("Validation features shape: ", X_val_tfidf.shape)
    print("Test features shape: ", X_test_tfidf.shape)

    ''' Converted all training emails into a numeric matrix with 10,000 features (words). 
    Each email becomes a vector of length 10,000, where each position represents a word (or token).'''

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


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


def predictions(lr_model, X_test_tfidf, y_test):
    print('\n', '\n', "*** Making predictions on the testing set ***", '\n')
### QTA: How many emails are in the testing set?
    start_time = time.time()
    y_pred = lr_model.predict(X_test_tfidf)
    y_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
    end_time = time.time()

    testing_time = end_time - start_time
    print("Testing time: ", round(testing_time, 2), " seconds AND ", round(testing_time/60, 2), " minutes")
    print("Predictions made successfully")
    return y_pred, y_pred_proba, testing_time


def results(y_test, y_pred):
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


def savingModel(lr_model, vectorizer):
    print('\n', '\n', "*** Saving the MODEL and VECTORIZER ***", '\n')
    joblib.dump(lr_model, "models/baseline/LogisticRegression/LRmodel.joblib")
    joblib.dump(vectorizer, "models/baseline/LogisticRegression/LRvectorizer.joblib")
    print("Model and vectorizer saved successfully to models/baseline/LogisticRegression/LRmodel.joblib and models/baseline/LogisticRegression/LRvectorizer.joblib")


def crossValidation(lr_model, X_train_tfidf, y_train):
    ''' Splits training data into 5 parts. Trains on 4 parts, tests on 1 part. Repeats 5 times. Averages the results '''

    print('\n', '\n', "*** Performing Cross-validation ***", '\n')

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    cv_scores = cross_val_score(lr_model, X_train_tfidf, y_train, cv = cv, scoring = 'accuracy', n_jobs = -1)
### QTA: Why cross-validation is performed on the training set only?
    print("Cross-validation scores: ", cv_scores)
    print("Mean CV accuracy: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std() * 2))
    print("Min CV accuracy: {:.4f}".format(cv_scores.min()))
    print("Max CV accuracy: {:.4f}".format(cv_scores.max()))
    print("Cross-validation completed successfully")
    
    return cv_scores


def learningCurve(lr_model, X_train_tfidf, y_train):

    ''' Shows how model performance changes as training data increases. Training accuracy (blue line) and Validation accuracy (red line). Gap between them indicates overfitting '''
    print('\n', '\n', "*** LEARNING CURVE ***", '\n')

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    train_sizes, train_scores, val_scores = learning_curve(
        lr_model, X_train_tfidf, y_train, 
        cv = cv, 
        n_jobs = -1, 
        train_sizes = np.linspace(0.1, 1.0, 10), 
        scoring = 'accuracy'
    )
    
    train_mean = train_scores.mean(axis = 1) # axis = 1 means column
    train_std = train_scores.std(axis = 1)
    val_mean = val_scores.mean(axis = 1)
    val_std = val_scores.std(axis = 1)

    plt.figure(figsize = (10, 6))
    plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', label = 'Training accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'blue')
    plt.plot(train_sizes, val_mean, color = 'red', marker = 's', label = 'Validation accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha = 0.1, color = 'red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.savefig('models/baseline/LogisticRegression/LRlearningCurve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Learning curve plot saved successfully")
    return train_sizes, train_mean, val_mean, train_std, val_std


def hyperparameterTuning(X_train_tfidf, y_train):
    ''' Finds the best settings for the model automatically. TESTS: Different C values, penalties (L1 vs L2), solvers '''
    print('\n', '\n', "*** Hyperparameter Tuning with GridSearchCV ***", '\n')
    
    # Simplified parameter grid (adjust based on solver compatibility)
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Note: For 'l1' penalty, only 'liblinear' and 'saga' work
    # For 'l2' penalty, 'lbfgs' also works but excluded for consistency
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting grid search...")
    start_time = time.time()
    grid_search.fit(X_train_tfidf, y_train)
    end_time = time.time()
    
    print("Grid search completed in {:.2f} seconds".format(end_time - start_time))
    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    
    return grid_search.best_estimator_, grid_search.best_params_


def additionalMetrics(lr_model, X_test_tfidf, y_test):
    ''' Purpose: Provides more detailed performance metrics beyond accuracy, such as ROC-AUC, ROC curve, Precision-Recall curve'''
    print('\n', '\n', "*** Additional Metrics: ROC-AUC, ROC curve, Precision-Recall curve ***", '\n')
    
    # Get prediction probabilities
    y_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
### QTA: Why test set is used? how many test emails are there?
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("ROC-AUC Score: {:.4f}".format(roc_auc))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(roc_auc), linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/baseline/LogisticRegression/LRroc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC curve saved successfully")
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve (AP = {:.4f})'.format(avg_precision), linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/baseline/LogisticRegression/LRpr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Precision-Recall curve saved successfully")
    
    return roc_auc, avg_precision, y_pred_proba


def featureImportance(lr_model, vectorizer, top_n=100):
    print('\n', '\n', "*** Feature Importance Analysis ***", '\n')
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients (for binary classification)
    coefficients = lr_model.coef_[0]
    
    # Create DataFrame for analysis
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Sort by absolute coefficient value
    feature_importance_df = feature_importance_df.sort_values('abs_coefficient', ascending=False)
    
    # Top features for phishing detection (positive coefficients)
    print("Top {} Features for Phishing Detection (Positive coefficients):".format(top_n))
    phishing_features = feature_importance_df[feature_importance_df['coefficient'] > 0].head(top_n)
    print(phishing_features[['feature', 'coefficient']].to_string(index=False))
    
    # Top features for safe email detection (negative coefficients)
    print("\nTop {} Features for Safe Email Detection (Negative coefficients):".format(top_n))
    safe_features = feature_importance_df[feature_importance_df['coefficient'] < 0].head(top_n)
    print(safe_features[['feature', 'coefficient']].to_string(index=False))
    
    # Save to file
    feature_importance_df.head(100).to_csv(
        'models/baseline/LogisticRegression/LRfeature_importance.csv', 
        index=False
    )
    print("\nFeature importance saved to LRfeature_importance.csv")
    
    return feature_importance_df


def errorAnalysis(lr_model, X_test, X_test_tfidf, y_test, y_pred, y_pred_proba):
    print('\n', '\n', "*** Error Analysis ***", '\n')
    
    # Identify misclassified samples
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    print("Total misclassified: {} out of {}".format(len(misclassified_indices), len(y_test)))
    print("Misclassification rate: {:.2f}%".format(len(misclassified_indices)/len(y_test)*100))
    
    # False Positives (Safe emails predicted as Phishing)
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    print("\nFalse Positives (Safe → Phishing): {}".format(len(fp_indices)))
    
    # False Negatives (Phishing emails predicted as Safe)
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    print("False Negatives (Phishing → Safe): {}".format(len(fn_indices)))
    
    # Analyze misclassified samples
    misclassified_df = pd.DataFrame({
        'text': X_test[misclassified_indices],
        'true_label': y_test[misclassified_indices],
        'predicted_label': y_pred[misclassified_indices],
        'prediction_probability': y_pred_proba[misclassified_indices]
    })
    
    # Save misclassified samples for analysis
    misclassified_df.to_csv(
        'models/baseline/LogisticRegression/LRmisclassified_samples.csv', 
        index=False
    )
    print("Misclassified samples saved to LRmisclassified_samples.csv")
    
    # Show some examples
    if len(fp_indices) > 0:
        print("\nSample False Positives (Safe emails misclassified as Phishing):")
        for i, idx in enumerate(fp_indices[:3]):
            print("\nExample {}:".format(i+1))
            print("Text: {}...".format(X_test[idx][:200]))
            print("True: Safe, Predicted: Phishing, Probability: {:.4f}".format(y_pred_proba[idx]))
    
    if len(fn_indices) > 0:
        print("\nSample False Negatives (Phishing emails misclassified as Safe):")
        for i, idx in enumerate(fn_indices[:3]):
            print("\nExample {}:".format(i+1))
            print("Text: {}...".format(X_test[idx][:200]))
            print("True: Phishing, Predicted: Safe, Probability: {:.4f}".format(y_pred_proba[idx]))
    
    return misclassified_df, fp_indices, fn_indices


def regularizationAnalysis(X_train_tfidf, y_train, X_val_tfidf, y_val):
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


def classWeightAnalysis(X_train_tfidf, y_train, X_test_tfidf, y_test):
    print('\n', '\n', "*** Class Weight Analysis ***", '\n')
    
    # Check class distribution
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print("Class weights (balanced): {}".format(dict(zip(np.unique(y_train), class_weights))))
    
    # Train without class weights
    lr_model_no_weights = LogisticRegression(
        random_state=42, 
        C=1.0, 
        max_iter=1000, 
        solver='lbfgs'
    )
    lr_model_no_weights.fit(X_train_tfidf, y_train)
    y_pred_no_weights = lr_model_no_weights.predict(X_test_tfidf)
    accuracy_no_weights = accuracy_score(y_test, y_pred_no_weights)
    
    # Train with class weights
    lr_model_balanced = LogisticRegression(
        random_state=42, 
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced',
        solver='lbfgs'
    )
    lr_model_balanced.fit(X_train_tfidf, y_train)
    y_pred_balanced = lr_model_balanced.predict(X_test_tfidf)
    accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
    
    print("\nWithout class weights - Accuracy: {:.4f}".format(accuracy_no_weights))
    print("With class weights (balanced) - Accuracy: {:.4f}".format(accuracy_balanced))
    
    # Compare F1 scores
    f1_no_weights = f1_score(y_test, y_pred_no_weights)
    f1_balanced = f1_score(y_test, y_pred_balanced)
    print("Without class weights - F1 Score: {:.4f}".format(f1_no_weights))
    print("With class weights (balanced) - F1 Score: {:.4f}".format(f1_balanced))
    
    return lr_model_balanced, accuracy_balanced


def comprehensiveResults(lr_model, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, 
                        y_pred, y_pred_proba, cv_scores, roc_auc, avg_precision, 
                        training_time, testing_time, best_params=None):
    print('\n', '\n', "*** Comprehensive Results Report ***", '\n')
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Validation set metrics
    y_val_pred = lr_model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    # Create comprehensive results dictionary
    results = {
        'model': 'Logistic Regression',
        'dataset_info': {
            'train_size': X_train_tfidf.shape[0],
            'validation_size': X_val_tfidf.shape[0],
            'test_size': X_test_tfidf.shape[0],
            'total_samples': X_train_tfidf.shape[0] + X_val_tfidf.shape[0] + X_test_tfidf.shape[0]
        },
        'hyperparameters': best_params if best_params else {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000
        },
        'cross_validation': {
            'mean_score': float(cv_scores.mean()) if cv_scores is not None else None,
            'std_score': float(cv_scores.std()) if cv_scores is not None else None,
            'min_score': float(cv_scores.min()) if cv_scores is not None else None,
            'max_score': float(cv_scores.max()) if cv_scores is not None else None
        },
        'validation_set_metrics': {
            'accuracy': float(val_accuracy),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1_score': float(val_f1)
        },
        'test_set_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision)
        },
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        },
        'performance': {
            'training_time_seconds': float(training_time),
            'testing_time_seconds': float(testing_time)
        }
    }
    
    # Save results as JSON
    with open('models/baseline/LogisticRegression/LRcomprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Comprehensive results saved to LRcomprehensive_results.json")
    print("\nSummary:")
    print("=" * 60)
    print("Cross-Validation Accuracy: {:.4f} (+/- {:.4f})".format(
        results['cross_validation']['mean_score'], 
        results['cross_validation']['std_score'] * 2
    ))
    print("Validation Set Accuracy: {:.4f}".format(results['validation_set_metrics']['accuracy']))
    print("Test Set Accuracy: {:.4f}".format(results['test_set_metrics']['accuracy']))
    print("Test Set ROC-AUC: {:.4f}".format(results['test_set_metrics']['roc_auc']))
    print("=" * 60)
    
    return results


# Main execution function
if __name__ == "__main__":
    # Step 1: Load and combine datasets
    df = datasetsCombining()
    
    # Step 2: Preprocess data
    X, y = preprocessing(df)
    
    # Step 3: Split into train/validation/test
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(X, y)
    
    # Step 4: Vectorize text data
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = vectorizing(X_train, X_val, X_test)
    
    # Step 5: Train model
    lr_model, training_time = training(X_train_tfidf, y_train)
    
    # Step 6: Cross-validation
    cv_scores = crossValidation(lr_model, X_train_tfidf, y_train)
    
    # Step 7: Learning curves
    learningCurve(lr_model, X_train_tfidf, y_train)
    
    # Step 8: Hyperparameter tuning (optional - can be commented out if takes too long)
    # best_model, best_params = hyperparameterTuning(X_train_tfidf, y_train)
    # lr_model = best_model  # Use best model for remaining steps
    
    # Step 9: Make predictions
    y_pred, y_pred_proba, testing_time = predictions(lr_model, X_test_tfidf, y_test)
    
    # Step 10: Basic results
    results(y_test, y_pred)
    
    # Step 11: Additional metrics
    roc_auc, avg_precision, y_pred_proba = additionalMetrics(lr_model, X_test_tfidf, y_test)
    
    # Step 12: Feature importance
    feature_importance_df = featureImportance(lr_model, vectorizer)
    
    # Step 13: Error analysis
    misclassified_df, fp_indices, fn_indices = errorAnalysis(lr_model, X_test, X_test_tfidf, y_test, y_pred, y_pred_proba)
    
    # Step 14: Regularization analysis
    C_values, train_scores, val_scores, best_C = regularizationAnalysis(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Step 15: Class weight analysis
    lr_model_balanced, accuracy_balanced = classWeightAnalysis(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Step 16: Comprehensive results
    comprehensive_results = comprehensiveResults(
        lr_model, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test,
        y_pred, y_pred_proba, cv_scores, roc_auc, avg_precision,
        training_time, testing_time
    )
    
    # Step 17: Save model
    savingModel(lr_model, vectorizer)
    
    print('\n', '\n', "*** ALL ANALYSIS COMPLETE ***", '\n')

     