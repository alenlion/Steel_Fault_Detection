# ===========================================
# Models Module - Project 2: Machine Learning
# ===========================================

"""
Module for training and evaluating machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models():
    """
    Get dictionary of all models to compare.
    """
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    
    return models


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Train a model and return evaluation metrics.
    """
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC-AUC for multi-class
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics, y_pred


def compare_models(X_train, X_test, y_train, y_test, models=None):
    """
    Compare multiple models and return results DataFrame.
    """
    
    if models is None:
        models = get_models()
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        metrics, y_pred = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })
        
        trained_models[name] = model
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\nModel comparison completed!")
    
    return results_df, trained_models


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return scores.
    """
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    return {
        'cv_scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }


def get_classification_report_str(model, X_test, y_test, target_names=None):
    """
    Get detailed classification report as string.
    """
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)


def get_confusion_matrix(model, X_test, y_test):
    """
    Get confusion matrix.
    """
    
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)

