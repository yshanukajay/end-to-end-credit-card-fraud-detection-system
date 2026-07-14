import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates trained machine learning models on a test set, generating
    classification metrics, confusion matrix, and saving results and plots.
    """
    def __init__(self, model: BaseEstimator, model_name: str):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        logger.info(f"ModelEvaluator initialized for model: {model_name}")

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        Y_test: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION - {self.model_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Test dataset shape: {X_test.shape}")
        
        # Generate predictions
        Y_pred = self.model.predict(X_test)
        
        # Attempt to get prediction probabilities if supported
        Y_proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                Y_proba = self.model.predict_proba(X_test)[:, 1]
            except Exception as e:
                logger.warning(f"Could not calculate prediction probabilities: {e}")
                
        # Metrics calculation
        acc = accuracy_score(Y_test, Y_pred)
        prec = precision_score(Y_test, Y_pred, zero_division=0)
        rec = recall_score(Y_test, Y_pred, zero_division=0)
        f1 = f1_score(Y_test, Y_pred, zero_division=0)
        cm = confusion_matrix(Y_test, Y_pred)
        
        self.evaluation_results = {
            'cm': cm,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        
        if Y_proba is not None:
            try:
                roc_auc = roc_auc_score(Y_test, Y_proba)
                avg_prec = average_precision_score(Y_test, Y_proba)
                self.evaluation_results['roc_auc'] = float(roc_auc)
                self.evaluation_results['average_precision'] = float(avg_prec)
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC / PR AUC: {e}")
            
        logger.info("\nEvaluation Metrics:")
        logger.info(f"  • Accuracy:          {self.evaluation_results['accuracy']:.4f}")
        logger.info(f"  • Precision:         {self.evaluation_results['precision']:.4f}")
        logger.info(f"  • Recall:            {self.evaluation_results['recall']:.4f}")
        logger.info(f"  • F1 Score:          {self.evaluation_results['f1']:.4f}")
        if 'roc_auc' in self.evaluation_results:
            logger.info(f"  • ROC AUC:           {self.evaluation_results['roc_auc']:.4f}")
            logger.info(f"  • PR AUC (Avg Prec): {self.evaluation_results['average_precision']:.4f}")
            
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"{'='*60}\n")
        
        return self.evaluation_results

    def save_evaluation_report(self, report_path: str) -> None:
        """
        Saves a text report of classification metrics to the specified path.
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Call evaluate() first.")
            
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"="*60 + "\n")
            f.write(f"MODEL EVALUATION REPORT: {self.model_name}\n")
            f.write(f"="*60 + "\n")
            f.write(f"Accuracy:          {self.evaluation_results['accuracy']:.6f}\n")
            f.write(f"Precision:         {self.evaluation_results['precision']:.6f}\n")
            f.write(f"Recall:            {self.evaluation_results['recall']:.6f}\n")
            f.write(f"F1 Score:          {self.evaluation_results['f1']:.6f}\n")
            if 'roc_auc' in self.evaluation_results:
                f.write(f"ROC AUC:           {self.evaluation_results['roc_auc']:.6f}\n")
                f.write(f"PR AUC (Avg Prec): {self.evaluation_results['average_precision']:.6f}\n")
            f.write("\nConfusion Matrix:\n")
            cm = self.evaluation_results['cm']
            f.write(f"[[{cm[0,0]} {cm[0,1]}]\n")
            f.write(f" [{cm[1,0]} {cm[1,1]}]]\n")
            f.write(f"="*60 + "\n")
            
        logger.info(f"✓ Saved evaluation report text to: {report_path}")

    def plot_confusion_matrix(self, save_path: str) -> None:
        """
        Plots the confusion matrix and saves it as an image.
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Call evaluate() first.")
            
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        cm = self.evaluation_results['cm']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"✓ Saved confusion matrix plot to: {save_path}")
