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
    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        self.is_spark = type(model).__module__.startswith("pyspark.ml")
        self.evaluation_results = {}
        logger.info(f"ModelEvaluator initialized for model: {model_name} (is_spark={self.is_spark})")

    def evaluate(
        self,
        X_test: Any,
        Y_test: Any
    ) -> Dict[str, Any]:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION - {self.model_name.upper()}")
        logger.info(f"{'='*60}")
        
        if self.is_spark:
            from pyspark.sql import DataFrame
            logger.info("Evaluating using PySpark MLlib...")
            
            # Recombine features X_test and target Y_test if they are separate Spark DataFrames
            if isinstance(X_test, DataFrame) and isinstance(Y_test, DataFrame):
                from pyspark.sql.window import Window
                from pyspark.sql.functions import row_number, lit
                w = Window.orderBy(lit(1))
                X_indexed = X_test.withColumn("row_id", row_number().over(w))
                Y_indexed = Y_test.withColumn("row_id", row_number().over(w))
                test_df = X_indexed.join(Y_indexed, "row_id").drop("row_id")
                target_col = Y_test.columns[0]
            else:
                test_df = X_test
                target_col = "is_fraud"
                
            logger.info(f"Test dataset rows: {test_df.count()}")
            predictions_df = self.model.transform(test_df)
            
            # Convert only prediction and target columns to Pandas for metrics
            pdf = predictions_df.select(target_col, "prediction", "probability").toPandas()
            Y_test_vals = pdf[target_col].values
            Y_pred = pdf["prediction"].values
            Y_proba = np.array([float(val[1]) for val in pdf["probability"]])
        else:
            logger.info(f"Test dataset shape: {X_test.shape}")
            Y_test_vals = Y_test
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
        acc = accuracy_score(Y_test_vals, Y_pred)
        prec = precision_score(Y_test_vals, Y_pred, zero_division=0)
        rec = recall_score(Y_test_vals, Y_pred, zero_division=0)
        f1 = f1_score(Y_test_vals, Y_pred, zero_division=0)
        cm = confusion_matrix(Y_test_vals, Y_pred)
        
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
