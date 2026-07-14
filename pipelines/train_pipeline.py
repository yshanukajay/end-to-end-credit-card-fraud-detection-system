import os
import sys
import yaml
import time
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional

# Resolve relative paths against project root so that all runs find dependencies
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from pipelines.data_pipeline import data_pipeline
from src.model_building import get_model_builder
from src.model_training import create_trainer_from_config
from src.model_evaluation import ModelEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)

# Try to import mlflow to track experiments
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠ mlflow package not found. MLflow tracking will be skipped.")


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_performance_visualizations(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    evaluation_results: dict, 
    model_dir: str, 
    model_name: str
):
    """Create comprehensive model performance visualizations (confusion matrix, ROC, feature importance)."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        if 'cm' in evaluation_results:
            plt.figure(figsize=(7, 6))
            sns.heatmap(
                evaluation_results['cm'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'], 
                yticklabels=['Legitimate', 'Fraud']
            )
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(model_dir, f'confusion_matrix_{model_name}.png')
            plt.savefig(cm_path, dpi=300)
            plt.close()
            
            logger.info(f"  ✓ Saved confusion matrix plot to: {cm_path}")
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifact(cm_path, "plots")
        
        # 2. Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = model.feature_importances_
            feature_names = X_test.columns
            indices = np.argsort(importance)[::-1]
            
            sns.barplot(x=importance[indices], y=feature_names[indices], palette="viridis")
            plt.title(f'{model_name} - Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            
            importance_path = os.path.join(model_dir, f'feature_importance_{model_name}.png')
            plt.savefig(importance_path, dpi=300)
            plt.close()
            
            logger.info(f"  ✓ Saved feature importance plot to: {importance_path}")
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_artifact(importance_path, "plots")
        
        # 3. ROC Curve
        try:
            from sklearn.metrics import roc_curve, auc
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(7, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                roc_path = os.path.join(model_dir, f'roc_curve_{model_name}.png')
                plt.savefig(roc_path, dpi=300)
                plt.close()
                
                logger.info(f"  ✓ Saved ROC curve plot to: {roc_path}")
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_artifact(roc_path, "plots")
        except Exception as e:
            logger.warning(f"Could not create ROC curve: {e}")
            
    except Exception as e:
        logger.error(f"✗ Failed to create model performance visualizations: {e}")


def train_pipeline(force_rebuild_data: bool = False) -> Dict[str, Any]:
    """
    Executes the training pipeline:
    1. Loads dataset splits (running data pipeline if they don't exist)
    2. Builds the target classifier model based on config.yaml
    3. Trains the classifier using configured training strategy (Simple vs KFold CV)
    4. Evaluates the trained model on test set, computing metrics and plots
    5. Saves the trained model to disk
    6. Tracks execution parameters and metrics in MLflow
    """
    logger.info(f"\n{'='*80}")
    logger.info("STARTING TRAINING PIPELINE — Credit Card Fraud Detection")
    logger.info(f"{'='*80}")
    
    # Load configuration
    config = load_config()
    data_paths_cfg = config.get('data_paths', {})
    model_cfg = config.get('model', {})
    mlflow_cfg = config.get('mlflow', {})
    
    # Construct target directories
    artifacts_data_dir = os.path.join(PROJECT_ROOT, data_paths_cfg.get('data_artifacts_dir', 'artifacts/data'))
    artifacts_model_dir = os.path.join(PROJECT_ROOT, data_paths_cfg.get('model_artifacts_dir', 'artifacts/models'))
    os.makedirs(artifacts_model_dir, exist_ok=True)
    
    x_train_path = os.path.join(PROJECT_ROOT, data_paths_cfg.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
    x_test_path  = os.path.join(PROJECT_ROOT, data_paths_cfg.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
    y_train_path = os.path.join(PROJECT_ROOT, data_paths_cfg.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
    y_test_path  = os.path.join(PROJECT_ROOT, data_paths_cfg.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
    features_json_path = os.path.join(artifacts_data_dir, 'features.json')
    
    # 1. Trigger data pipeline if splits are missing
    data_exists = all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path, features_json_path])
    if not data_exists or force_rebuild_data:
        logger.info("Training data splits not found or force_rebuild set. Triggering data pipeline...")
        data_pipeline(force_rebuild=True)
    else:
        logger.info("✓ Loading existing data splits from data pipeline.")
        
    # Load splits from NPZ
    X_train_arr = np.load(x_train_path)['X_train']
    X_test_arr  = np.load(x_test_path)['X_test']
    y_train_arr = np.load(y_train_path)['y_train']
    y_test_arr  = np.load(y_test_path)['y_test']
    
    with open(features_json_path, 'r') as f:
        feature_names = json.load(f)
        
    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr, columns=feature_names)
    y_train = pd.Series(y_train_arr)
    y_test  = pd.Series(y_test_arr)
    
    logger.info(f"Loaded datasets successfully:")
    logger.info(f"  • X_train: {X_train.shape}")
    logger.info(f"  • X_test:  {X_test.shape}")
    logger.info(f"  • y_train: {y_train.shape} (Fraud: {int(y_train.sum())}, Legitimate: {len(y_train) - int(y_train.sum())})")
    logger.info(f"  • y_test:  {y_test.shape} (Fraud: {int(y_test.sum())}, Legitimate: {len(y_test) - int(y_test.sum())})")
    
    # 2. Setup MLflow Tracking
    active_run = None
    if MLFLOW_AVAILABLE:
        try:
            tracking_uri = mlflow_cfg.get('tracking_uri', 'sqlite:///mlflow.db')
            experiment_name = mlflow_cfg.get('experiment_name', 'Credit Card Fraud Detection')
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            run_name = f"{model_cfg.get('model_type', 'random_forest')}_{model_cfg.get('training_strategy', 'cv')}"
            active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"✓ Started MLflow run: '{run_name}' in experiment '{experiment_name}'")
            
            # Log dataset stats
            mlflow.log_params({
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'num_features': X_train.shape[1],
                'features': list(X_train.columns)
            })
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking: {e}")
            
    # 3. Build Model
    model_type = model_cfg.get('model_type', 'random_forest')
    model_params = model_cfg.get('model_params', {})
    logger.info(f"Building model type '{model_type}' with parameters: {model_params}")
    model_builder = get_model_builder(model_type, **model_params)
    model = model_builder.build_model()
    
    # 4. Train Model
    logger.info("Initializing model trainer...")
    trainer = create_trainer_from_config()
    
    start_time = time.time()
    trained_model, training_metrics = trainer.train(model, X_train, y_train)
    training_time = time.time() - start_time
    
    # 5. Save Model
    dest_model_path = model_cfg.get('model_path', 'artifacts/model/random_forest_cv_model.pkl')
    if not os.path.isabs(dest_model_path):
        dest_model_path = os.path.join(PROJECT_ROOT, dest_model_path)
        
    trainer.save_model(trained_model, dest_model_path)
    
    # 6. Evaluate Model
    logger.info("Evaluating model performance on test set...")
    evaluator = ModelEvaluator(trained_model, model_type)
    evaluation_results = evaluator.evaluate(X_test, y_test)
    
    # Generate text report and plot
    report_path = model_cfg.get('evaluation_path', 'artifacts/model/evaluation_report.txt')
    if not os.path.isabs(report_path):
        report_path = os.path.join(PROJECT_ROOT, report_path)
        
    evaluator.save_evaluation_report(report_path)
    
    # Generate visual performance plots
    plots_dir = os.path.join(os.path.dirname(report_path), 'plots')
    create_model_performance_visualizations(
        trained_model, X_test, y_test, evaluation_results, plots_dir, model_type
    )
    
    # 7. Log to MLflow
    if MLFLOW_AVAILABLE and active_run:
        try:
            # Log model params
            mlflow.log_params({
                'model_type': model_type,
                'training_strategy': type(trainer.strategy).__name__,
                **model_params
            })
            
            # Log training metrics
            mlflow.log_metric('training_time_s', training_time)
            for k, v in training_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'train_{k}', v)
                    
            # Log evaluation metrics
            for k, v in evaluation_results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'test_{k}', v)
                    
            # Log artifacts
            mlflow.log_artifact(dest_model_path, "model")
            mlflow.log_artifact(report_path, "evaluation")
            
            # Auto-log model if possible
            try:
                if 'xgb' in model_type.lower():
                    mlflow.xgboost.log_model(trained_model, "xgb_model")
                else:
                    mlflow.sklearn.log_model(trained_model, "sklearn_model")
            except Exception as e:
                logger.warning(f"Could not auto-log model artifact: {e}")
                
            mlflow.end_run()
            logger.info("✓ MLflow tracking metrics and artifacts successfully uploaded.")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            
    logger.info(f"\n{'='*80}")
    logger.info("✓ TRAINING PIPELINE COMPLETED")
    logger.info(f"{'='*80}\n")
    
    return {
        'model': trained_model,
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_results
    }


if __name__ == '__main__':
    try:
        results = train_pipeline()
        eval_metrics = results['evaluation_metrics']
        
        # Display clean summary
        print("\n" + "=" * 60)
        print("[TRAINING PIPELINE SUMMARY]")
        print("=" * 60)
        print(f"[OK] Trained Model Type : {type(results['model']).__name__}")
        print(f"[OK] Accuracy on Test Set: {eval_metrics.get('accuracy', 0.0):.4%}")
        print(f"[OK] Precision on Test:    {eval_metrics.get('precision', 0.0):.4%}")
        print(f"[OK] Recall on Test Set:  {eval_metrics.get('recall', 0.0):.4%}")
        print(f"[OK] F1 Score on Test Set: {eval_metrics.get('f1', 0.0):.4%}")
        if 'roc_auc' in eval_metrics:
            print(f"[OK] ROC AUC Score:        {eval_metrics.get('roc_auc', 0.0):.4%}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to execute training pipeline: {e}")
        print(f"\n[FAILED] Training pipeline failed: {e}")
        sys.exit(1)