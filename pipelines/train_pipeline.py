import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import json
import time
from pathlib import Path

# Resolve relative paths against project root so that all runs find dependencies
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from pipelines.data_pipeline import data_pipeline
from src.model_training import create_trainer_from_config
from src.model_evaluation import ModelEvaluator
from src.model_building import get_model_builder
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
from utils.config import get_model_config, get_data_paths
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model_performance_visualizations(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    evaluation_results: dict, 
    artifacts_dir: str, 
    model_name: str
):
    """Create comprehensive model performance visualizations (confusion matrix, ROC, feature importance)."""
    try:
        # Create model-specific directory
        model_dir = os.path.join(artifacts_dir, f"model_performance_{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        if 'cm' in evaluation_results:
            plt.figure(figsize=(8, 6))
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
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log confusion matrix as artifact
            mlflow.log_artifact(cm_path, f"model_performance/{model_name}")
        
        # 2. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Plot top 15 features
            top_features = feature_importance.tail(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 15 Feature Importances')
            plt.tight_layout()
            
            importance_path = os.path.join(model_dir, f'feature_importance_{model_name}.png')
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance as JSON
            importance_json_path = os.path.join(model_dir, f'feature_importance_{model_name}.json')
            feature_importance.to_json(importance_json_path, indent=2)
            
            # Log artifacts
            mlflow.log_artifact(importance_path, f"model_performance/{model_name}")
            mlflow.log_artifact(importance_json_path, f"model_performance/{model_name}")
        
        # 3. ROC Curve (if probabilities available)
        try:
            from sklearn.metrics import roc_curve, auc
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
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
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log ROC curve
                mlflow.log_artifact(roc_path, f"model_performance/{model_name}")
                mlflow.log_metric(f'{model_name}_roc_auc', roc_auc)
                
        except Exception as e:
            logger.warning(f"Could not create ROC curve: {str(e)}")
        
        # 4. Prediction Distribution
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prediction distribution
            pred_counts = pd.Series(y_pred).value_counts()
            axes[0].bar(['Legitimate', 'Fraud'], [pred_counts.get(0, 0), pred_counts.get(1, 0)], color=['g', 'r'])
            axes[0].set_title('Prediction Distribution')
            axes[0].set_ylabel('Count')
            
            # Probability distribution
            axes[1].hist(y_proba, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1].set_xlabel('Fraud Probability')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Fraud Probability Distribution')
            
            plt.suptitle(f'{model_name} - Prediction Analysis')
            plt.tight_layout()
            
            pred_dist_path = os.path.join(model_dir, f'prediction_distribution_{model_name}.png')
            plt.savefig(pred_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(pred_dist_path, f"model_performance/{model_name}")
            
        except Exception as e:
            logger.warning(f"Could not create prediction distribution: {str(e)}")
        
        logger.info(f"✓ Model performance visualizations created for {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to create model performance visualizations: {str(e)}")


def log_model_metadata(
    model, 
    model_name: str, 
    model_params: dict, 
    training_time: float, 
    artifacts_dir: str
):
    """Log comprehensive model metadata."""
    try:
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'model_parameters': model_params,
            'training_time_seconds': training_time,
            'sklearn_version': None,
            'model_size_mb': None,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Try to get sklearn version
        try:
            import sklearn
            metadata['sklearn_version'] = sklearn.__version__
        except:
            pass
        
        # Try to get model size
        try:
            model_path = os.path.join(artifacts_dir, f'temp_{model_name}_model.pkl')
            joblib.dump(model, model_path)
            metadata['model_size_mb'] = os.path.getsize(model_path) / (1024**2)
            os.remove(model_path)  # Clean up temp file
        except:
            pass
        
        # Save metadata
        metadata_path = os.path.join(artifacts_dir, f'model_metadata_{model_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Log as MLflow artifact
        mlflow.log_artifact(metadata_path, f"model_metadata/{model_name}")
        
        # Log key metadata as parameters and metrics
        mlflow.log_params({
            f'{model_name}_model_type': type(model).__name__,
            f'{model_name}_sklearn_version': metadata.get('sklearn_version', 'unknown')
        })
        
        mlflow.log_metrics({
            f'{model_name}_training_time_seconds': training_time,
            f'{model_name}_model_size_mb': metadata.get('model_size_mb', 0)
        })
        
        logger.info(f"✓ Model metadata logged for {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to log model metadata: {str(e)}")


def training_pipeline(
    data_path: str = 'dataset/raw/fraudTrain.csv',
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    model_path: str = 'artifacts/models/xgboost_tuned_model.pkl',
    force_rebuild_data: bool = False
) -> Dict[str, Any]:
    """
    Executes model training pipeline:
    1. Loads dataset splits (NPZ) - running data_pipeline if missing
    2. Builds model using get_model_builder
    3. Trains model using ModelTrainer (Simple or CV based on config)
    4. Evaluates model and logs metrics/plots to MLflow
    5. Saves model to disk
    """
    logger.info(f"\n{'='*80}")
    logger.info("STARTING TRAINING PIPELINE — Credit Card Fraud Detection")
    logger.info(f"{'='*80}")

    data_paths = get_data_paths()
    model_cfg = get_model_config()

    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    # Create target directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    x_train_path = os.path.join(PROJECT_ROOT, data_paths.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
    x_test_path  = os.path.join(PROJECT_ROOT, data_paths.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
    y_train_path = os.path.join(PROJECT_ROOT, data_paths.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
    y_test_path  = os.path.join(PROJECT_ROOT, data_paths.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
    features_json_path = os.path.join(PROJECT_ROOT, 'artifacts/data/features.json')

    # Trigger data pipeline if splits are missing
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

    # Initialize MLflow tracking
    mlflow_tracker = None
    run = None
    try:
        mlflow_tracker = MLflowTracker()
        run_name = f"{model_cfg.get('model_type', 'xgboost')}_{model_cfg.get('training_strategy', 'cv')}"
        run_tags = create_mlflow_run_tags('training_pipeline', {
            'model_type': model_cfg.get('model_type', 'xgboost'),
            'training_strategy': model_cfg.get('training_strategy', 'cv'),
            'data_path': data_path,
            'model_path': model_path
        })
        run = mlflow_tracker.start_run(run_name=run_name, tags=run_tags)
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow tracking: {e}")

    # Build model
    model_type = model_cfg.get('model_type', 'xgboost')
    if model_params is None:
        model_params = model_cfg.get('model_params', {})
    
    logger.info(f"Building model type '{model_type}' with parameters: {model_params}")
    model_builder = get_model_builder(model_type, **model_params)
    model = model_builder.build_model()

    # Get model trainer from config
    trainer = create_trainer_from_config()

    # Train model
    logger.info("Starting model training...")
    training_start_time = time.time()
    trained_model, training_metrics = trainer.train(model, X_train, y_train)
    training_time = time.time() - training_start_time
    logger.info(f"✓ Model training completed in {training_time:.2f} seconds")

    # Save model
    trainer.save_model(trained_model, model_path)
    logger.info(f"✓ Model saved to: {model_path}")

    # Create run artifacts directory
    run_artifacts_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'mlflow_training_artifacts')
    if run:
        run_artifacts_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'mlflow_training_artifacts', run.info.run_id)
    os.makedirs(run_artifacts_dir, exist_ok=True)

    # Evaluate model
    logger.info("Evaluating model performance on test set...")
    evaluator = ModelEvaluator(trained_model, model_type)
    evaluation_results = evaluator.evaluate(X_test, y_test)
    evaluation_results_cp = evaluation_results.copy()

    # Log model to MLflow
    if run:
        try:
            mlflow.log_artifact(model_path, "trained_models")
            
            # Create visualizations
            create_model_performance_visualizations(
                trained_model, X_test, y_test, evaluation_results, 
                run_artifacts_dir, model_type
            )
            
            # Log metadata
            log_model_metadata(trained_model, model_type, model_params, training_time, run_artifacts_dir)
            
            # Remove confusion matrix for metrics logging
            if 'cm' in evaluation_results_cp:
                del evaluation_results_cp['cm']
                
            evaluation_results_cp.update({
                'training_time_seconds': training_time,
                'model_complexity': getattr(trained_model, 'n_estimators', 0),
                'max_depth': getattr(trained_model, 'max_depth', 0)
            })
            
            mlflow_tracker.log_training_metrics(trained_model, evaluation_results_cp, model_params)

            # Log training summary
            training_summary = {
                'model_type': model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': X_train.shape[1],
                'training_time': training_time,
                'model_path': model_path,
                'performance_metrics': evaluation_results_cp,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            summary_path = os.path.join(run_artifacts_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
                
            mlflow.log_artifact(summary_path, "training_summary")
            mlflow_tracker.end_run()
            logger.info("✓ MLflow tracking metrics and artifacts successfully uploaded.")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            try:
                mlflow_tracker.end_run()
            except:
                pass

    logger.info(f"\n{'='*80}")
    logger.info("✓ TRAINING PIPELINE COMPLETED")
    logger.info(f"{'='*80}\n")
    
    return {
        'model': trained_model,
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_results
    }


if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params', {})
    
    try:
        results = training_pipeline(model_params=model_params)
        eval_metrics = results['evaluation_metrics']
        
        # Display summary
        print("\n" + "=" * 60)
        print("[TRAINING PIPELINE SUMMARY]")
        print("=" * 60)
        print(f"[OK] Trained Model Type : {type(results['model']).__name__}")
        print(f"[OK] Accuracy on Test Set: {eval_metrics.get('accuracy', 0.0):.4%}")
        print(f"[OK] Precision on Test:    {eval_metrics.get('precision', 0.0):.4%}")
        print(f"[OK] Recall on Test Set:   {eval_metrics.get('recall', 0.0):.4%}")
        print(f"[OK] F1 Score on Test Set: {eval_metrics.get('f1', 0.0):.4%}")
        if 'roc_auc' in eval_metrics:
            print(f"[OK] ROC AUC Score:        {eval_metrics.get('roc_auc', 0.0):.4%}")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Failed to execute training pipeline: {e}")
        sys.exit(1)