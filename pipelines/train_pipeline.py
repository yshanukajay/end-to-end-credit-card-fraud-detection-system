import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Resolve relative paths against project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from pipelines.data_pipeline import data_pipeline
from utils.spark_session import create_spark_session, stop_spark_session
from utils.spark_utils import spark_to_pandas

from src.model_training import ModelTrainer, create_trainer_from_config
from src.model_evaluation import ModelEvaluator
from src.model_building import XGBoostModelBuilder as XGboostModelBuilder, RandomForestModelBuilder, get_model_builder
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
from utils.config import get_model_config, get_data_paths, get_config
import mlflow
import mlflow.spark
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def training_pipeline(
    data_path: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    model_path: Optional[str] = None,
    data_format: str = 'parquet'
):
    """
    Execute comprehensive model training pipeline with structured logging.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING MACHINE LEARNING TRAINING PIPELINE")
    logger.info(f"{'='*80}")

    data_paths = get_data_paths()
    model_cfg = get_model_config()
    
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, model_cfg.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl'))
        
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Run data pipeline first to ensure outputs exist
    data_pipeline(data_path=data_path, test_size=test_size)

    # Initialize Spark session
    spark = create_spark_session("CreditCardFraudDetectionTrainingPipeline")
    
    try:
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
            'training_pipeline', {
                'model_type' : 'XGboost',
                'training_strategy' : 'simple',
                'data_path': data_path,
                'model_path': model_path,
                'data_format': data_format,
                'processing_engine': 'pyspark'
            }
        )
        run = mlflow_tracker.start_run(run_name='training_pipeline', tags=run_tags)
        
        # Create artifacts directory for this run
        run_artifacts_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'mlflow_training_artifacts', run.info.run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)

        # Load training data with logging
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA LOADING STEP")
        logger.info(f"{'='*80}")
        
        processed_dir = os.path.join(PROJECT_ROOT, 'artifacts/data')
        
        config = get_config()
        model_cfg = config.get('model', {})
        framework = model_cfg.get('framework', 'scikit-learn')
        spark_model_type = model_cfg.get('spark_model_type', 'gbt')

        if framework == 'pyspark':
            logger.info(f"Loading training and test datasets (format: {data_format}) using PySpark...")
            if data_format == 'parquet':
                X_train_spark = spark.read.parquet(f"{processed_dir}/X_train.parquet")
                Y_train_spark = spark.read.parquet(f"{processed_dir}/Y_train.parquet")
                X_test_spark = spark.read.parquet(f"{processed_dir}/X_test.parquet")
                Y_test_spark = spark.read.parquet(f"{processed_dir}/Y_test.parquet")
            else:
                X_train_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{processed_dir}/X_train.csv")
                Y_train_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{processed_dir}/Y_train.csv")
                X_test_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{processed_dir}/X_test.csv")
                Y_test_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{processed_dir}/Y_test.csv")

            # Recombine features and label for PySpark ML compatibility
            from pyspark.sql.window import Window
            from pyspark.sql.functions import row_number, lit
            w = Window.orderBy(lit(1))
            X_train_indexed = X_train_spark.withColumn("row_id", row_number().over(w))
            Y_train_indexed = Y_train_spark.withColumn("row_id", row_number().over(w))
            train_spark_full = X_train_indexed.join(Y_train_indexed, "row_id").drop("row_id")
            
            # Split train data into sub-train and validation sets for threshold tuning
            train_spark_df, val_spark_df = train_spark_full.randomSplit([0.85, 0.15], seed=42)
            
            # Under-sample majority class to handle class imbalance
            logger.info("Applying RandomUnderSampler (PySpark) to handle class imbalance (target ratio 1:20)...")
            fraud_df = train_spark_df.filter(train_spark_df.is_fraud == 1)
            legit_df = train_spark_df.filter(train_spark_df.is_fraud == 0)
            
            fraud_count = fraud_df.count()
            legit_count = legit_df.count()
            target_legit_count = fraud_count * 20
            
            if target_legit_count < legit_count:
                fraction = target_legit_count / legit_count
                sampled_legit_df = legit_df.sample(withReplacement=False, fraction=fraction, seed=42)
                train_spark_resampled = fraud_df.union(sampled_legit_df)
            else:
                train_spark_resampled = train_spark_df
                
            logger.info(f"✓ Resampled training set size: {train_spark_resampled.count()} rows")
            
            num_train_samples = train_spark_full.count()
            num_test_samples = X_test_spark.count()
            feature_cols = [col for col in train_spark_resampled.columns if col not in ['is_fraud']]
            num_features = len(feature_cols)
            
            mlflow.log_metrics({
                'train_samples': num_train_samples,
                'test_samples': num_test_samples,
                'num_features': num_features
            })
            
            # Mock structure to ensure metrics logging code below executes without modifications
            class SparkMetadataMock:
                def __init__(self, length, shape):
                    self.length = length
                    self.shape = shape
                    self.columns = feature_cols
                def __len__(self):
                    return self.length
            
            X_train = SparkMetadataMock(num_train_samples, (num_train_samples, num_features))
            X_test = SparkMetadataMock(num_test_samples, (num_test_samples, num_features))
            y_train_s = None
            y_test_s = None
            
            # Model building and training with timing
            logger.info(f"\n{'='*80}")
            logger.info(f"MODEL TRAINING STEP (PYSPARK)")
            logger.info(f"{'='*80}")
            import time
            training_start_time = time.time()
            
            logger.info(f"Building PySpark MLlib model pipeline: '{spark_model_type}'")
            spark_params = model_cfg.get('spark_model_params', {})
            builder = get_model_builder(spark_model_type, framework="pyspark", feature_cols=feature_cols, label_col="is_fraud", **spark_params)
            model = builder.build_model()
            
            trainer = create_trainer_from_config()
            fitted_model, metrics = trainer.train(model, train_spark_resampled)
            
            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            logger.info(f"✓ Model training completed in {training_time:.2f} seconds")
            
            # Save Spark PipelineModel
            trainer.save_model(fitted_model, model_path)
            logger.info(f"✓ Model saved to: {model_path}")
            try:
                mlflow.spark.log_model(fitted_model, "spark_model")
                logger.info("✓ Logged Spark model to MLflow")
            except Exception as me:
                logger.warning(f"Failed to log Spark model to MLflow: {me}")
            
            # Tune threshold on validation set
            logger.info("Tuning decision threshold on validation set...")
            val_predictions = fitted_model.transform(val_spark_df)
            val_probs_pdf = spark_to_pandas(val_predictions.select("is_fraud", "probability"))
            val_probs = np.array([float(p[1]) for p in val_probs_pdf["probability"]])
            y_val = val_probs_pdf["is_fraud"].values
            
            # Prepare test predictions
            X_test_indexed = X_test_spark.withColumn("row_id", row_number().over(w))
            Y_test_indexed = Y_test_spark.withColumn("row_id", row_number().over(w))
            test_spark_full = X_test_indexed.join(Y_test_indexed, "row_id").drop("row_id")
            
            logger.info("Evaluating model performance on test set using optimized threshold...")
            test_predictions = fitted_model.transform(test_spark_full)
            test_probs_pdf = spark_to_pandas(test_predictions.select("is_fraud", "probability"))
            test_probs = np.array([float(p[1]) for p in test_probs_pdf["probability"]])
            y_test_s = test_probs_pdf["is_fraud"].values
            
            model = fitted_model
        else:
            # ############### PANDAS / SKLEARN CODES (Active) ###########################
            logger.info(f"Loading training and test datasets (format: {data_format}) using Pandas...")
            if data_format == 'parquet':
                X_train = pd.read_parquet(f"{processed_dir}/X_train.parquet")
                Y_train = pd.read_parquet(f"{processed_dir}/Y_train.parquet")
                X_test = pd.read_parquet(f"{processed_dir}/X_test.parquet")
                Y_test = pd.read_parquet(f"{processed_dir}/Y_test.parquet")
            else:
                X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
                Y_train = pd.read_csv(f"{processed_dir}/Y_train.csv")
                X_test = pd.read_csv(f"{processed_dir}/X_test.csv")
                Y_test = pd.read_csv(f"{processed_dir}/Y_test.csv")
            
            # Ensure target column is not in feature sets
            target_col = 'is_fraud'
            if target_col in X_train.columns:
                X_train = X_train.drop(columns=[target_col])
            if target_col in X_test.columns:
                X_test = X_test.drop(columns=[target_col])
                
            y_train_s = Y_train.squeeze()
            y_test_s = Y_test.squeeze()
            
            logger.info(f"✓ Converted to pandas - Training: {X_train.shape}, Test: {X_test.shape}")
            
            # Log dataset information
            mlflow.log_metrics({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': X_train.shape[1],
                'train_class_0': int((y_train_s == 0).sum()),
                'train_class_1': int((y_train_s == 1).sum()),
                'test_class_0': int((y_test_s == 0).sum()),
                'test_class_1': int((y_test_s == 1).sum())
            })
            
            # Log feature names
            mlflow.log_param('feature_names', list(X_train.columns))

            # Model building and training with timing
            logger.info(f"\n{'='*80}")
            logger.info(f"MODEL TRAINING STEP")
            logger.info(f"{'='*80}")
            logger.info("Splitting train data and applying resampling...")
            import time
            training_start_time = time.time()
            
            # Split train data into sub-train and validation sets for threshold tuning
            from sklearn.model_selection import train_test_split
            logger.info("Carving out 15% validation set from train split for threshold optimization...")
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train, y_train_s, test_size=0.15, random_state=42, stratify=y_train_s
            )
            
            # Apply RandomUnderSampler to majority class to improve precision/recall balance and training speed
            from imblearn.under_sampling import RandomUnderSampler
            logger.info("Applying RandomUnderSampler to handle class imbalance (target ratio 1:20)...")
            rus = RandomUnderSampler(sampling_strategy=0.05, random_state=42)
            X_train_res, y_train_res = rus.fit_resample(X_train_sub, y_train_sub)
            logger.info(f"✓ Resampled training set size: {X_train_res.shape[0]} rows (Legitimate: {(y_train_res == 0).sum()}, Fraud: {(y_train_res == 1).sum()})")
            
            # Perform hyperparameter search
            from sklearn.model_selection import RandomizedSearchCV
            from xgboost import XGBClassifier
            
            logger.info("Optimizing hyperparameters via RandomizedSearchCV...")
            param_dist = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_child_weight': [1, 5, 10],
                'gamma': [0.0, 0.1, 0.2],
                'scale_pos_weight': [5.0, 10.0, 20.0]
            }
            
            base_xgb = XGBClassifier(
                n_estimators=100,
                eval_metric='logloss',
                random_state=42
            )
            
            search = RandomizedSearchCV(
                estimator=base_xgb,
                param_distributions=param_dist,
                n_iter=5,
                scoring='f1',
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train_res, y_train_res)
            best_params = search.best_params_
            logger.info(f"✓ Best hyperparameters found: {best_params}")
            
            # Log best parameters to MLflow
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            
            # Build final model with best hyperparameters
            model = XGBClassifier(
                n_estimators=100,
                eval_metric='logloss',
                random_state=42,
                **best_params
            )
            model.fit(X_train_res, y_train_res)
            
            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            logger.info(f"✓ Model training completed in {training_time:.2f} seconds")
            
            # Save model
            trainer = create_trainer_from_config()
            trainer.save_model(model, model_path)
            logger.info(f"✓ Model saved to: {model_path}")
            
            # Log model to MLflow artifacts
            mlflow.log_artifact(model_path, "trained_models")
            
            # Tune threshold on validation set targeting precision >= 70%
            logger.info("Tuning decision threshold on validation set...")
            val_probs = model.predict_proba(X_val)[:, 1]
            
            test_probs = model.predict_proba(X_test)[:, 1]
            
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_probs)
        
        # Sweep thresholds and select the one that meets target precision of 70%+
        target_precision = 0.70
        best_threshold = 0.5
        best_recall = -1.0
        
        valid_indices = np.where(precisions >= target_precision)[0]
        if len(valid_indices) > 0:
            best_idx = -1
            for idx in valid_indices:
                if idx < len(pr_thresholds):
                    if recalls[idx] > best_recall:
                        best_recall = recalls[idx]
                        best_idx = idx
            if best_idx != -1:
                best_threshold = pr_thresholds[best_idx]
                logger.info(f"✓ Found optimal threshold meeting target precision >= {target_precision:.2f}: {best_threshold:.4f} (Validation Recall: {best_recall:.4%})")
            else:
                # Fallback to maximizing F1
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                best_f1_idx = np.argmax(f1_scores)
                if best_f1_idx < len(pr_thresholds):
                    best_threshold = pr_thresholds[best_f1_idx]
                logger.warning(f"⚠ Fallback: using threshold maximizing F1 score: {best_threshold:.4f}")
        else:
            # Fallback to maximizing F1
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_f1_idx = np.argmax(f1_scores)
            if best_f1_idx < len(pr_thresholds):
                best_threshold = pr_thresholds[best_f1_idx]
            logger.warning(f"⚠ Could not find threshold meeting target precision >= {target_precision:.2f}. Fallback to threshold maximizing F1: {best_threshold:.4f}")
            
        review_threshold = best_threshold * 0.4
        logger.info(f"✓ Decision threshold: {best_threshold:.4f}, Review threshold: {review_threshold:.4f}")
        
        # Save model threshold metadata
        metadata = {
            'decision_threshold': float(best_threshold),
            'review_threshold': float(review_threshold),
            'target_precision': target_precision,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        if model_path.endswith('.pkl'):
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        else:
            metadata_path = model_path + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved model threshold metadata to: {metadata_path}")
        mlflow.log_artifact(metadata_path, "model_metadata")
        
        # Plot curves and Confusion Matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot 1: Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, color='blue', lw=2, label='Precision-Recall Curve')
        plt.axvline(best_recall if best_recall > 0 else recalls[best_f1_idx], color='red', linestyle='--', label=f'Chosen Point (Recall: {best_recall if best_recall > 0 else recalls[best_f1_idx]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        pr_curve_path = os.path.join(run_artifacts_dir, 'precision_recall_curve.png')
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log PR Curve to MLflow
        mlflow.log_artifact(pr_curve_path, "evaluation")
        
        # Model evaluation on test set using the tuned threshold
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL EVALUATION STEP")
        logger.info(f"{'='*80}")
        logger.info("Evaluating model performance on test set using optimized threshold...")
        
        if framework != 'pyspark':
            test_probs = model.predict_proba(X_test)[:, 1]
        y_pred_test = (test_probs >= best_threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        acc = accuracy_score(y_test_s, y_pred_test)
        prec = precision_score(y_test_s, y_pred_test, zero_division=0)
        rec = recall_score(y_test_s, y_pred_test, zero_division=0)
        f1 = f1_score(y_test_s, y_pred_test, zero_division=0)
        cm = confusion_matrix(y_test_s, y_pred_test)
        roc_auc = roc_auc_score(y_test_s, test_probs)
        
        evaluation_results = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'cm': cm
        }
        evaluation_results_cp = evaluation_results.copy()
        
        # Plot 2: Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Threshold: {best_threshold:.2f})')
        cm_path = os.path.join(run_artifacts_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log Confusion Matrix to MLflow
        mlflow.log_artifact(cm_path, "evaluation")
        
        # Also copy Confusion Matrix to project-level evaluation artifacts dir
        proj_eval_dir = os.path.join(PROJECT_ROOT, 'artifacts/evaluation')
        os.makedirs(proj_eval_dir, exist_ok=True)
        proj_cm_path = os.path.join(proj_eval_dir, 'confusion_matrix.png')
        import shutil
        shutil.copy(cm_path, proj_cm_path)
        logger.info(f"✓ Saved project-level confusion matrix to: {proj_cm_path}")
        
        # Log thresholds as model parameters
        mlflow.log_params({
            'decision_threshold': float(best_threshold),
            'review_threshold': float(review_threshold),
            'target_precision': target_precision
        })
        
        # Log training metrics (remove confusion matrix for MLflow logging)
        if 'cm' in evaluation_results_cp:
            del evaluation_results_cp['cm']
        
        # Add additional training metrics
        evaluation_results_cp.update({
            'training_time_seconds': training_time,
            'model_complexity': model.n_estimators if hasattr(model, 'n_estimators') else 0,
            'max_depth': model.max_depth if hasattr(model, 'max_depth') else 0
        })
        
        # Get model config for logging
        model_config = model_cfg.get('model_params', {})
        mlflow_tracker.log_training_metrics(model, evaluation_results_cp, model_config)
        
        # Log training summary
        training_summary = {
            'model_type': 'XGboost',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': X_train.shape[1],
            'training_time': training_time,
            'model_path': model_path,
            'performance_metrics': evaluation_results_cp,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save training summary
        summary_path = os.path.join(run_artifacts_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        mlflow.log_artifact(summary_path, "training_summary")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info("✓ Training pipeline completed successfully!")
        logger.info(f"  • Model Performance - Accuracy: {evaluation_results.get('accuracy', 0.0):.4%}")
        logger.info(f"  • Model Performance - Precision: {evaluation_results.get('precision', 0.0):.4%}")
        logger.info(f"  • Model Performance - Recall: {evaluation_results.get('recall', 0.0):.4%}")
        logger.info(f"  • Model Performance - F1 Score: {evaluation_results.get('f1', 0.0):.4%}")
        logger.info(f"  • Training Time: {training_time:.2f} seconds")
        logger.info(f"  • Model saved to: {model_path}")
        logger.info(f"  • Training samples: {len(X_train)}")
        logger.info(f"  • Test samples: {len(X_test)}")
        logger.info(f"  • Features used: {X_train.shape[1]}")
        
        mlflow_tracker.end_run()
        
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {str(e)}")
        if 'mlflow_tracker' in locals() and mlflow.active_run():
            try:
                mlflow_tracker.end_run()
            except Exception:
                pass
        raise
    finally:
        # Stop Spark session
        stop_spark_session(spark)


if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params', {})
    training_pipeline(model_params=model_params, data_format='parquet')