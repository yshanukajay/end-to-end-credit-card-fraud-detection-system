import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion import DataIngestorFactory
from src.handling_missing_values import create_default_missing_value_pipeline
from src.handling_outliers import create_outlier_pipeline
from src.feature_binning import create_default_binning_pipeline
from src.feature_encoding import create_default_encoding_pipeline
from src.feature_scaling import create_default_scaling_pipeline
from src.data_splitter import create_default_resampled_splitter

from utils.config import get_data_paths, get_columns, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
import mlflow


def create_data_visualizations(df: pd.DataFrame, stage: str, artifacts_dir: str):
    """Create essential data visualizations and save them to disk."""
    try:
        stage_dir = os.path.join(artifacts_dir, f"visualizations_{stage}")
        os.makedirs(stage_dir, exist_ok=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # --- distribution plot (top 4 numeric columns) ---
        if numeric_cols:
            n_plots = min(4, len(numeric_cols))
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.flatten()

            for i, col in enumerate(numeric_cols[:n_plots]):
                df[col].hist(bins=30, ax=axes[i], alpha=0.75, color='steelblue')
                axes[i].set_title(f'{col} Distribution ({stage})')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

            # Hide unused slots
            for j in range(n_plots, 4):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'distributions_{stage}.png'))
            plt.close()

        # --- correlation heatmap ---
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlation Matrix - {stage.upper()}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'correlation_matrix_{stage}.png'))
            plt.close()

        logger.info(f"✓ Visualizations saved for stage: {stage}")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def data_pipeline(
    data_path: Optional[str] = None,
    target_column: str = 'is_fraud',
    test_size: float = 0.2,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Execute credit card fraud detection data processing pipeline with MLflow tracking.
    
    Args:
        data_path: Path to the raw data file
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        force_rebuild: Whether to force rebuild of existing artifacts
        
    Returns:
        Dictionary containing processed train/test splits
    """
    # Load configurations
    data_paths = get_data_paths()
    columns_cfg = get_columns()
    outlier_cfg = get_outlier_config()
    binning_cfg = get_binning_config()
    encoding_cfg = get_encoding_config()
    scaling_cfg = get_scaling_config()
    splitting_cfg = get_splitting_config()

    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
        
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING DATA PIPELINE — Credit Card Fraud Detection")
    logger.info(f"{'='*80}")
    logger.info(f"  Data path     : {data_path}")
    logger.info(f"  Target column : {target_column}")
    logger.info(f"  Test size     : {test_size}")
    logger.info(f"  Force rebuild : {force_rebuild}")
    
    # Construct target directories & paths
    artifacts_data_dir = os.path.join(PROJECT_ROOT, data_paths.get('data_artifacts_dir', 'artifacts/data'))
    os.makedirs(artifacts_data_dir, exist_ok=True)
    
    x_train_path = os.path.join(PROJECT_ROOT, data_paths.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
    x_test_path  = os.path.join(PROJECT_ROOT, data_paths.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
    y_train_path = os.path.join(PROJECT_ROOT, data_paths.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
    y_test_path  = os.path.join(PROJECT_ROOT, data_paths.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
    features_json_path = os.path.join(artifacts_data_dir, 'features.json')
    
    artifacts_exist = all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path, features_json_path])
    
    # Try to initialize MLflow tracker
    mlflow_tracker = None
    try:
        mlflow_tracker = MLflowTracker()
    except Exception as e:
        logger.warning(f"MLflow not configured or running: {e}")
        
    if artifacts_exist and not force_rebuild:
        logger.info("✓ Existing processed artifacts found – skipping rebuild.")
        X_train = np.load(x_train_path)['X_train']
        X_test  = np.load(x_test_path)['X_test']
        y_train = np.load(y_train_path)['y_train']
        y_test  = np.load(y_test_path)['y_test']
        
        logger.info(f"  ✓ X_train : {X_train.shape}")
        logger.info(f"  ✓ X_test  : {X_test.shape}")
        logger.info(f"  ✓ y_train : {y_train.shape}  (fraud rate: {y_train.mean():.3%})")
        logger.info(f"  ✓ y_test  : {y_test.shape}   (fraud rate: {y_test.mean():.3%})")
        logger.info(f"{'='*80}\n")
        
        # Log to MLflow if tracking
        if mlflow_tracker and mlflow.active_run():
            try:
                mlflow_tracker.log_data_pipeline_metrics({
                    'total_rows': len(X_train) + len(X_test),
                    'train_rows': len(X_train),
                    'test_rows': len(X_test),
                    'num_features': X_train.shape[1],
                    'test_size': test_size
                })
            except Exception as e:
                logger.warning(f"Failed to log data pipeline metrics to MLflow: {e}")
                
        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': y_train,
            'Y_test': y_test
        }

    # Otherwise build from scratch
    # Start MLflow run
    run = None
    if mlflow_tracker:
        try:
            run_tags = create_mlflow_run_tags('data_pipeline', {
                'data_source': data_path,
                'force_rebuild': str(force_rebuild),
                'target_column': target_column
            })
            run = mlflow_tracker.start_run(run_name='data_pipeline', tags=run_tags)
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            
    # Setup visualization directory
    viz_dir = os.path.join(artifacts_data_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    try:
        # Step 1: Data Ingestion
        logger.info("\n[Step 1/8] Data Ingestion")
        ingestor = DataIngestorFactory.get_ingestor(data_path)
        df = ingestor.ingest(data_path)
        logger.info(f"  ✓ Raw data loaded: {df.shape}")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. "
                             f"Available columns: {list(df.columns)}")
                             
        create_data_visualizations(df, 'raw', viz_dir)

        # Step 2: Missing Value Handling & Feature Engineering
        logger.info("\n[Step 2/8] Missing Value Handling")
        mv_pipeline = create_default_missing_value_pipeline()
        df = mv_pipeline.execute(df)
        logger.info(f"  ✓ After missing value handling: {df.shape}")

        # Step 3: Outlier Handling
        logger.info("\n[Step 3/8] Outlier Handling")
        outlier_cols = columns_cfg.get('outlier_columns', ['customer_age', 'distance_to_merchant'])
        outlier_method = outlier_cfg.get('handling_method', 'cap')
        outlier_pipeline = create_outlier_pipeline(method=outlier_method, cap_columns=outlier_cols)
        df = outlier_pipeline.execute(df)
        logger.info(f"  ✓ After outlier handling: {df.shape}")

        # Step 4: Feature Binning
        logger.info("\n[Step 4/8] Feature Binning")
        binning_pipeline = create_default_binning_pipeline()
        df = binning_pipeline.execute(df)
        logger.info(f"  ✓ After feature binning: {df.shape}")

        # Step 5: Feature Encoding
        logger.info("\n[Step 5/8] Feature Encoding")
        encoding_pipeline = create_default_encoding_pipeline()
        df = encoding_pipeline.execute(df)
        logger.info(f"  ✓ After feature encoding: {df.shape}")

        create_data_visualizations(df, 'encoded', viz_dir)

        # Step 6: Feature Scaling
        logger.info("\n[Step 6/8] Feature Scaling")
        scaling_pipeline = create_default_scaling_pipeline()
        df = scaling_pipeline.execute(df)
        logger.info(f"  ✓ After feature scaling: {df.shape}")

        # Save scaler
        scaler_save_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'scale')
        saved = scaling_pipeline.strategy.save_scaler(
            columns_to_scale=scaling_cfg.get('columns_to_scale', [
                'amount', 'amount_log', 
                'velocity_last_24h', 'velocity_last_24h_log', 
                'city_population', 'city_population_log'
            ]),
            save_dir=scaler_save_dir
        )
        if saved:
            logger.info(f"  ✓ Scaler artifacts saved to: {scaler_save_dir}")
        else:
            logger.warning("  ⚠ Scaler could not be saved.")

        # Step 7: Drop non-feature columns
        logger.info("\n[Step 7/8] Dropping Non-Feature Columns")
        drop_cols = columns_cfg.get('drop_columns', ['transaction_id'])
        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            logger.info(f"  ✓ Dropped columns: {existing_drop_cols}")
        else:
            logger.info("  ✓ No columns to drop (already absent or none configured).")

        logger.info(f"  Final feature set ({df.shape[1] - 1} features + target): {[c for c in df.columns if c != target_column]}")

        # Step 8: Splitting & SMOTENC Oversampling
        logger.info("\n[Step 8/8] Stratified Split + SMOTENC Oversampling")
        pipeline = create_default_resampled_splitter()
        X_train, X_test, y_train, y_test = pipeline.split_data(df, target_column)

        logger.info(f"  ✓ X_train : {X_train.shape}  (fraud rate after SMOTENC: {y_train.mean():.3%})")
        logger.info(f"  ✓ X_test  : {X_test.shape}   (fraud rate: {y_test.mean():.3%})")

        # Save to disk in NPZ format
        logger.info("\nSaving processed splits to disk in NPZ format...")
        np.savez(x_train_path, X_train=X_train.values.astype(float))
        np.savez(x_test_path, X_test=X_test.values.astype(float))
        np.savez(y_train_path, y_train=y_train.values.astype(int).ravel())
        np.savez(y_test_path, y_test=y_test.values.astype(int).ravel())
        
        with open(features_json_path, 'w') as f:
            json.dump(list(X_train.columns), f)
            
        logger.info(f"  ✓ Artifacts saved to: {artifacts_data_dir}")

        create_data_visualizations(pd.concat([X_train, X_test]), 'final', viz_dir)

        logger.info(f"\n{'='*80}")
        logger.info("✓ DATA PIPELINE COMPLETE")
        logger.info(f"{'='*80}\n")

        # Log to MLflow
        if mlflow_tracker and run:
            try:
                mlflow_tracker.log_data_pipeline_metrics({
                    'total_rows': len(df),
                    'train_rows': len(X_train),
                    'test_rows': len(X_test),
                    'num_features': X_train.shape[1],
                    'test_size': test_size
                })
                # Log files as artifacts
                mlflow.log_artifact(x_train_path, "data")
                mlflow.log_artifact(x_test_path, "data")
                mlflow.log_artifact(y_train_path, "data")
                mlflow.log_artifact(y_test_path, "data")
                mlflow.log_artifact(features_json_path, "data")
            except Exception as e:
                logger.warning(f"Failed to log metrics/artifacts to MLflow: {e}")
                
            mlflow_tracker.end_run()

        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'Y_train': y_train.values.ravel(),
            'Y_test': y_test.values.ravel()
        }

    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        if mlflow_tracker:
            try:
                mlflow_tracker.end_run()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Data Pipeline")
    parser.add_argument('--force', action='store_true', help='Force rebuild data artifacts')
    args = parser.parse_args()
    
    try:
        data_pipeline(force_rebuild=args.force)
    except Exception as e:
        logger.error(f"Failed to execute data pipeline: {e}")
        sys.exit(1)