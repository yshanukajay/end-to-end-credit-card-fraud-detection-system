import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict

# ---------------------------------------------------------------------------
# Project root resolution — ensures imports work from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging — use the project-wide logger utility
# ---------------------------------------------------------------------------
from utils.logger import get_logger
logger = get_logger(__name__)

# Try to import mlflow to track experiments
try:
    import mlflow
    from utils.mlflow_utils import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠ mlflow package not found. MLflow tracking will be skipped.")

# ---------------------------------------------------------------------------
# src imports — wired to actual module filenames on disk
# ---------------------------------------------------------------------------
from src.data_ingestion import DataIngestorFactory

from src.handling_missing_values import (
    MissingValuePipeline,
    DropDuplicatesStrategy,
    MedianImputationStrategy,
    ModeImputationStrategy,
    create_default_missing_value_pipeline,
)

from src.handling_outliers import (
    OutlierHandlingPipeline,
    IQRClipStrategy,
    create_outlier_pipeline,
)

from src.feature_binning import (
    FeatureBinningPipeline,
    CustomBinningStrategy,
    create_default_binning_pipeline,
)

from src.feature_encoding import (
    FeatureEncodingPipeline,
    NominalEncodingStrategy,
    OrdinalEncodingStrategy,
    create_default_encoding_pipeline,
)

from src.feature_scaling import (
    FeatureScalingPipeline,
    StandardScalingStrategy,
    create_default_scaling_pipeline,
)

from src.data_splitter import (
    SplitAndResampleStrategy,
    StratifiedTrainTestSplitStrategy,
    SMOTENCOversampler,
    create_default_resampled_splitter,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

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
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

            for i in range(n_plots, 4):
                axes[i].set_visible(False)

            plt.suptitle(f'Distributions — {stage}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'distributions_{stage}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # --- correlation heatmap ---
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=0.5)
            plt.title(f'Feature Correlation — {stage}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'correlation_{stage}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        logger.info(f"✓ Visualizations saved for stage: {stage}")

    except Exception as e:
        logger.warning(f"⚠ Could not create visualizations for '{stage}': {e}")


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def data_pipeline(
    data_path: str = None,
    target_column: str = None,
    test_size: float = None,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Execute the end-to-end data processing pipeline for credit card fraud detection.

    Steps
    -----
    1. Data ingestion (CSV)
    2. Missing value handling  (drop duplicates → median impute continuous → mode impute discrete)
    3. Outlier handling        (IQR capping)
    4. Feature binning         (device_trust_score → Poor / Fair / Good / Excellent)
    5. Feature encoding        (one-hot: merchant_category | ordinal: device_trust_score_binned)
    6. Feature scaling         (StandardScaler on continuous columns)
    7. Drop non-feature columns (transaction_id)
    8. Stratified train/test split + SMOTENC oversampling on training set

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test  (numpy arrays)
    """
    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    config = load_config()

    data_paths_cfg   = config.get('data_paths', {})
    columns_cfg      = config.get('columns', {})
    outlier_cfg      = config.get('outlier_detection', {})
    binning_cfg      = config.get('feature_binning', {})
    encoding_cfg     = config.get('feature_encoding', {})
    scaling_cfg      = config.get('feature_scaling', {})
    splitting_cfg    = config.get('data_splitting', {})

    # Allow callers to override config defaults via arguments
    if data_path is None:
        data_path = data_paths_cfg.get('raw_data', 'dataset/raw/credit_card_fraud_10k.csv')
    if target_column is None:
        target_column = columns_cfg.get('target', 'is_fraud')
    if test_size is None:
        test_size = splitting_cfg.get('test_size', 0.2)

    # Resolve relative paths against project root
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    logger.info(f"\n{'='*80}")
    logger.info("STARTING DATA PIPELINE — Credit Card Fraud Detection")
    logger.info(f"{'='*80}")
    logger.info(f"  Data path     : {data_path}")
    logger.info(f"  Target column : {target_column}")
    logger.info(f"  Test size     : {test_size}")
    logger.info(f"  Force rebuild : {force_rebuild}")

    # -----------------------------------------------------------------------
    # Path validation
    # -----------------------------------------------------------------------
    if not os.path.exists(data_path):
        logger.error(f"✗ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got: {test_size}")

    # -----------------------------------------------------------------------
    # Artifact paths
    # -----------------------------------------------------------------------
    artifacts_data_dir = os.path.join(PROJECT_ROOT, data_paths_cfg.get('data_artifacts_dir', 'artifacts/data'))
    os.makedirs(artifacts_data_dir, exist_ok=True)

    x_train_path = os.path.join(PROJECT_ROOT, data_paths_cfg.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
    x_test_path  = os.path.join(PROJECT_ROOT, data_paths_cfg.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
    y_train_path = os.path.join(PROJECT_ROOT, data_paths_cfg.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
    y_test_path  = os.path.join(PROJECT_ROOT, data_paths_cfg.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
    features_json_path = os.path.join(artifacts_data_dir, 'features.json')

    viz_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Fast-path: return existing artifacts if present and rebuild not forced
    # -----------------------------------------------------------------------
    artifacts_exist = all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path, features_json_path])

    if artifacts_exist and not force_rebuild:
        logger.info("✓ Existing processed artifacts found — skipping rebuild.")
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

        logger.info(f"  X_train : {X_train.shape}")
        logger.info(f"  X_test  : {X_test.shape}")
        logger.info(f"  y_train : {y_train.shape}  (fraud rate: {y_train.mean():.3%})")
        logger.info(f"  y_test  : {y_test.shape}   (fraud rate: {y_test.mean():.3%})")
        logger.info(f"{'='*80}\n")

        # Log data pipeline metrics to MLflow
        if MLFLOW_AVAILABLE:
            try:
                tracker = None
                active_run = mlflow.active_run()
                if not active_run:
                    tracker = MLflowTracker()
                    tracker.start_run(run_name="data_pipeline")
                
                dataset_info = {
                    'total_rows': X_train.shape[0] + X_test.shape[0],
                    'train_rows': X_train.shape[0],
                    'test_rows': X_test.shape[0],
                    'num_features': X_train.shape[1],
                    'missing_values': 0,
                    'outliers_removed': 0,
                    'test_size': float(test_size),
                    'random_state': int(splitting_cfg.get('random_state', 42)),
                    'missing_strategy': str(config.get('missing_values', {}).get('strategy', 'fill')),
                    'outlier_method': str(config.get('outlier_detection', {}).get('handling_method', 'cap')),
                    'encoding_applied': True,
                    'scaling_applied': True,
                    'feature_names': list(X_train.columns)
                }
                
                if tracker:
                    tracker.log_data_pipeline_metrics(dataset_info)
                    tracker.end_run()
                else:
                    temp_tracker = MLflowTracker()
                    temp_tracker.log_data_pipeline_metrics(dataset_info)
                logger.info("✓ Data pipeline metrics successfully logged to MLflow.")
            except Exception as e:
                logger.warning(f"Failed to log data pipeline metrics to MLflow: {e}")

        return {
            'X_train': X_train.values,
            'X_test':  X_test.values,
            'y_train': y_train.values.ravel(),
            'y_test':  y_test.values.ravel(),
        }

    # -----------------------------------------------------------------------
    # Step 1 — Data ingestion
    # -----------------------------------------------------------------------
    logger.info("\n[Step 1/8] Data Ingestion")
    ingestor = DataIngestorFactory.get_ingestor(data_path)
    df = ingestor.ingest(data_path)
    logger.info(f"  ✓ Raw data loaded: {df.shape}")

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. "
                         f"Available columns: {list(df.columns)}")

    create_data_visualizations(df, 'raw', viz_dir)

    # -----------------------------------------------------------------------
    # Step 2 — Missing value handling
    # -----------------------------------------------------------------------
    logger.info("\n[Step 2/8] Missing Value Handling")
    mv_pipeline = create_default_missing_value_pipeline()
    df = mv_pipeline.execute(df)
    logger.info(f"  ✓ After missing value handling: {df.shape}")

    # -----------------------------------------------------------------------
    # Step 3 — Outlier handling (IQR capping)
    # -----------------------------------------------------------------------
    logger.info("\n[Step 3/8] Outlier Handling")
    outlier_cols   = columns_cfg.get('outlier_columns', ['amount', 'velocity_last_24h', 'device_trust_score', 'cardholder_age'])
    outlier_method = outlier_cfg.get('handling_method', 'cap')
    outlier_pipeline = create_outlier_pipeline(method=outlier_method, cap_columns=outlier_cols)
    df = outlier_pipeline.execute(df)
    logger.info(f"  ✓ After outlier handling: {df.shape}")

    # -----------------------------------------------------------------------
    # Step 4 — Feature binning (device_trust_score)
    # -----------------------------------------------------------------------
    logger.info("\n[Step 4/8] Feature Binning")
    binning_pipeline = create_default_binning_pipeline()
    df = binning_pipeline.execute(df)
    logger.info(f"  ✓ After feature binning: {df.shape}")

    # -----------------------------------------------------------------------
    # Step 5 — Feature encoding (one-hot + ordinal)
    # -----------------------------------------------------------------------
    logger.info("\n[Step 5/8] Feature Encoding")
    encoding_pipeline = create_default_encoding_pipeline()
    df = encoding_pipeline.execute(df)
    logger.info(f"  ✓ After feature encoding: {df.shape}")

    create_data_visualizations(df, 'encoded', viz_dir)

    # -----------------------------------------------------------------------
    # Step 6 — Feature scaling (StandardScaler on continuous columns)
    # -----------------------------------------------------------------------
    logger.info("\n[Step 6/8] Feature Scaling")
    scaling_pipeline = create_default_scaling_pipeline()
    df = scaling_pipeline.execute(df)
    logger.info(f"  ✓ After feature scaling: {df.shape}")

    # Save scaler for later inference use
    scaler_save_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'scale')
    saved = scaling_pipeline.strategy.save_scaler(
        columns_to_scale=scaling_cfg.get('columns_to_scale', ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']),
        save_dir=scaler_save_dir,
    )
    if saved:
        logger.info(f"  ✓ Scaler artifacts saved to: {scaler_save_dir}")
    else:
        logger.warning("  ⚠ Scaler could not be saved.")

    # -----------------------------------------------------------------------
    # Step 7 — Drop non-feature columns (e.g. transaction_id)
    # -----------------------------------------------------------------------
    logger.info("\n[Step 7/8] Dropping Non-Feature Columns")
    drop_cols = columns_cfg.get('drop_columns', ['transaction_id'])
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        logger.info(f"  ✓ Dropped columns: {existing_drop_cols}")
    else:
        logger.info("  ✓ No columns to drop (already absent or none configured).")

    logger.info(f"  Final feature set ({df.shape[1] - 1} features + target): {[c for c in df.columns if c != target_column]}")

    # -----------------------------------------------------------------------
    # Step 8 — Stratified split + SMOTENC oversampling on training set
    # -----------------------------------------------------------------------
    logger.info("\n[Step 8/8] Stratified Split + SMOTENC Oversampling")
    pipeline = create_default_resampled_splitter()
    X_train, X_test, y_train, y_test = pipeline.split_data(df, target_column)

    logger.info(f"  ✓ X_train : {X_train.shape}  (fraud rate after SMOTENC: {y_train.mean():.3%})")
    logger.info(f"  ✓ X_test  : {X_test.shape}   (fraud rate: {y_test.mean():.3%})")

    # -----------------------------------------------------------------------
    # Persist processed splits to disk
    # -----------------------------------------------------------------------
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

    # Log data pipeline metrics to MLflow
    if MLFLOW_AVAILABLE:
        try:
            tracker = None
            active_run = mlflow.active_run()
            if not active_run:
                tracker = MLflowTracker()
                tracker.start_run(run_name="data_pipeline")
            
            dataset_info = {
                'total_rows': X_train.shape[0] + X_test.shape[0],
                'train_rows': X_train.shape[0],
                'test_rows': X_test.shape[0],
                'num_features': X_train.shape[1],
                'missing_values': 0,
                'outliers_removed': 0,
                'test_size': float(test_size),
                'random_state': int(splitting_cfg.get('random_state', 42)),
                'missing_strategy': str(config.get('missing_values', {}).get('strategy', 'fill')),
                'outlier_method': str(config.get('outlier_detection', {}).get('handling_method', 'cap')),
                'encoding_applied': True,
                'scaling_applied': True,
                'feature_names': list(X_train.columns)
            }
            
            if tracker:
                tracker.log_data_pipeline_metrics(dataset_info)
                tracker.end_run()
            else:
                temp_tracker = MLflowTracker()
                temp_tracker.log_data_pipeline_metrics(dataset_info)
            logger.info("✓ Data pipeline metrics successfully logged to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log data pipeline metrics to MLflow: {e}")

    return {
        'X_train': X_train.values,
        'X_test':  X_test.values,
        'y_train': y_train.values.ravel(),
        'y_test':  y_test.values.ravel(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        force = '--force' in sys.argv or '-f' in sys.argv
        result = data_pipeline(force_rebuild=force)

        print("\n" + "=" * 60)
        print("[DATA PIPELINE EXECUTION SUMMARY]")
        print("=" * 60)
        print(f"[OK] Training samples : {result['X_train'].shape[0]:,}  (after SMOTENC)")
        print(f"[OK] Test samples     : {result['X_test'].shape[0]:,}")
        print(f"[OK] Features         : {result['X_train'].shape[1]}")
        print(f"[OK] Data artifacts   : artifacts/data/")
        print(f"[OK] Scaler artifacts : artifacts/scale/")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        print(f"\n[FAILED] Data pipeline failed: {e}")
        exit(1)