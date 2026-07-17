import os
import sys
import logging
import json
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

# Resolve relative paths against project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils.spark_session import create_spark_session, stop_spark_session
from utils.spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info, check_missing_values
from src.data_ingestion import DataIngestorCSV
from src.handling_missing_values import create_default_missing_value_pipeline
from src.handling_outliers import create_outlier_pipeline
from src.feature_binning import create_default_binning_pipeline
from src.feature_encoding import create_default_encoding_pipeline
from src.feature_scaling import create_default_scaling_pipeline
from src.data_splitter import SimpleTrainTestSplitStrategy

from utils.config import (
    get_data_paths,
    get_columns,
    get_missing_values_config,
    get_outlier_config,
    get_binning_config,
    get_encoding_config,
    get_scaling_config,
    get_splitting_config,
)
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
import mlflow


def log_stage_metrics(df: DataFrame, stage: str, additional_metrics: Dict = None, spark: SparkSession = None):
    """Log key metrics for each processing stage."""
    try:
        # Calculate missing values count efficiently
        missing_counts = []
        for col in df.columns:
            missing_counts.append(df.filter(F.col(col).isNull()).count())
        total_missing = sum(missing_counts)
        
        metrics = {
            f'{stage}_rows': df.count(),
            f'{stage}_columns': len(df.columns),
            f'{stage}_missing_values': total_missing,
            f'{stage}_partitions': df.rdd.getNumPartitions()
        }
        
        if additional_metrics:
            metrics.update({f'{stage}_{k}': v for k, v in additional_metrics.items()})
        
        mlflow.log_metrics(metrics)
        logger.info(f"✓ Metrics logged for {stage}: ({metrics[f'{stage}_rows']}, {metrics[f'{stage}_columns']})")
        
    except Exception as e:
        logger.error(f"✗ Failed to log metrics for {stage}: {str(e)}")


def save_processed_data(
    X_train_pd: pd.DataFrame, 
    X_test_pd: pd.DataFrame, 
    Y_train_pd: pd.DataFrame, 
    Y_test_pd: pd.DataFrame,
    output_format: str = "both"
) -> Dict[str, str]:
    """
    Save processed data in specified format(s) including NPZ compatibility format.
    
    Args:
        X_train_pd, X_test_pd, Y_train_pd, Y_test_pd: pandas DataFrames
        output_format: "csv", "parquet", or "both"
        
    Returns:
        Dictionary of output paths
    """
    data_dir = os.path.join(PROJECT_ROOT, 'artifacts/data')
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    
    # Retrieve configurations for NPZ output paths
    data_paths = get_data_paths()
    x_train_npz = os.path.join(PROJECT_ROOT, data_paths.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
    x_test_npz  = os.path.join(PROJECT_ROOT, data_paths.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
    y_train_npz = os.path.join(PROJECT_ROOT, data_paths.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
    y_test_npz  = os.path.join(PROJECT_ROOT, data_paths.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
    features_json_path = os.path.join(PROJECT_ROOT, data_paths.get('data_artifacts_dir', 'artifacts/data'), 'features.json')
    
    # Save NPZ files for scikit-learn model training compatibility
    logger.info("Saving compatibility NPZ files...")
    np.savez(x_train_npz, X_train=X_train_pd.values.astype(float))
    np.savez(x_test_npz, X_test=X_test_pd.values.astype(float))
    np.savez(y_train_npz, y_train=Y_train_pd.values.astype(int).ravel())
    np.savez(y_test_npz, y_test=Y_test_pd.values.astype(int).ravel())
    
    # Save features list
    with open(features_json_path, 'w') as f:
        json.dump(list(X_train_pd.columns), f)
        
    paths['X_train_npz'] = x_train_npz
    paths['X_test_npz'] = x_test_npz
    paths['Y_train_npz'] = y_train_npz
    paths['Y_test_npz'] = y_test_npz
    paths['features_json'] = features_json_path
    
    if output_format in ["csv", "both"]:
        # Save as CSV
        logger.info("Saving data in CSV format...")
        paths['X_train_csv'] = os.path.join(data_dir, 'X_train.csv')
        paths['X_test_csv'] = os.path.join(data_dir, 'X_test.csv')
        paths['Y_train_csv'] = os.path.join(data_dir, 'Y_train.csv')
        paths['Y_test_csv'] = os.path.join(data_dir, 'Y_test.csv')
        
        X_train_pd.to_csv(paths['X_train_csv'], index=False)
        X_test_pd.to_csv(paths['X_test_csv'], index=False)
        Y_train_pd.to_csv(paths['Y_train_csv'], index=False)
        Y_test_pd.to_csv(paths['Y_test_csv'], index=False)
        logger.info("✓ CSV files saved")
    
    if output_format in ["parquet", "both"]:
        # Save as Parquet
        logger.info("Saving data in Parquet format...")
        paths['X_train_parquet'] = os.path.join(data_dir, 'X_train.parquet')
        paths['X_test_parquet'] = os.path.join(data_dir, 'X_test.parquet')
        paths['Y_train_parquet'] = os.path.join(data_dir, 'Y_train.parquet')
        paths['Y_test_parquet'] = os.path.join(data_dir, 'Y_test.parquet')
        
        X_train_pd.to_parquet(paths['X_train_parquet'], index=False)
        X_test_pd.to_parquet(paths['X_test_parquet'], index=False)
        Y_train_pd.to_parquet(paths['Y_train_parquet'], index=False)
        Y_test_pd.to_parquet(paths['Y_test_parquet'], index=False)
        logger.info("✓ Parquet files saved")
    
    return paths


def data_pipeline(
    data_path: Optional[str] = None,
    target_column: str = 'is_fraud',
    test_size: float = 0.2,
    force_rebuild: bool = False,
    output_format: str = "both"
) -> Dict[str, np.ndarray]:
    """
    Execute comprehensive credit card fraud data processing pipeline with PySpark and MLflow tracking.
    
    Args:
        data_path: Path to the raw data file (resolved from config if None)
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        force_rebuild: Whether to force rebuild of existing artifacts
        output_format: Output format - "csv", "parquet", or "both"
        
    Returns:
        Dictionary containing processed train/test splits as numpy arrays
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING PYSPARK DATA PIPELINE")
    logger.info(f"{'='*80}")
    
    data_paths = get_data_paths()
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
        
    # Input validation
    if not os.path.exists(data_path):
        logger.error(f"✗ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0 < test_size < 1:
        logger.error(f"✗ Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")
    
    # Initialize Spark session
    spark = create_spark_session("CreditCardFraudDetectionDataPipeline")
    
    try:
        # Load configurations
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        
        # Initialize MLflow tracking
        mlflow_tracker = None
        try:
            mlflow_tracker = MLflowTracker()
        except Exception as e:
            logger.warning(f"MLflow not running or configured: {e}")
            
        run = None
        if mlflow_tracker:
            run_tags = create_mlflow_run_tags('data_pipeline_pyspark', {
                'data_source': data_path,
                'force_rebuild': str(force_rebuild),
                'target_column': target_column,
                'output_format': output_format,
                'processing_engine': 'pyspark'
            })
            run = mlflow_tracker.start_run(run_name='data_pipeline_pyspark', tags=run_tags)
        
        # Check for existing artifacts (NPZ format for compatibility)
        x_train_npz = os.path.join(PROJECT_ROOT, data_paths.get('X_train', 'artifacts/data/credit_card_fraud_X_train.npz'))
        x_test_npz  = os.path.join(PROJECT_ROOT, data_paths.get('X_test', 'artifacts/data/credit_card_fraud_X_test.npz'))
        y_train_npz = os.path.join(PROJECT_ROOT, data_paths.get('Y_train', 'artifacts/data/credit_card_fraud_y_train.npz'))
        y_test_npz  = os.path.join(PROJECT_ROOT, data_paths.get('Y_test', 'artifacts/data/credit_card_fraud_y_test.npz'))
        features_json_path = os.path.join(PROJECT_ROOT, data_paths.get('data_artifacts_dir', 'artifacts/data'), 'features.json')
        
        artifacts_exist = all(os.path.exists(p) for p in [x_train_npz, x_test_npz, y_train_npz, y_test_npz, features_json_path])
        
        if artifacts_exist and not force_rebuild:
            logger.info("✓ Loading existing processed data artifacts (NPZ)")
            X_train = np.load(x_train_npz)['X_train']
            X_test  = np.load(x_test_npz)['X_test']
            Y_train = np.load(y_train_npz)['y_train']
            Y_test  = np.load(y_test_npz)['y_test']
            
            if mlflow_tracker:
                mlflow_tracker.log_data_pipeline_metrics({
                    'total_samples': len(X_train) + len(X_test),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'processing_engine': 'existing_artifacts'
                })
                mlflow_tracker.end_run()
            
            logger.info("✓ Data pipeline completed using existing artifacts")
            return {
                'X_train': X_train,
                'X_test': X_test,
                'Y_train': Y_train,
                'Y_test': Y_test
            }
        
        # Process data from scratch with PySpark
        logger.info("Processing data from scratch with PySpark...")
        
        # Data ingestion
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA INGESTION STEP")
        logger.info(f"{'='*80}")
        ingestor = DataIngestorCSV(spark)
        df = ingestor.ingest(data_path)
        logger.info("✓ Raw data loaded.")
        
        # Log raw data metrics
        if mlflow_tracker:
            log_stage_metrics(df, 'raw', spark=spark)
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Handle missing values & feature engineering
        logger.info(f"\n{'='*80}")
        logger.info(f"HANDLING MISSING VALUES & FEATURE ENGINEERING STEP")
        logger.info(f"{'='*80}")
        initial_count = df.count()
        
        # Run preconfigured default missing values and feature engineering pipeline
        mv_pipeline = create_default_missing_value_pipeline(spark=spark)
        df = mv_pipeline.execute(df)
        
        if mlflow_tracker:
            log_stage_metrics(df, 'missing_handled', spark=spark)
        logger.info(f"✓ Missing values/feature engineering step completed: {initial_count} → {df.count()}")
        
        # Outlier handling
        logger.info(f"\n{'='*80}")
        logger.info(f"OUTLIER HANDLING STEP")
        logger.info(f"{'='*80}")
        outlier_pipeline = create_outlier_pipeline(spark=spark)
        df = outlier_pipeline.execute(df)
        logger.info("✓ Outliers handled")
        
        # Feature binning
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE BINNING STEP")
        logger.info(f"{'='*80}")
        binning_pipeline = create_default_binning_pipeline(spark=spark)
        df = binning_pipeline.execute(df)
        logger.info("✓ Feature binning completed")
        
        # Feature encoding
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE ENCODING STEP")
        logger.info(f"{'='*80}")
        encoding_pipeline = create_default_encoding_pipeline(spark=spark)
        df = encoding_pipeline.execute(df)
        
        if mlflow_tracker:
            log_stage_metrics(df, 'encoded', spark=spark)
        logger.info("✓ Feature encoding completed")
        
        # Feature scaling
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE SCALING STEP")
        logger.info(f"{'='*80}")
        scaling_pipeline = create_default_scaling_pipeline(spark=spark)
        df = scaling_pipeline.execute(df)
        logger.info("✓ Feature scaling completed")
        
        # Save scaler artifacts for inference
        logger.info("Saving scaler artifacts for inference...")
        scaler_save_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'scale')
        scaler_saved = scaling_pipeline.strategy.save_scaler(
            columns_to_scale=scaling_pipeline.columns_to_scale,
            save_dir=scaler_save_dir
        )
        if scaler_saved:
            logger.info("✓ Scaler artifacts saved successfully")
        else:
            logger.warning("⚠ Failed to save scaler artifacts")
        
        # Post-processing - drop unnecessary columns
        # In Credit Card Fraud, drop_columns is defined in configurations (e.g. ['transaction_id'])
        drop_columns = columns.get('drop_columns', ['transaction_id'])
        # Also drop outlier indicator columns if any
        outlier_columns = [col for col in df.columns if col.endswith('_outlier')]
        drop_columns.extend(outlier_columns)
        
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        if existing_drop_columns:
            df = df.drop(*existing_drop_columns)
            logger.info(f"✓ Dropped columns: {existing_drop_columns}")
        
        # Data splitting
        logger.info(f"\n{'='*80}")
        logger.info(f"DATA SPLITTING STEP")
        logger.info(f"{'='*80}")
        
        # Convert preprocessed PySpark DataFrame to Pandas to maintain perfect row alignment
        logger.info("Converting preprocessed PySpark DataFrame to Pandas to avoid row-mismatch during splitting...")
        df_pd = spark_to_pandas(df)
        
        # Perform stratified split in Pandas
        from sklearn.model_selection import train_test_split
        logger.info(f"Performing stratified train/test split with test_size={test_size}...")
        train_pd, test_pd = train_test_split(
            df_pd, test_size=test_size, random_state=42, 
            stratify=df_pd[target_column]
        )
        
        X_train_pd = train_pd.drop(columns=[target_column])
        Y_train_pd = train_pd[[target_column]]
        X_test_pd = test_pd.drop(columns=[target_column])
        Y_test_pd = test_pd[[target_column]]
        
        # Save processed data
        output_paths = save_processed_data(X_train_pd, X_test_pd, Y_train_pd, Y_test_pd, output_format)
        
        logger.info("✓ Data splitting completed")
        logger.info(f"\nDataset shapes after splitting:")
        logger.info(f"  • X_train: {len(X_train_pd)} rows, {X_train_pd.shape[1]} columns")
        logger.info(f"  • X_test:  {len(X_test_pd)} rows, {X_test_pd.shape[1]} columns")
        logger.info(f"  • Y_train: {len(Y_train_pd)} rows, 1 column")
        logger.info(f"  • Y_test:  {len(Y_test_pd)} rows, 1 column")
        logger.info(f"  • Feature columns: {list(X_train_pd.columns)}")
        
        # Save preprocessing pipeline model
        model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'encode', 'fitted_preprocessing_model')
        os.makedirs(model_path, exist_ok=True)
        
        # Save metadata about the preprocessing
        preprocessing_metadata = {
            'scaling_columns': scaling_config.get('columns_to_scale', [
                'amount_log', 
                'velocity_last_24h_log', 
                'city_population_log'
            ]),
            'encoding_columns': encoding_config.get('nominal_columns', ["merchant_category", "gender"]),
            'ordinal_mappings': encoding_config.get('ordinal_mappings', {}),
            'binning_config': binning_config,
            'spark_version': spark.version
        }
        
        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(preprocessing_metadata, f, indent=2)
        logger.info(f"✓ Saved preprocessing metadata to {model_path}")
        
        # Log final metrics to MLflow
        if mlflow_tracker:
            total_missing_train = int(X_train_pd.isna().sum().sum())
            total_missing_test = int(X_test_pd.isna().sum().sum())
            
            mlflow.log_metrics({
                'final_train_rows': len(X_train_pd),
                'final_train_columns': X_train_pd.shape[1],
                'final_train_missing_values': total_missing_train,
                'final_test_rows': len(X_test_pd),
                'final_test_columns': X_test_pd.shape[1],
                'final_test_missing_values': total_missing_test,
            })
            
            # Log comprehensive pipeline metrics
            comprehensive_metrics = {
                'total_samples': len(X_train_pd) + len(X_test_pd),
                'train_samples': len(X_train_pd),
                'test_samples': len(X_test_pd),
                'final_features': X_train_pd.shape[1],
                'processing_engine': 'pyspark',
                'output_format': output_format
            }
            
            # Get class distribution
            train_dist = Y_train_pd[target_column].value_counts()
            test_dist = Y_test_pd[target_column].value_counts()
            
            for val, count in train_dist.items():
                comprehensive_metrics[f'train_class_{val}'] = int(count)
            for val, count in test_dist.items():
                comprehensive_metrics[f'test_class_{val}'] = int(count)
            
            mlflow_tracker.log_data_pipeline_metrics(comprehensive_metrics)
            
            # Log parameters
            mlflow.log_params({
                'final_feature_names': list(X_train_pd.columns),
                'preprocessing_steps': ['missing_values', 'outlier_detection', 'feature_binning', 
                                      'feature_encoding', 'feature_scaling'],
                'data_pipeline_version': '3.0_pyspark'
            })
            
            # Log artifacts
            for path_key, path_value in output_paths.items():
                if os.path.exists(path_value):
                    mlflow.log_artifact(path_value, "processed_datasets")
            
            mlflow_tracker.end_run()
        
        # Convert to numpy arrays for return
        X_train_np = X_train_pd.values
        X_test_np = X_test_pd.values
        Y_train_np = Y_train_pd.values.ravel()
        Y_test_np = Y_test_pd.values.ravel()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL DATASET SHAPES")
        logger.info(f"{'='*80}")
        logger.info(f"✓ Final dataset shapes:")
        logger.info(f"  • X_train shape: {X_train_np.shape} (rows: {X_train_np.shape[0]}, features: {X_train_np.shape[1]})")
        logger.info(f"  • X_test shape:  {X_test_np.shape} (rows: {X_test_np.shape[0]}, features: {X_test_np.shape[1]})")
        logger.info(f"  • Y_train shape: {Y_train_np.shape} (rows: {Y_train_np.shape[0]})")
        logger.info(f"  • Y_test shape:  {Y_test_np.shape} (rows: {Y_test_np.shape[0]})")
        logger.info(f"  • Total samples: {X_train_np.shape[0] + X_test_np.shape[0]}")
        logger.info(f"  • Train/Test ratio: {X_train_np.shape[0]/(X_train_np.shape[0] + X_test_np.shape[0]):.1%} / {X_test_np.shape[0]/(X_train_np.shape[0] + X_test_np.shape[0]):.1%}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info("✓ PySpark data pipeline completed successfully!")
        
        return {
            'X_train': X_train_np,
            'X_test': X_test_np,
            'Y_train': Y_train_np,
            'Y_test': Y_test_np
        }
            
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        if mlflow_tracker and mlflow.active_run():
            try:
                mlflow_tracker.end_run()
            except Exception:
                pass
        raise
    finally:
        # Stop Spark session
        stop_spark_session(spark)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection PySpark Data Pipeline")
    parser.add_argument('--force', action='store_true', help='Force rebuild data artifacts')
    args = parser.parse_args()
    
    # Load configuration to get default data path
    data_paths = get_data_paths()
    default_data_path = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
    
    processed_data = data_pipeline(
        data_path=default_data_path,
        target_column='is_fraud',
        force_rebuild=args.force,
        output_format="both"
    )
    logger.info(f"Pipeline completed. Train samples: {processed_data['X_train'].shape[0]}")