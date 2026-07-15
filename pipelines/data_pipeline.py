import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStratergy
from data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow


def create_data_visualizations(df: pd.DataFrame, stage: str, artifacts_dir: str):
    """Create essential data visualizations for MLflow artifacts."""
    try:
        stage_dir = os.path.join(artifacts_dir, f"visualizations_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 1. Data distribution for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):  # Top 4 numeric columns
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(numeric_cols), 4):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Data Distributions - {stage.title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'distributions_{stage}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap for numeric features
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title(f'Feature Correlation - {stage.title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, f'correlation_{stage}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Log visualizations to MLflow
        for viz_file in os.listdir(stage_dir):
            if viz_file.endswith('.png'):
                mlflow.log_artifact(os.path.join(stage_dir, viz_file), f"visualizations/{stage}")
        
        logger.info(f"✓ Visualizations created for {stage}")
        
    except Exception as e:
        logger.error(f"✗ Failed to create visualizations for {stage}: {str(e)}")


def log_stage_metrics(df: pd.DataFrame, stage: str, additional_metrics: Dict = None):
    """Log key metrics for each processing stage."""
    try:
        metrics = {
            f'{stage}_rows': df.shape[0],
            f'{stage}_columns': df.shape[1],
            f'{stage}_missing_values': df.isnull().sum().sum(),
            f'{stage}_memory_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        if additional_metrics:
            metrics.update({f'{stage}_{k}': v for k, v in additional_metrics.items()})
        
        mlflow.log_metrics(metrics)
        logger.info(f"✓ Metrics logged for {stage}: {df.shape}")
        
    except Exception as e:
        logger.error(f"✗ Failed to log metrics for {stage}: {str(e)}")


def log_csv_artifacts(csv_files: Dict[str, str], artifacts_dir: str):
    """Log final CSV files as MLflow artifacts with metadata."""
    try:
        csv_metadata = {
            'csv_files': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Create CSV artifacts directory
        csv_artifacts_dir = os.path.join(artifacts_dir, 'final_csv_files')
        os.makedirs(csv_artifacts_dir, exist_ok=True)
        
        total_files_logged = 0
        
        for file_name, file_path in csv_files.items():
            if os.path.exists(file_path):
                try:
                    # Get file metadata
                    file_size = os.path.getsize(file_path) / (1024**2)  # MB
                    df = pd.read_csv(file_path)
                    
                    csv_metadata['csv_files'][file_name] = {
                        'file_path': file_path,
                        'file_size_mb': round(file_size, 2),
                        'shape': df.shape,
                        'columns': list(df.columns) if len(df.columns) <= 20 else f"{len(df.columns)} columns",
                        'sample_values': df.head(2).to_dict() if df.shape[0] > 0 else "No data"
                    }
                    
                    # Log the CSV file as artifact
                    mlflow.log_artifact(file_path, "final_datasets")
                    
                    # Log key metrics
                    mlflow.log_metrics({
                        f'final_{file_name}_rows': df.shape[0],
                        f'final_{file_name}_columns': df.shape[1],
                        f'final_{file_name}_size_mb': file_size
                    })
                    
                    total_files_logged += 1
                    logger.info(f"✓ Logged {file_name}: {df.shape} ({file_size:.2f}MB)")
                    
                except Exception as e:
                    logger.warning(f"⚠ Could not process {file_name}: {str(e)}")
                    csv_metadata['csv_files'][file_name] = {
                        'file_path': file_path,
                        'error': str(e)
                    }
            else:
                logger.warning(f"⚠ File not found: {file_path}")
                csv_metadata['csv_files'][file_name] = {
                    'file_path': file_path,
                    'status': 'not_found'
                }
        
        # Save CSV metadata
        metadata_path = os.path.join(csv_artifacts_dir, 'final_csv_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(csv_metadata, f, indent=2, default=str)
        
        # Log metadata as artifact
        mlflow.log_artifact(metadata_path, "final_datasets")
        
        # Log summary metrics
        mlflow.log_metrics({
            'total_csv_files_logged': total_files_logged,
            'csv_artifacts_created': len(csv_files)
        })
        
        logger.info(f"✓ CSV artifacts logged: {total_files_logged}/{len(csv_files)} files")
        
    except Exception as e:
        logger.error(f"✗ Failed to log CSV artifacts: {str(e)}")


def data_pipeline(
    data_path: str = 'data/raw/ChurnModelling.csv',
    target_column: str = 'Exited',
    test_size: float = 0.2,
    force_rebuild: bool = False
) -> Dict[str, np.ndarray]:
    """
    Execute comprehensive data processing pipeline with MLflow tracking.
    
    Args:
        data_path: Path to the raw data file
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        force_rebuild: Whether to force rebuild of existing artifacts
        
    Returns:
        Dictionary containing processed train/test splits
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING DATA PIPELINE")
    logger.info(f"{'='*80}")
    
    # Input validation
    if not os.path.exists(data_path):
        logger.error(f"✗ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if not 0 < test_size < 1:
        logger.error(f"✗ Invalid test_size: {test_size}")
        raise ValueError(f"Invalid test_size: {test_size}")
    
    try:
        # Load configurations
        data_paths = get_data_paths()
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        
        # Initialize MLflow tracking
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags('data_pipeline', {
            'data_source': data_path,
            'force_rebuild': str(force_rebuild),
            'target_column': target_column
        })
        run = mlflow_tracker.start_run(run_name='data_pipeline', tags=run_tags)
        
        # Create artifacts directory
        run_artifacts_dir = os.path.join('artifacts', 'mlflow_run_artifacts', run.info.run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)
        
        # Check for existing artifacts
        x_train_path = os.path.join('artifacts', 'data', 'X_train.csv')
        x_test_path = os.path.join('artifacts', 'data', 'X_test.csv')
        y_train_path = os.path.join('artifacts', 'data', 'Y_train.csv')
        y_test_path = os.path.join('artifacts', 'data', 'Y_test.csv')
        
        artifacts_exist = all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path])
        
        if artifacts_exist and not force_rebuild:
            logger.info("✓ Loading existing processed data artifacts")
            X_train = pd.read_csv(x_train_path)
            X_test = pd.read_csv(x_test_path)
            Y_train = pd.read_csv(y_train_path)
            Y_test = pd.read_csv(y_test_path)
            
            # Log existing data metrics
            log_stage_metrics(X_train, 'existing_train')
            log_stage_metrics(X_test, 'existing_test')
            
            # Log existing datasets as MLflow dataset artifacts
            try:
                import mlflow.data
                
                # Create training dataset from existing data
                train_dataset = mlflow.data.from_pandas(
                    pd.concat([X_train, Y_train], axis=1),
                    source=f"existing_processed_from_{data_path}",
                    name="existing_churn_train_data",
                    targets=target_column
                )
                
                # Create test dataset from existing data
                test_dataset = mlflow.data.from_pandas(
                    pd.concat([X_test, Y_test], axis=1),
                    source=f"existing_processed_from_{data_path}",
                    name="existing_churn_test_data",
                    targets=target_column
                )
                
                # Log the datasets
                mlflow.log_input(train_dataset, context="training")
                mlflow.log_input(test_dataset, context="testing")
                
                logger.info("✓ Existing datasets logged as MLflow dataset artifacts")
                
            except Exception as e:
                logger.warning(f"⚠ Could not log existing dataset artifacts: {str(e)}")
            
            # Log existing CSV files as artifacts with metadata
            logger.info("Logging existing train/test CSV files as MLflow artifacts...")
            existing_csv_files = {
                'X_train': x_train_path,
                'X_test': x_test_path,
                'Y_train': y_train_path,
                'Y_test': y_test_path
            }
            log_csv_artifacts(existing_csv_files, run_artifacts_dir)
            
            mlflow_tracker.log_data_pipeline_metrics({
                'total_samples': len(X_train) + len(X_test),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
            mlflow_tracker.end_run()
            
            logger.info("✓ Data pipeline completed using existing artifacts")
            return {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'Y_train': Y_train.values.ravel(),
                'Y_test': Y_test.values.ravel()
            }
        
        # Process data from scratch
        logger.info("Processing data from scratch...")
        
        # Data ingestion
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        logger.info(f"✓ Raw data loaded: {df.shape}")
        
        # Log raw data metrics and create visualizations
        log_stage_metrics(df, 'raw')
        create_data_visualizations(df, 'raw', run_artifacts_dir)
        
        # Log raw dataset as MLflow dataset artifact
        try:
            import mlflow.data
            from mlflow.data.pandas_dataset import PandasDataset
            
            # Create MLflow dataset from raw data
            raw_dataset = mlflow.data.from_pandas(
                df, 
                source=data_path,
                name="raw_churn_data",
                targets=target_column
            )
            
            # Log the dataset
            mlflow.log_input(raw_dataset, context="raw_data")
            logger.info("✓ Raw dataset logged as MLflow dataset artifact")
            
        except Exception as e:
            logger.warning(f"⚠ Could not log raw dataset artifact: {str(e)}")
            # Fallback: log raw data file as regular artifact
            mlflow.log_artifact(data_path, "raw_data")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Handle missing values
        logger.info("Handling missing values...")
        initial_shape = df.shape
        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])
        age_handler = FillMissingValuesStrategy(method='mean', relevant_column='Age')
        gender_handler = FillMissingValuesStrategy(
            relevant_column='Gender',
            is_custom_imputer=True,
            custom_imputer=GenderImputer()
        )
        
        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = gender_handler.handle(df)
        
        rows_removed = initial_shape[0] - df.shape[0]
        log_stage_metrics(df, 'missing_handled', {'rows_removed': rows_removed})
        logger.info(f"✓ Missing values handled: {initial_shape} → {df.shape}")
        
        # Outlier detection
        logger.info("Detecting and removing outliers...")
        initial_shape = df.shape
        outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
        df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
        
        outliers_removed = initial_shape[0] - df.shape[0]
        log_stage_metrics(df, 'outliers_removed', {'outliers_removed': outliers_removed})
        logger.info(f"✓ Outliers removed: {initial_shape} → {df.shape}")
        
        # Feature binning
        logger.info("Applying feature binning...")
        binning = CustomBinningStratergy(binning_config['credit_score_bins'])
        df = binning.bin_feature(df, 'CreditScore')
        
        # Log binning distribution
        if 'CreditScoreBins' in df.columns:
            bin_dist = df['CreditScoreBins'].value_counts().to_dict()
            mlflow.log_metrics({f'credit_score_bin_{k}': v for k, v in bin_dist.items()})
        
        logger.info("✓ Feature binning completed")
        
        # Feature encoding
        logger.info("Applying feature encoding...")
        nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])
        ordinal_strategy = OrdinalEncodingStratergy(encoding_config['ordinal_mappings'])
        
        df = nominal_strategy.encode(df)
        df = ordinal_strategy.encode(df)
        
        log_stage_metrics(df, 'encoded')
        create_data_visualizations(df, 'encoded', run_artifacts_dir)
        logger.info("✓ Feature encoding completed")
        
        # Feature scaling
        logger.info("Applying feature scaling...")
        minmax_strategy = MinMaxScalingStratergy()
        df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])
        
        # Save scaler artifacts for inference
        logger.info("Saving scaler artifacts for inference...")
        scaler_saved = minmax_strategy.save_scaler(scaling_config['columns_to_scale'], 'artifacts/scale')
        if scaler_saved:
            logger.info("✓ Scaler artifacts saved successfully")
        else:
            logger.warning("⚠ Failed to save scaler artifacts")
        
        logger.info("✓ Feature scaling completed")
        
        # Post-processing
        drop_columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        if existing_drop_columns:
            df = df.drop(columns=existing_drop_columns)
            logger.info(f"✓ Dropped columns: {existing_drop_columns}")
        
        # Data splitting
        logger.info("Splitting data...")
        splitting_strategy = SimpleTrainTestSplitStratergy(test_size=splitting_config['test_size'])
        X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, target_column)
        
        # Create directories and save splits
        os.makedirs('artifacts/data', exist_ok=True)
        X_train.to_csv(x_train_path, index=False)
        X_test.to_csv(x_test_path, index=False)
        Y_train.to_csv(y_train_path, index=False)
        Y_test.to_csv(y_test_path, index=False)
        
        logger.info("✓ Data splitting completed")
        logger.info(f"  • X_train: {X_train.shape}")
        logger.info(f"  • X_test: {X_test.shape}")
        logger.info(f"  • Y_train: {Y_train.shape}")
        logger.info(f"  • Y_test: {Y_test.shape}")
        
        # Final metrics and visualizations
        log_stage_metrics(X_train, 'final_train')
        log_stage_metrics(X_test, 'final_test')
        create_data_visualizations(pd.concat([X_train, X_test]), 'final', run_artifacts_dir)
        
        # Log final processed datasets as MLflow dataset artifacts
        try:
            import mlflow.data
            
            # Create training dataset
            train_dataset = mlflow.data.from_pandas(
                pd.concat([X_train, Y_train], axis=1),
                source=f"processed_from_{data_path}",
                name="processed_churn_train_data",
                targets=target_column
            )
            
            # Create test dataset  
            test_dataset = mlflow.data.from_pandas(
                pd.concat([X_test, Y_test], axis=1),
                source=f"processed_from_{data_path}",
                name="processed_churn_test_data", 
                targets=target_column
            )
            
            # Log the datasets
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="testing")
            
            logger.info("✓ Final processed datasets logged as MLflow dataset artifacts")
            
        except Exception as e:
            logger.warning(f"⚠ Could not log processed dataset artifacts: {str(e)}")
        
        # Log comprehensive pipeline metrics
        comprehensive_metrics = {
            'total_samples': len(X_train) + len(X_test),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'final_features': X_train.shape[1],
            'train_class_0': (Y_train == 0).sum().iloc[0],
            'train_class_1': (Y_train == 1).sum().iloc[0],
            'test_class_0': (Y_test == 0).sum().iloc[0],
            'test_class_1': (Y_test == 1).sum().iloc[0]
        }
        
        mlflow_tracker.log_data_pipeline_metrics(comprehensive_metrics)
        
        # Log parameters
        mlflow.log_params({
            'final_feature_names': list(X_train.columns),
            'preprocessing_steps': ['missing_values', 'outlier_detection', 'feature_binning', 'feature_encoding', 'feature_scaling'],
            'data_pipeline_version': '2.1_optimized'
        })
        
        # Log processed datasets as artifacts
        mlflow.log_artifact(x_train_path, "processed_datasets")
        mlflow.log_artifact(x_test_path, "processed_datasets")
        mlflow.log_artifact(y_train_path, "processed_datasets")
        mlflow.log_artifact(y_test_path, "processed_datasets")
        
        # Log final CSV files as artifacts with detailed metadata
        logger.info("Logging final train/test CSV files as MLflow artifacts...")
        final_csv_files = {
            'X_train': x_train_path,
            'X_test': x_test_path,
            'Y_train': y_train_path,
            'Y_test': y_test_path
        }
        log_csv_artifacts(final_csv_files, run_artifacts_dir)
        
        mlflow_tracker.end_run()
        
        logger.info("✓ Data pipeline completed successfully!")
        logger.info(f"{'='*80}\n")
        
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'Y_train': Y_train.values.ravel(),
            'Y_test': Y_test.values.ravel()
        }
        
    except Exception as e:
        logger.error(f"✗ Data pipeline failed: {str(e)}")
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise


if __name__ == "__main__":
    """
    Execute the data pipeline when the script is run directly.
    """
    try:
        logger.info("Starting data pipeline execution...")
        result = data_pipeline()
        logger.info("Data pipeline execution completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 DATA PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"✅ Training samples: {result['X_train'].shape[0]:,}")
        print(f"✅ Test samples: {result['X_test'].shape[0]:,}")
        print(f"✅ Features: {result['X_train'].shape[1]}")
        print(f"✅ Data artifacts saved to: artifacts/data/")
        print(f"✅ Scaler artifacts saved to: artifacts/scale/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to execute data pipeline: {str(e)}")
        print(f"\n❌ Data pipeline execution failed: {str(e)}")
        exit(1)