"""
Feature scaling strategies for PySpark DataFrames.
Supports MinMaxScaler and StandardScaler transformations.
"""

import logging
import os
import json
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from utils.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Derive project root for dynamic path resolution
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
        self.fitted_model = None
        
    def _configure_s3(self):
        """Configure Spark Hadoop configuration for S3 access if S3 is enabled."""
        # Load credentials/region using our helper config functions
        from utils.config import get_aws_config
        aws_cfg = get_aws_config()
        region = aws_cfg.get('region', 'ap-south-1')
        
        # Extract keys from system environment first, then config
        access_key = os.environ.get('AWS_ACCESS_KEY_ID') or aws_cfg.get('aws_access_key_id')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or aws_cfg.get('aws_secret_access_key')
        
        # Fallback to local profile check if boto3 is available
        if not access_key or not secret_key:
            try:
                import boto3
                # Pop env overrides first
                for env_var in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE']:
                    val = os.getenv(env_var)
                    if val and not os.path.exists(val):
                        del os.environ[env_var]
                session = boto3.Session()
                creds = session.get_credentials()
                if creds:
                    access_key = creds.access_key
                    secret_key = creds.secret_key
            except Exception:
                pass

        # Configure Hadoop S3A settings
        sc = self.spark.sparkContext
        hadoop_conf = sc._jsc.hadoopConfiguration()
        
        hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        if access_key and secret_key:
            hadoop_conf.set("fs.s3a.access.key", access_key)
            hadoop_conf.set("fs.s3a.secret.key", secret_key)
        else:
            hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
            
        hadoop_conf.set("fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
        hadoop_conf.set("fs.s3a.connection.ssl.enabled", "true")
        hadoop_conf.set("fs.s3a.fast.upload", "true")
    
    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Scale specified columns in the DataFrame.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled features
        """
        pass


class ScalingType(str, Enum):
    """Enumeration of scaling types."""
    MINMAX = 'minmax'
    STANDARD = 'standard'


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy to scale features to [0, 1] range."""
    
    def __init__(self, output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Min-Max scaling strategy.
        
        Args:
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        self.pipeline_model = None
        self.columns_to_scale = None
        logger.info("MinMaxScalingStrategy initialized (PySpark)")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        self.columns_to_scale = columns_to_scale
        df_scaled = df
        
        logger.info(f"Applying MinMax scaling to columns: {columns_to_scale}")
        
        # Build a single pipeline for all columns
        stages = []
        
        # Create vector assembler for all columns
        assembler = VectorAssembler(
            inputCols=columns_to_scale,
            outputCol="features_to_scale"
        )
        stages.append(assembler)
        
        # Create MinMaxScaler
        scaler = MinMaxScaler(
            inputCol="features_to_scale",
            outputCol="scaled_features"
        )
        stages.append(scaler)
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=stages)
        self.pipeline_model = pipeline.fit(df_scaled)
        
        # Transform data
        df_scaled = self.pipeline_model.transform(df_scaled)
        
        # Extract scaled values back to original columns
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        # Log scaling statistics
        scaler_model = self.pipeline_model.stages[1]
        for i, col in enumerate(columns_to_scale):
            min_val = float(scaler_model.originalMin[i])
            max_val = float(scaler_model.originalMax[i])
            logger.info(f"✓ Scaled '{col}': min={min_val:.4f}, max={max_val:.4f}")
        
        return df_scaled
    
    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """
        Save the fitted scaler model and metadata for inference.
        
        Args:
            columns_to_scale: List of columns that were scaled
            save_dir: Directory to save scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, save_dir))
                
            os.makedirs(save_dir, exist_ok=True)
            
            if self.pipeline_model is None:
                logger.error("✗ No fitted scaler model to save")
                return False
            
            # Save the pipeline model
            model_path = os.path.join(save_dir, 'minmax_scaler_pipeline')
            self.pipeline_model.write().overwrite().save(model_path)
            logger.info(f"✓ Saved PySpark scaler pipeline to: {model_path}")
            
            # Extract and save metadata for compatibility
            scaler_model = self.pipeline_model.stages[1]  # MinMaxScaler is second stage
            
            # Convert Spark vectors to lists for JSON serialization
            metadata = {
                'columns_to_scale': columns_to_scale,
                'data_min': [float(x) for x in scaler_model.originalMin.toArray()],
                'data_max': [float(x) for x in scaler_model.originalMax.toArray()],
                'n_features': len(columns_to_scale),
                'scaling_type': 'minmax',
                'framework': 'pyspark'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Saved scaling metadata to: {metadata_path}")
            
            # Upload to S3 if S3 I/O is enabled
            try:
                from utils.config import force_s3_io
                if force_s3_io():
                    from utils.config import get_s3_bucket
                    from utils.s3_io import upload_file
                    
                    bucket = get_s3_bucket()
                    self._configure_s3()
                    
                    timestamp = os.environ.get('ACTIVE_RUN_TIMESTAMP')
                    if timestamp:
                        s3_model_path = f"s3a://{bucket}/artifacts/scale/run_{timestamp}/minmax_scaler_pipeline"
                        s3_key = f"artifacts/scale/run_{timestamp}/scaling_metadata.json"
                    else:
                        s3_model_path = f"s3a://{bucket}/artifacts/scale/minmax_scaler_pipeline"
                        s3_key = "artifacts/scale/scaling_metadata.json"
                        
                    # Write Spark pipeline directly to S3
                    self.pipeline_model.write().overwrite().save(s3_model_path)
                    logger.info(f"✓ Saved PySpark scaler pipeline directly to S3: {s3_model_path}")
                    
                    # Upload metadata json
                    upload_file(metadata_path, key=s3_key)
            except Exception as se:
                logger.warning(f"⚠️ Failed to upload MinMaxScaler artifacts to S3: {se}")

            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False
    
    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """
        Load the fitted scaler model for inference.
        
        Args:
            save_dir: Directory containing scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, save_dir))
                
            model_path = os.path.join(save_dir, 'minmax_scaler_pipeline')
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logger.error(f"✗ Scaler artifacts not found in: {save_dir}")
                return False
            
            # Load pipeline model
            self.pipeline_model = PipelineModel.load(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.columns_to_scale = metadata['columns_to_scale']
            
            self.fitted_model = self.pipeline_model
            logger.info(f"✓ Loaded scaler from: {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            return False
    
    def transform(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply the loaded scaler to transform data (no fitting).
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        if self.pipeline_model is None:
            raise ValueError("Scaler not loaded/fitted. Call load_scaler() or scale() first.")
        
        # Transform using the loaded pipeline
        df_scaled = self.pipeline_model.transform(df)
        
        # Extract scaled values back to original columns
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        return df_scaled


class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling strategy to scale features to zero mean and unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, 
                 output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Standard scaling strategy.
        
        Args:
            with_mean: Whether to center the data before scaling
            with_std: Whether to scale the data to unit variance
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.output_col_suffix = output_col_suffix
        self.pipeline_model = None
        self.columns_to_scale = None
        logger.info(f"StandardScalingStrategy initialized (PySpark) - "
                   f"with_mean={with_mean}, with_std={with_std}")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Standard scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        self.columns_to_scale = columns_to_scale
        df_scaled = df
        
        logger.info(f"Applying Standard scaling to columns: {columns_to_scale}")
        
        # Build a single pipeline for all columns
        stages = []
        
        # Create vector assembler for all columns
        assembler = VectorAssembler(
            inputCols=columns_to_scale,
            outputCol="features_to_scale"
        )
        stages.append(assembler)
        
        # Create StandardScaler
        scaler = StandardScaler(
            inputCol="features_to_scale",
            outputCol="scaled_features",
            withMean=self.with_mean,
            withStd=self.with_std
        )
        stages.append(scaler)
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=stages)
        self.pipeline_model = pipeline.fit(df_scaled)
        
        # Transform data
        df_scaled = self.pipeline_model.transform(df_scaled)
        
        # Extract scaled values back to original columns
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        # Log scaling statistics
        scaler_model = self.pipeline_model.stages[1]
        for i, col in enumerate(columns_to_scale):
            mean_val = float(scaler_model.mean[i]) if self.with_mean else 0.0
            std_val = float(scaler_model.std[i]) if self.with_std else 1.0
            logger.info(f"✓ Scaled '{col}': mean={mean_val:.4f}, std={std_val:.4f}")
        
        return df_scaled

    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """
        Save the fitted scaler model and metadata for inference.
        
        Args:
            columns_to_scale: List of columns that were scaled
            save_dir: Directory to save scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, save_dir))
                
            os.makedirs(save_dir, exist_ok=True)
            
            if self.pipeline_model is None:
                logger.error("✗ No fitted scaler model to save")
                return False
            
            # Save the pipeline model
            model_path = os.path.join(save_dir, 'standard_scaler_pipeline')
            self.pipeline_model.write().overwrite().save(model_path)
            logger.info(f"✓ Saved PySpark standard scaler pipeline to: {model_path}")
            
            # Extract and save metadata for compatibility
            scaler_model = self.pipeline_model.stages[1]  # StandardScaler is second stage
            
            # Convert Spark vectors to lists for JSON serialization
            metadata = {
                'columns_to_scale': columns_to_scale,
                'mean': [float(x) for x in scaler_model.mean.toArray()] if self.with_mean else [],
                'std': [float(x) for x in scaler_model.std.toArray()] if self.with_std else [],
                'n_features': len(columns_to_scale),
                'scaling_type': 'standard',
                'framework': 'pyspark'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Saved scaling metadata to: {metadata_path}")
            
            # Upload to S3 if S3 I/O is enabled
            try:
                from utils.config import force_s3_io
                if force_s3_io():
                    from utils.config import get_s3_bucket
                    from utils.s3_io import upload_file
                    
                    bucket = get_s3_bucket()
                    self._configure_s3()
                    
                    timestamp = os.environ.get('ACTIVE_RUN_TIMESTAMP')
                    if timestamp:
                        s3_model_path = f"s3a://{bucket}/artifacts/scale/run_{timestamp}/standard_scaler_pipeline"
                        s3_key = f"artifacts/scale/run_{timestamp}/scaling_metadata.json"
                    else:
                        s3_model_path = f"s3a://{bucket}/artifacts/scale/standard_scaler_pipeline"
                        s3_key = "artifacts/scale/scaling_metadata.json"
                        
                    # Write Spark pipeline directly to S3
                    self.pipeline_model.write().overwrite().save(s3_model_path)
                    logger.info(f"✓ Saved PySpark standard scaler pipeline directly to S3: {s3_model_path}")
                    
                    # Upload metadata json
                    upload_file(metadata_path, key=s3_key)
            except Exception as se:
                logger.warning(f"⚠️ Failed to upload StandardScaler artifacts to S3: {se}")

            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False
            
    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """
        Load the fitted scaler model for inference.
        
        Args:
            save_dir: Directory containing scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, save_dir))
                
            model_path = os.path.join(save_dir, 'standard_scaler_pipeline')
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logger.error(f"✗ Scaler artifacts not found in: {save_dir}")
                return False
            
            # Load pipeline model
            self.pipeline_model = PipelineModel.load(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.columns_to_scale = metadata['columns_to_scale']
            
            self.fitted_model = self.pipeline_model
            logger.info(f"✓ Loaded standard scaler from: {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            return False

    def transform(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply the loaded scaler to transform data (no fitting).
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        if self.pipeline_model is None:
            raise ValueError("Scaler not loaded/fitted. Call load_scaler() or scale() first.")
        
        # Transform using the loaded pipeline
        df_scaled = self.pipeline_model.transform(df)
        
        # Extract scaled values back to original columns
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        return df_scaled

    @classmethod
    def create_scaler_artifacts_from_raw_data(
        cls, 
        raw_data_path: str = 'dataset/raw/fraudTrain.csv', 
        columns_to_scale: List[str] = None, 
        save_dir: str = 'artifacts/scale'
    ) -> bool:
        """Create scaler artifacts from raw data for inference pipeline using PySpark."""
        logger.info(f"\n{'='*60}")
        logger.info("CREATING SCALER ARTIFACTS FROM RAW DATA (PySpark)")
        logger.info(f"{'='*60}")
        
        if not os.path.isabs(raw_data_path):
            raw_data_path = os.path.abspath(os.path.join(_PROJECT_ROOT, raw_data_path))
            
        if not os.path.exists(raw_data_path):
            logger.error(f"✗ Raw data file not found: {raw_data_path}")
            return False
            
        if columns_to_scale is None:
            try:
                from utils.config import get_scaling_config
                scaling_cfg = get_scaling_config()
                columns_to_scale = scaling_cfg.get('columns_to_scale', [
                    'amount_log', 'velocity_last_24h_log', 'city_population_log'
                ])
            except Exception as e:
                logger.warning(f"Failed to load scaling configuration, using default: {e}")
                columns_to_scale = [
                    'amount_log', 'velocity_last_24h_log', 'city_population_log'
                ]
            
        try:
            spark = get_or_create_spark_session()
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(raw_data_path)
            
            # Add log transformed columns if they don't exist yet
            for col in ['amount', 'velocity_last_24h', 'city_population']:
                # Rename 'amt' or 'city_pop' if present
                if col == 'amount' and 'amt' in df.columns:
                    df = df.withColumnRenamed('amt', 'amount')
                if col == 'city_population' and 'city_pop' in df.columns:
                    df = df.withColumnRenamed('city_pop', 'city_population')
                
                log_col = f'{col}_log'
                if log_col not in df.columns:
                    df = df.withColumn(log_col, F.log1p(F.col(col).cast("double")))
            
            scaler_instance = cls(spark=spark)
            scaler_instance.scale(df, columns_to_scale)
            
            success = scaler_instance.save_scaler(columns_to_scale, save_dir)
            return success
        except Exception as e:
            logger.error(f"✗ Failed to create scaler from raw data: {str(e)}")
            return False


class FeatureScalingPipeline:
    """
    Pipeline that runs a scaling strategy on a set of columns in a PySpark DataFrame.
    """
    def __init__(self, strategy: FeatureScalingStrategy, columns_to_scale: List[str]):
        self.strategy = strategy
        self.columns_to_scale = columns_to_scale

    def execute(self, df: DataFrame) -> DataFrame:
        logger.info("Starting PySpark feature scaling pipeline...")
        processed_df = self.strategy.scale(df, self.columns_to_scale)
        logger.info("✓ Feature scaling pipeline complete.")
        return processed_df


def create_default_scaling_pipeline(spark: Optional[SparkSession] = None) -> FeatureScalingPipeline:
    """
    Creates a pre-configured pipeline matching the strategy:
    - Z-score StandardScaler scaling of columns loaded from configuration.
    """
    spark_sess = spark or get_or_create_spark_session()
    
    try:
        from utils.config import get_scaling_config
        scaling_cfg = get_scaling_config()
        columns = scaling_cfg.get('columns_to_scale', [
            'amount_log', 'velocity_last_24h_log', 'city_population_log'
        ])
    except Exception as e:
        logger.warning(f"Failed to load scaling configuration, using default: {e}")
        columns = [
            'amount_log', 'velocity_last_24h_log', 'city_population_log'
        ]
    return FeatureScalingPipeline(
        strategy=StandardScalingStrategy(spark=spark_sess),
        columns_to_scale=columns
    )


if __name__ == "__main__":
    print("🔧 Creating PySpark Scaler Artifacts for Inference")
    success = StandardScalingStrategy.create_scaler_artifacts_from_raw_data()
    if success:
        print("🎉 Scaler artifacts created successfully!")
    else:
        print("✗ Failed to create scaler artifacts.")
        exit(1)