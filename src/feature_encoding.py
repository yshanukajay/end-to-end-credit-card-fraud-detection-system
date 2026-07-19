"""
Feature encoding strategies for PySpark DataFrames.
Supports nominal encoding (StringIndexer, OneHotEncoder) and ordinal encoding.
"""

import logging
import os
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml import Pipeline
from utils.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Derive the project root from this file's location so artifact paths always
# resolve to the project root regardless of the caller's working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ENCODE_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'encode')


class FeatureEncodingStrategy(ABC):
    """Abstract base class for feature encoding strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Encode features in the DataFrame.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with encoded features
        """
        pass


class VariableType(str, Enum):
    """Enumeration of variable types."""
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """
    Nominal encoding strategy using StringIndexer followed by OneHotEncoder.
    Creates one-hot encoded binary columns for categorical values.
    """
    
    def __init__(self, nominal_columns: List[str], one_hot: bool = True, spark: Optional[SparkSession] = None):
        """
        Initialize nominal encoding strategy.
        
        Args:
            nominal_columns: List of column names to encode
            one_hot: Whether to apply one-hot encoding after indexing (default: True)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.nominal_columns = nominal_columns
        self.one_hot = one_hot
        self.encoder_dicts = {}
        self.indexers = {}
        self.encoders = {}
        self.pipeline_model = None
        os.makedirs(_ENCODE_DIR, exist_ok=True)
        logger.info(f"NominalEncodingStrategy initialized for ONE-HOT encoding: {nominal_columns}")
        logger.info(f"One-hot encoding: {one_hot}")
    
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Apply one-hot encoding to specified columns.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        logger.info("ONE-HOT ENCODING (NOMINAL)")
        df_encoded = df
        
        # Build pipeline stages
        stages = []
        
        for column in self.nominal_columns:
            # String indexer stage
            indexer = StringIndexer(
                inputCol=column,
                outputCol=f"{column}_index",
                handleInvalid="keep"
            )
            
            # One-hot encoder stage
            encoder = OneHotEncoder(
                inputCol=f"{column}_index",
                outputCol=f"{column}_encoded",
                dropLast=False  # Keep all categories for consistency
            )
            
            stages.extend([indexer, encoder])
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=stages)
        self.pipeline_model = pipeline.fit(df_encoded)
        
        # Transform data
        df_encoded = self.pipeline_model.transform(df_encoded)
        
        # Save encoder mappings for inference
        for i, column in enumerate(self.nominal_columns):
            # Get StringIndexer model from pipeline
            indexer_model = self.pipeline_model.stages[i * 2]
            self.indexers[column] = indexer_model
            
            # Get labels (categories)
            labels = indexer_model.labels
            
            # Create encoder dictionary for pandas compatibility
            encoder_dict = {label: idx for idx, label in enumerate(labels)}
            self.encoder_dicts[column] = encoder_dict
            
            # Save encoder mapping to JSON for inference
            encoder_path = os.path.join(_ENCODE_DIR, f"{column}_encoder.json")
            with open(encoder_path, "w") as f:
                json.dump({
                    'categories': list(labels),
                    'encoding_type': 'one_hot',
                    'mappings': encoder_dict
                }, f, indent=2)
            
            # Upload to S3 if S3 I/O is enabled
            try:
                from utils.config import force_s3_io
                if force_s3_io():
                    from utils.s3_io import upload_file
                    timestamp = os.environ.get('ACTIVE_RUN_TIMESTAMP')
                    if timestamp:
                        s3_key = f"artifacts/encode/run_{timestamp}/{column}_encoder.json"
                    else:
                        s3_key = f"artifacts/encode/{column}_encoder.json"
                    upload_file(encoder_path, key=s3_key)
            except Exception as se:
                logger.warning(f"⚠️ Failed to upload encoder mapping for '{column}' to S3: {se}")

            logger.info(f"✓ One-hot encoded '{column}': {len(labels)} categories → {len(labels)} binary columns")
            
            # Extract one-hot encoded values to separate columns
            # This makes the data compatible with pandas/sklearn models
            for idx, label in enumerate(labels):
                new_col_name = f"{column}_{label}"
                # Extract the idx-th element from the sparse vector
                df_encoded = df_encoded.withColumn(
                    new_col_name,
                    F.when(F.col(f"{column}_index") == idx, 1).otherwise(0)
                )
            
            # Drop original column, index column, and encoded vector column
            df_encoded = df_encoded.drop(column, f"{column}_index", f"{column}_encoded")
        
        logger.info("✓ One-hot encoding completed")
        return df_encoded

    def get_encoder_dicts(self) -> Dict[str, Dict[str, int]]:
        """Get the encoder dictionaries for all columns."""
        return self.encoder_dicts
    
    def get_indexers(self) -> Dict[str, StringIndexer]:
        """Get the fitted StringIndexer models."""
        return self.indexers


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """
    Ordinal encoding strategy with custom ordering.
    Maps categorical values to ordered numeric values.
    """
    
    def __init__(self, ordinal_mappings: Dict[str, Dict[str, int]], spark: Optional[SparkSession] = None):
        """
        Initialize ordinal encoding strategy.
        
        Args:
            ordinal_mappings: Dictionary mapping column names to value->order mappings
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")
    
    def encode(self, df: DataFrame) -> DataFrame:
        """
        Apply ordinal encoding to specified columns.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with encoded columns
        """
        df_encoded = df

        for column, mapping in self.ordinal_mappings.items():
            mapping_expr = F.when(F.col(column).isNull(), None)
            for value, code in mapping.items():
                mapping_expr = mapping_expr.when(F.col(column) == value, code)

            df_encoded = df_encoded.withColumn(column, mapping_expr)

        return df_encoded


class FeatureEncodingPipeline:
    """
    Pipeline that runs a sequence of encoding strategies on a PySpark DataFrame.
    """
    def __init__(self, strategies: List[FeatureEncodingStrategy]):
        self.strategies = strategies

    def execute(self, df: DataFrame) -> DataFrame:
        logger.info("Starting PySpark feature encoding pipeline...")
        processed_df = df
        for strategy in self.strategies:
            processed_df = strategy.encode(processed_df)
        logger.info("✓ Feature encoding pipeline complete.")
        return processed_df


def create_default_encoding_pipeline(spark: Optional[SparkSession] = None) -> FeatureEncodingPipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Nominal/One-hot encoding of 'merchant_category' and 'gender'
    - Ordinal encoding of binned age, hour, and distance features
    """
    spark_sess = spark or get_or_create_spark_session()
    
    nominal_cols = ["merchant_category", "gender"]
    ordinal_mappings = {
        "customer_age_binned": {
            "Youth": 0,
            "Young-Adult": 1,
            "Middle-Aged": 2,
            "Senior": 3
        },
        "transaction_hour_binned": {
            "Morning": 0,
            "Afternoon": 1,
            "Evening": 2,
            "Night": 3
        },
        "distance_to_merchant_binned": {
            "Close": 0,
            "Moderate": 1,
            "Far": 2
        }
    }
    
    return FeatureEncodingPipeline([
        NominalEncodingStrategy(nominal_columns=nominal_cols, spark=spark_sess),
        OrdinalEncodingStrategy(ordinal_mappings=ordinal_mappings, spark=spark_sess)
    ])
