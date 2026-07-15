import os
import json
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)

# Derive the project root from this file's location so artifact paths always
# resolve to the project root regardless of the caller's working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ENCODE_DIR = os.path.join(_PROJECT_ROOT, 'artifacts', 'encode')


class FeatureEncodingStrategy(ABC):
    """
    Abstract Base Class for feature encoding strategies.
    """
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """
    Strategy to encode nominal categorical features using one-hot encoding.
    Stores and saves category maps for inference consistency.
    """
    def __init__(self, nominal_columns: List[str]):
        self.nominal_columns = nominal_columns
        self.encoder_mappings = {}  # Store category mappings for inference
        os.makedirs(_ENCODE_DIR, exist_ok=True)
        logger.info(f"NominalEncodingStrategy initialized for ONE-HOT encoding of columns: {nominal_columns}")

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("ONE-HOT ENCODING (NOMINAL)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting one-hot encoding for {len(self.nominal_columns)} columns")
        
        df_encoded = df.copy()
        
        for column in self.nominal_columns:
            if column not in df_encoded.columns:
                logger.warning(f"⚠ Column [{column}] not found in the DataFrame.")
                continue
                
            logger.info(f"\n--- Processing column: {column} ---")
            unique_values = sorted(df_encoded[column].dropna().unique())  # Sort for consistency
            logger.info(f"  Unique values ({len(unique_values)}): {unique_values}")
            
            # Store mapping for inference
            self.encoder_mappings[column] = unique_values
            
            # Save encoder mapping
            encoder_path = os.path.join(_ENCODE_DIR, f"{column}_encoder.json")
            with open(encoder_path, "w") as f:
                json.dump({'categories': unique_values, 'encoding_type': 'one_hot'}, f)
            logger.info(f"  ✓ Saved one-hot encoder mapping to {encoder_path}")
            
            # Check for missing values before encoding
            missing_count = df_encoded[column].isnull().sum()
            if missing_count > 0:
                logger.warning(f"  ⚠ Column has {missing_count} missing values before encoding")
            
            # Create one-hot encoded columns
            for value in unique_values:
                new_col_name = f"{column}_{value}"
                df_encoded[new_col_name] = (df_encoded[column] == value).astype(int)
                logger.info(f"    ✓ Created binary column: {new_col_name}")
            
            # Drop the original column
            df_encoded = df_encoded.drop(columns=[column])
            logger.info(f"  ✓ Dropped original column '{column}'")
            logger.info(f"  ✓ One-hot encoding complete for '{column}' -> {len(unique_values)} binary columns")
            
        logger.info(f"\n{'='*60}")
        logger.info("✓ ONE-HOT ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df_encoded
    
    def get_encoder_mappings(self) -> Dict[str, List]:
        return self.encoder_mappings


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """
    Strategy to encode ordinal features using custom mapping dictionaries.
    """
    def __init__(self, ordinal_mappings: Dict[str, Dict]):
        self.ordinal_mappings = ordinal_mappings
        logger.info(f"OrdinalEncodingStrategy initialized for columns: {list(ordinal_mappings.keys())}")

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("ORDINAL ENCODING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting ordinal encoding for {len(self.ordinal_mappings)} columns")
        
        df_encoded = df.copy()
        
        for column, mapping in self.ordinal_mappings.items():
            if column not in df_encoded.columns:
                logger.warning(f"⚠ Column [{column}] not found in the DataFrame.")
                continue
                
            logger.info(f"\n--- Processing column: {column} ---")
            initial_values = df_encoded[column].value_counts()
            logger.info(f"  Mapping: {mapping}")
            logger.info(f"  Before encoding: {initial_values.to_dict()}")
            
            # Check for missing values before encoding
            missing_count = df_encoded[column].isnull().sum()
            if missing_count > 0:
                logger.warning(f"  ⚠ Column has {missing_count} missing values before encoding")
            
            # Map values and cast to Int64 to match the notebook type structure
            df_encoded[column] = df_encoded[column].map(mapping).astype('Int64')
            
            # Check for unmapped values
            unmapped_count = df_encoded[column].isnull().sum()
            if unmapped_count > missing_count:
                logger.warning(f"  ⚠ Column has {unmapped_count - missing_count} unmapped values after encoding")
                
            # Log encoded value distribution
            encoded_values = df_encoded[column].value_counts()
            logger.info(f"  ✓ Encoded with {len(mapping)} categories")
            logger.info(f"  After encoding: {encoded_values.to_dict()}")
            
        logger.info(f"\n{'='*60}")
        logger.info("✓ ORDINAL ENCODING COMPLETE")
        logger.info(f"{'='*60}\n")
        return df_encoded


class FeatureEncodingPipeline:
    """
    Pipeline that runs a sequence of encoding strategies on a DataFrame.
    """
    def __init__(self, strategies: List[FeatureEncodingStrategy]):
        self.strategies = strategies

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature encoding pipeline...")
        processed_df = df.copy()
        for strategy in self.strategies:
            processed_df = strategy.encode(processed_df)
        logger.info("✓ Feature encoding pipeline complete.")
        return processed_df


def create_default_encoding_pipeline() -> FeatureEncodingPipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Nominal/One-hot encoding of 'merchant_category' and 'gender'
    - Ordinal encoding of binned age, hour, and distance features
    """
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
        NominalEncodingStrategy(nominal_columns=nominal_cols),
        OrdinalEncodingStrategy(ordinal_mappings=ordinal_mappings)
    ])
