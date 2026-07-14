import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class FeatureBinningStrategy(ABC):
    """
    Abstract Base Class for feature binning strategies.
    """
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """
    Strategy to bin continuous numerical columns into discrete bins using custom definitions.
    Fits the structure of credit_card_fraud_detection (e.g. device_trust_score).
    """
    def __init__(self, bin_definitions: Dict[str, List[float]]):
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}")

    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE BINNING - {column.upper()}")
        logger.info(f"{'='*60}")
        
        if column not in df.columns:
            logger.warning(f"⚠ Column [{column}] not found in the DataFrame.")
            logger.info(f"{'='*60}\n")
            return df
            
        logger.info(f"Starting binning for column: {column}")
        initial_unique = df[column].nunique()
        
        # Check if column is empty or all NaN
        if df[column].dropna().empty:
            value_range = (np.nan, np.nan)
        else:
            value_range = (df[column].min(), df[column].max())
            
        logger.info(f"  Unique values: {initial_unique}, Range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")
        
        df_binned = df.copy()
        
        def assign_bin(value):
            if pd.isna(value):
                return np.nan
            
            # Check range match
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    # check if within [low, high)
                    if bin_range[0] <= value < bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    # check if >= threshold
                    if value >= bin_range[0]:
                        return bin_label
                        
            # Fallback/last check for max bounds (e.g. 100 or 850 depending on column max)
            # Find the bin with the largest upper limit
            max_limit = -1.0
            max_label = "Excellent"
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2 and bin_range[1] > max_limit:
                    max_limit = bin_range[1]
                    max_label = bin_label
            if value >= max_limit:
                return max_label
                
            return "Invalid"

        binned_column_name = f"{column}_binned"
        df_binned[binned_column_name] = df_binned[column].apply(assign_bin)
        
        # Log binning results
        bin_counts = df_binned[binned_column_name].value_counts(dropna=False)
        logger.info("\nBinning Results:")
        for bin_name, count in bin_counts.items():
            percentage = (count / len(df_binned)) * 100
            logger.info(f"  ✓ {bin_name}: {count} ({percentage:.2f}%)")
            
        invalid_count = (df_binned[binned_column_name] == "Invalid").sum()
        if invalid_count > 0:
            logger.warning(f"  ⚠ Found {invalid_count} invalid values in column '{column}'")
            
        df_binned.drop(columns=[column], inplace=True)
        logger.info(f"✓ Original column '{column}' removed, replaced with '{binned_column_name}'")
        logger.info(f"{'='*60}\n")
        
        return df_binned

class FeatureBinningPipeline:
    """
    Pipeline that runs a sequence of feature binning strategies on specific columns.
    """
    def __init__(self, steps: List[Tuple[FeatureBinningStrategy, str]]):
        """
        steps: List of tuples (strategy, column_name)
        """
        self.steps = steps

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature binning pipeline...")
        processed_df = df.copy()
        for strategy, column in self.steps:
            processed_df = strategy.bin_feature(processed_df, column)
        logger.info("✓ Feature binning pipeline complete.")
        return processed_df


def create_default_binning_pipeline() -> FeatureBinningPipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Bins 'device_trust_score' with Poor (<25), Fair (<50), Good (<75), Excellent (>=75)
    """
    device_trust_bins = {
        "Poor": [0.0, 25.0],
        "Fair": [25.0, 50.0],
        "Good": [50.0, 75.0],
        "Excellent": [75.0, 100.0]
    }
    
    return FeatureBinningPipeline([
        (CustomBinningStrategy(bin_definitions=device_trust_bins), "device_trust_score")
    ])
