import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class MissingValueHandlingStrategy(ABC):
    """
    Abstract Base Class for missing value handling strategies.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropDuplicatesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to drop duplicate rows in the dataframe.
    """
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("DUPLICATE REMOVAL")
        logger.info(f"{'='*60}")
        
        duplicates = df.duplicated().sum()
        logger.info(f"Number of duplicate rows found: {duplicates}")
        
        if duplicates > 0:
            df_cleaned = df.drop_duplicates().reset_index(drop=True)
            logger.info(f"✓ Dropped duplicate rows. Shape went from {df.shape} to {df_cleaned.shape}")
            logger.info(f"{'='*60}\n")
            return df_cleaned
        else:
            logger.info("✓ No duplicates found. Dataset is clean.")
            logger.info(f"{'='*60}\n")
            return df


class MedianImputationStrategy(MissingValueHandlingStrategy):
    """
    Strategy to impute continuous numerical columns with their median.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("MEDIAN IMPUTATION (Continuous Numerical Features)")
        logger.info(f"{'='*60}")
        
        df_imputed = df.copy()
        for col in self.columns:
            if col in df_imputed.columns:
                n_missing = df_imputed[col].isnull().sum()
                if n_missing > 0:
                    median_val = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_val)
                    logger.info(f"✓ Column [{col}]: Filled {n_missing} missing NaNs with median = {median_val:.4f}")
                else:
                    logger.info(f"  Column [{col}]: No missing values found.")
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"{'='*60}\n")
        return df_imputed


class ModeImputationStrategy(MissingValueHandlingStrategy):
    """
    Strategy to impute binary, count, or categorical columns with their mode.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("MODE IMPUTATION (Binary / Count / Categorical Features)")
        logger.info(f"{'='*60}")
        
        df_imputed = df.copy()
        for col in self.columns:
            if col in df_imputed.columns:
                n_missing = df_imputed[col].isnull().sum()
                if n_missing > 0:
                    mode_series = df_imputed[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series[0]
                        df_imputed[col] = df_imputed[col].fillna(mode_val)
                        logger.info(f"✓ Column [{col}]: Filled {n_missing} missing NaNs with mode = {mode_val}")
                    else:
                        logger.warning(f"⚠ Column [{col}] mode is empty, cannot impute.")
                else:
                    logger.info(f"  Column [{col}]: No missing values found.")
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"{'='*60}\n")
        return df_imputed


class MissingValuePipeline:
    """
    Pipeline that runs a sequence of missing value handling/cleaning strategies.
    """
    def __init__(self, strategies: List[MissingValueHandlingStrategy]):
        self.strategies = strategies

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting missing value handling pipeline...")
        processed_df = df.copy()
        for strategy in self.strategies:
            processed_df = strategy.handle(processed_df)
            
        remaining_missing = processed_df.isnull().sum().sum()
        logger.info(f"Remaining missing values in dataset: {remaining_missing}")
        if remaining_missing == 0:
            logger.info("✓ All missing values successfully resolved.")
        else:
            logger.warning(f"⚠ Warning: {remaining_missing} missing values still remain in the dataset.")
            
        return processed_df


def create_default_missing_value_pipeline() -> MissingValuePipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Removes duplicate rows
    - Imputes continuous columns ('amount', 'device_trust_score', 'cardholder_age') with median
    - Imputes binary, count, and categorical columns with mode
    """
    continuous_cols = ['amount', 'device_trust_score', 'cardholder_age']
    discrete_cols = [
        'foreign_transaction', 'location_mismatch', 'is_fraud',
        'transaction_hour', 'velocity_last_24h', 'merchant_category'
    ]
    
    return MissingValuePipeline([
        DropDuplicatesStrategy(),
        MedianImputationStrategy(columns=continuous_cols),
        ModeImputationStrategy(columns=discrete_cols)
    ])