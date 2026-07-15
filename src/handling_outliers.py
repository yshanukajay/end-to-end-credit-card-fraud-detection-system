import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class OutlierHandlingStrategy(ABC):
    """
    Abstract Base Class for outlier handling strategies.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class ThreeSigmaClipStrategy(OutlierHandlingStrategy):
    """
    Outlier handling strategy using the 3-Sigma (empirical rule) method to cap values.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.bounds: Dict[str, Tuple[float, float]] = {}

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("OUTLIER HANDLING - 3-SIGMA CAPPING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting 3-Sigma capping for columns: {self.columns}")
        
        df_cleaned = df.copy()
        total_capped = 0
        
        for col in self.columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(float)
                mean_val = df_cleaned[col].mean()
                std_val = df_cleaned[col].std()
                lower_fence = mean_val - 3 * std_val
                upper_fence = mean_val + 3 * std_val
                self.bounds[col] = (lower_fence, upper_fence)
                
                before_low = (df_cleaned[col] < lower_fence).sum()
                before_high = (df_cleaned[col] > upper_fence).sum()
                col_capped = before_low + before_high
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_fence, upper=upper_fence)
                
                logger.info(
                    f"✓ Column [{col}]: Fences [{lower_fence:.4f}, {upper_fence:.4f}] | "
                    f"Capped Low: {before_low} | Capped High: {before_high} | Total Capped: {col_capped}"
                )
                total_capped += col_capped
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"Total values capped across all columns: {total_capped}")
        logger.info(f"{'='*60}\n")
        return df_cleaned


class LogTransformStrategy(OutlierHandlingStrategy):
    """
    Outlier handling strategy that applies np.log1p to features and saves them as new columns with '_log' suffix.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("OUTLIER HANDLING - LOG TRANSFORMATION")
        logger.info(f"{'='*60}")
        logger.info(f"Starting log1p transformation for columns (creating _log columns): {self.columns}")
        
        df_cleaned = df.copy()
        for col in self.columns:
            if col in df_cleaned.columns:
                df_cleaned[f'{col}_log'] = np.log1p(df_cleaned[col].astype(float))
                logger.info(f"✓ Column [{col}]: Created [{col}_log] using np.log1p.")
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"{'='*60}\n")
        return df_cleaned


class OutlierHandlingPipeline:
    """
    Pipeline that runs a sequence of outlier handling/transformation strategies.
    """
    def __init__(self, strategies: List[OutlierHandlingStrategy]):
        self.strategies = strategies

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting outlier handling pipeline...")
        processed_df = df.copy()
        for strategy in self.strategies:
            processed_df = strategy.handle(processed_df)
        logger.info("✓ Outlier handling pipeline complete.")
        return processed_df


def create_outlier_pipeline(
    method: str = "default", 
    cap_columns: List[str] = None,
    log_columns: List[str] = None
) -> OutlierHandlingPipeline:
    """
    Factory function to configure the outlier handling pipeline.
    """
    if cap_columns is None:
        cap_columns = ['customer_age', 'distance_to_merchant']
    if log_columns is None:
        log_columns = ['amount', 'velocity_last_24h', 'city_population']
        
    strategies = []
    
    # Log transform highly skewed columns
    if log_columns:
        strategies.append(LogTransformStrategy(columns=log_columns))
    
    # Cap the specified columns using 3-Sigma method
    if cap_columns:
        strategies.append(ThreeSigmaClipStrategy(columns=cap_columns))
        
    return OutlierHandlingPipeline(strategies)


# Alias for backward compatibility
IQRClipStrategy = ThreeSigmaClipStrategy