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


class IQRClipStrategy(OutlierHandlingStrategy):
    """
    Outlier handling strategy using the IQR (Interquartile Range) method to clip/cap values.
    Also known as Winsorization. This aligns with notebooks/data_pipeline/1_handle_outliers.ipynb.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.fences: Dict[str, Tuple[float, float]] = {}

    def _compute_iqr_bounds(self, series: pd.Series) -> Tuple[float, float]:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        return lower_fence, upper_fence

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("OUTLIER HANDLING - IQR CAPPING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting IQR capping for columns: {self.columns}")
        
        df_cleaned = df.copy()
        total_capped = 0
        
        for col in self.columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(float)
                lower, upper = self._compute_iqr_bounds(df_cleaned[col])
                self.fences[col] = (lower, upper)
                
                before_low = (df_cleaned[col] < lower).sum()
                before_high = (df_cleaned[col] > upper).sum()
                col_capped = before_low + before_high
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
                
                logger.info(
                    f"✓ Column [{col}]: Fences [{lower:.4f}, {upper:.4f}] | "
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
    Alternative outlier handling strategy that applies np.log1p to highly skewed features
    (like transaction amount) instead of capping them. This addresses the 'outlier headache'
    where we want to preserve the magnitude of extreme legitimate transactions.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("OUTLIER HANDLING - LOG TRANSFORMATION")
        logger.info(f"{'='*60}")
        logger.info(f"Starting log1p transformation for columns: {self.columns}")
        
        df_cleaned = df.copy()
        for col in self.columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = np.log1p(df_cleaned[col].astype(float))
                logger.info(f"✓ Column [{col}]: Applied np.log1p transformation.")
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
    method: str = "cap", 
    cap_columns: List[str] = None,
    log_columns: List[str] = None
) -> OutlierHandlingPipeline:
    """
    Factory function to configure the outlier handling pipeline.
    
    Args:
        method: 'cap' to match the notebook's standard IQR capping,
                'log_and_cap' to handle the 'amount headache' by log-transforming amount
                and capping other numeric columns.
        cap_columns: List of columns to apply IQR capping. Defaults to standard numeric cols.
        log_columns: List of columns to log-transform. Defaults to ['amount'].
    """
    if cap_columns is None:
        cap_columns = ['velocity_last_24h', 'device_trust_score', 'cardholder_age']
    if log_columns is None:
        log_columns = ['amount']
        
    strategies = []
    
    if method == "cap":
        # Match standard notebook behavior: cap all numeric columns (including amount)
        all_cols = list(set(cap_columns + log_columns))
        strategies.append(IQRClipStrategy(columns=all_cols))
    elif method == "log_and_cap":
        # Resolve 'amount headache': log-transform amount and cap the rest
        if log_columns:
            strategies.append(LogTransformStrategy(columns=log_columns))
        if cap_columns:
            cap_filtered = [c for c in cap_columns if c not in log_columns]
            strategies.append(IQRClipStrategy(columns=cap_filtered))
    else:
        raise ValueError(f"Unknown outlier handling method: {method}")
        
    return OutlierHandlingPipeline(strategies)