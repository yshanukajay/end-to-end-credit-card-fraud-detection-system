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


import numpy as np

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_velocity_and_freq(df):
    if 'cc_num' not in df.columns or 'unix_time' not in df.columns:
        return df
    df_sorted = df.sort_values(by=['cc_num', 'unix_time']).reset_index(drop=True)
    
    velocities = np.zeros(len(df_sorted), dtype=int)
    frequencies = np.zeros(len(df_sorted), dtype=int)
    
    grouped = df_sorted.groupby('cc_num')
    
    for cc, group in grouped:
        times = group['unix_time'].values
        indices = group.index.values
        
        start_indices = np.searchsorted(times, times - 86400, side='left')
        local_indices = np.arange(len(group))
        
        velocities[indices] = local_indices - start_indices
        frequencies[indices] = local_indices
        
    df_sorted['velocity_last_24h'] = velocities
    df_sorted['transaction_frequency'] = frequencies
    return df_sorted


class RawFeatureEngineeringStrategy(MissingValueHandlingStrategy):
    """
    Strategy to engineer raw Kaggle credit card transactions dataset into engineered features.
    Saves and renames columns to match the recommended 13 columns.
    """
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("RAW FEATURE ENGINEERING")
        logger.info(f"{'='*60}")
        
        if 'distance_to_merchant' in df.columns and 'customer_age' in df.columns:
            logger.info("✓ Columns are already engineered. Skipping raw feature engineering.")
            logger.info(f"{'='*60}\n")
            return df
            
        logger.info("Starting feature engineering on raw transaction columns...")
        df_engineered = df.copy()
        
        # 1. Datetime conversions
        if 'trans_date_trans_time' in df_engineered.columns:
            df_engineered['transaction_datetime'] = pd.to_datetime(df_engineered['trans_date_trans_time'])
        if 'dob' in df_engineered.columns:
            df_engineered['dob'] = pd.to_datetime(df_engineered['dob'])
            
        # 2. Age calculation
        if 'transaction_datetime' in df_engineered.columns and 'dob' in df_engineered.columns:
            df_engineered['customer_age'] = (df_engineered['transaction_datetime'] - df_engineered['dob']).dt.days // 365
            
        # 3. Time features
        if 'transaction_datetime' in df_engineered.columns:
            df_engineered['transaction_hour'] = df_engineered['transaction_datetime'].dt.hour
            df_engineered['day_of_week'] = df_engineered['transaction_datetime'].dt.dayofweek
            df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6]).astype(int)
            df_engineered['transaction_month'] = df_engineered['transaction_datetime'].dt.month
            
        # 4. Distance features
        if all(col in df_engineered.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
            df_engineered['distance_to_merchant'] = haversine_np(
                df_engineered['lat'], df_engineered['long'], 
                df_engineered['merch_lat'], df_engineered['merch_long']
            )
            df_engineered['location_mismatch'] = (df_engineered['distance_to_merchant'] > 80).astype(int)
            
        # 5. Amount features
        if 'amt' in df_engineered.columns:
            df_engineered['amount_zscore'] = (df_engineered['amt'] - df_engineered['amt'].mean()) / df_engineered['amt'].std()
            df_engineered['amount_log'] = np.log1p(df_engineered['amt'])
            
        # 6. Night transaction
        if 'transaction_hour' in df_engineered.columns:
            df_engineered['night_transaction'] = df_engineered['transaction_hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
            
        # 7. Velocity and frequency
        df_engineered = compute_velocity_and_freq(df_engineered)
        
        # 8. Foreign transaction
        if 'distance_to_merchant' in df_engineered.columns:
            df_engineered['foreign_transaction'] = (df_engineered['distance_to_merchant'] > 150).astype(int)
            
        # Rename columns to standard names
        rename_dict = {
            'unix_time': 'transaction_unix_time',
            'amt': 'amount',
            'category': 'merchant_category',
            'city': 'customer_city',
            'state': 'customer_state',
            'lat': 'customer_latitude',
            'long': 'customer_longitude',
            'city_pop': 'city_population',
            'job': 'occupation',
            'merch_lat': 'merchant_latitude',
            'merch_long': 'merchant_longitude'
        }
        df_engineered = df_engineered.rename(columns={k: v for k, v in rename_dict.items() if k in df_engineered.columns})
        
        recommended_cols = [
            'distance_to_merchant',
            'customer_age',
            'transaction_hour',
            'day_of_week',
            'is_weekend',
            'location_mismatch',
            'velocity_last_24h',
            'foreign_transaction',
            'amount',
            'merchant_category',
            'city_population',
            'gender',
        ]
        
        # Keep features present in df_engineered
        existing_cols = [col for col in recommended_cols if col in df_engineered.columns]
        if 'is_fraud' in df_engineered.columns:
            existing_cols.append('is_fraud')
            
        df_final = df_engineered[existing_cols]
        logger.info(f"✓ Engineered raw features. Shape went from {df.shape} to {df_final.shape}")
        logger.info(f"{'='*60}\n")
        return df_final


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
    continuous_cols = ['amount', 'customer_age', 'distance_to_merchant', 'city_population']
    discrete_cols = [
        'foreign_transaction', 'location_mismatch', 'is_fraud',
        'transaction_hour', 'velocity_last_24h', 'merchant_category',
        'gender', 'day_of_week', 'is_weekend'
    ]
    
    return MissingValuePipeline([
        DropDuplicatesStrategy(),
        RawFeatureEngineeringStrategy(),
        MedianImputationStrategy(columns=continuous_cols),
        ModeImputationStrategy(columns=discrete_cols)
    ])