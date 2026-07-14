import os
import json
import joblib
import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class FeatureScalingStrategy(ABC):
    """
    Abstract Base Class for feature scaling strategies.
    """
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'


class StandardScalingStrategy(FeatureScalingStrategy):
    """
    Strategy to scale features using StandardScaler (Z-score normalization).
    Aligns with notebooks/data_pipeline/4_handle_scaling.ipynb.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        logger.info("StandardScalingStrategy initialized")

    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("FEATURE SCALING - STANDARD")
        logger.info(f"{'='*60}")
        logger.info(f'Starting Standard scaling for {len(columns_to_scale)} columns: {columns_to_scale}')
        
        df_scaled = df.copy()
        
        # Log statistics before scaling
        logger.info("\nStatistics BEFORE scaling:")
        for col in columns_to_scale:
            if col in df_scaled.columns:
                col_stats = {
                    'min': df_scaled[col].min(),
                    'max': df_scaled[col].max(),
                    'mean': df_scaled[col].mean(),
                    'std': df_scaled[col].std()
                }
                logger.info(f"  {col}: Min={col_stats['min']:.2f}, Max={col_stats['max']:.2f}, Mean={col_stats['mean']:.2f}, Std={col_stats['std']:.2f}")
        
        # Apply scaling
        df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
        self.fitted = True
        
        # Log means and variances learned by scaler
        logger.info("\nScaler Parameters:")
        for i, col in enumerate(columns_to_scale):
            logger.info(f"  {col}: Mean={self.scaler.mean_[i]:.2f}, Scale={self.scaler.scale_[i]:.2f}")
        
        # Log statistics after scaling
        logger.info("\nStatistics AFTER scaling:")
        for col in columns_to_scale:
            if col in df_scaled.columns:
                col_stats = {
                    'min': df_scaled[col].min(),
                    'max': df_scaled[col].max(),
                    'mean': df_scaled[col].mean(),
                    'std': df_scaled[col].std()
                }
                logger.info(f"  {col}: Min={col_stats['min']:.4f}, Max={col_stats['max']:.4f}, Mean={col_stats['mean']:.4f}, Std={col_stats['std']:.4f}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f'✓ STANDARD SCALING COMPLETE - {len(columns_to_scale)} columns processed')
        logger.info(f"{'='*60}\n")
        return df_scaled

    def get_scaler(self):
        return self.scaler

    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """Save the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("SAVING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        if not self.fitted:
            logger.error("✗ Scaler not fitted yet. Cannot save.")
            return False
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save scaler object
            scaler_path = os.path.join(save_dir, 'standard_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'columns_to_scale': columns_to_scale,
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'n_features': len(columns_to_scale),
                'scaling_type': 'standard'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Scaler saved to: {scaler_path}")
            logger.info(f"✓ Metadata saved to: {metadata_path}")
            logger.info(f"✓ Columns scaled: {columns_to_scale}")
            logger.info(f"{'='*50}\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False

    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """Load the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("LOADING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        scaler_path = os.path.join(save_dir, 'standard_scaler.joblib')
        metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
        
        if not os.path.exists(scaler_path):
            logger.error(f"✗ Scaler file not found: {scaler_path}")
            return False
        if not os.path.exists(metadata_path):
            logger.error(f"✗ Metadata file not found: {metadata_path}")
            return False
            
        try:
            self.scaler = joblib.load(scaler_path)
            self.fitted = True
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"✓ Scaler loaded from: {save_dir}")
            logger.info(f"✓ Columns to scale: {metadata['columns_to_scale']}")
            logger.info(f"{'='*50}\n")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            return False

    def transform(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """Apply the loaded scaler to transform data (no fitting)."""
        logger.info(f"\n{'='*60}")
        logger.info("FEATURE SCALING - TRANSFORM ONLY")
        logger.info(f"{'='*60}")
        
        if not self.fitted:
            logger.error("✗ Scaler not loaded/fitted. Cannot transform.")
            raise ValueError("Scaler not loaded/fitted. Call load_scaler() first.")
            
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        logger.info(f"✓ SCALING TRANSFORMATION COMPLETE - {len(columns_to_scale)} columns processed")
        logger.info(f"{'='*60}\n")
        return df_scaled

    @classmethod
    def create_scaler_artifacts_from_raw_data(
        cls, 
        raw_data_path: str = 'dataset/raw/credit_card_fraud_10k.csv', 
        columns_to_scale: List[str] = None, 
        save_dir: str = 'artifacts/scale'
    ) -> bool:
        """Create scaler artifacts from raw data for inference pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info("CREATING SCALER ARTIFACTS FROM RAW DATA")
        logger.info(f"{'='*60}")
        
        if not os.path.exists(raw_data_path):
            logger.error(f"✗ Raw data file not found: {raw_data_path}")
            return False
            
        if columns_to_scale is None:
            columns_to_scale = ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']
            
        try:
            df = pd.read_csv(raw_data_path)
            scaler_instance = cls()
            scaler_instance.scaler.fit(df[columns_to_scale])
            scaler_instance.fitted = True
            
            success = scaler_instance.save_scaler(columns_to_scale, save_dir)
            return success
        except Exception as e:
            logger.error(f"✗ Failed to create scaler: {str(e)}")
            return False


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """
    Strategy to scale features using MinMaxScaler.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False
        logger.info("MinMaxScalingStrategy initialized")

    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("FEATURE SCALING - MIN-MAX")
        logger.info(f"{'='*60}")
        logger.info(f'Starting Min-Max scaling for {len(columns_to_scale)} columns: {columns_to_scale}')
        
        df_scaled = df.copy()
        
        logger.info("\nStatistics BEFORE scaling:")
        for col in columns_to_scale:
            if col in df_scaled.columns:
                col_stats = {
                    'min': df_scaled[col].min(),
                    'max': df_scaled[col].max(),
                    'mean': df_scaled[col].mean(),
                    'std': df_scaled[col].std()
                }
                logger.info(f"  {col}: Min={col_stats['min']:.2f}, Max={col_stats['max']:.2f}, Mean={col_stats['mean']:.2f}, Std={col_stats['std']:.2f}")
        
        df_scaled[columns_to_scale] = self.scaler.fit_transform(df_scaled[columns_to_scale])
        self.fitted = True
        
        logger.info("\nScaler Parameters:")
        for i, col in enumerate(columns_to_scale):
            logger.info(f"  {col}: Data min={self.scaler.data_min_[i]:.2f}, Data max={self.scaler.data_max_[i]:.2f}")
            
        logger.info("\nStatistics AFTER scaling:")
        for col in columns_to_scale:
            if col in df_scaled.columns:
                col_stats = {
                    'min': df_scaled[col].min(),
                    'max': df_scaled[col].max(),
                    'mean': df_scaled[col].mean(),
                    'std': df_scaled[col].std()
                }
                logger.info(f"  {col}: Min={col_stats['min']:.4f}, Max={col_stats['max']:.4f}, Mean={col_stats['mean']:.4f}, Std={col_stats['std']:.4f}")
                
        logger.info(f"\n{'='*60}")
        logger.info(f'✓ MIN-MAX SCALING COMPLETE - {len(columns_to_scale)} columns processed')
        logger.info(f"{'='*60}\n")
        return df_scaled

    def get_scaler(self):
        return self.scaler

    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """Save the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("SAVING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        if not self.fitted:
            logger.error("✗ Scaler not fitted yet. Cannot save.")
            return False
            
        try:
            os.makedirs(save_dir, exist_ok=True)
            scaler_path = os.path.join(save_dir, 'minmax_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            metadata = {
                'columns_to_scale': columns_to_scale,
                'data_min': self.scaler.data_min_.tolist(),
                'data_max': self.scaler.data_max_.tolist(),
                'n_features': len(columns_to_scale),
                'scaling_type': 'minmax'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"✓ Scaler saved to: {scaler_path}")
            logger.info(f"✓ Metadata saved to: {metadata_path}")
            logger.info(f"{'='*50}\n")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False

    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """Load the fitted scaler and metadata for inference."""
        logger.info(f"\n{'='*50}")
        logger.info("LOADING SCALER ARTIFACTS")
        logger.info(f"{'='*50}")
        
        scaler_path = os.path.join(save_dir, 'minmax_scaler.joblib')
        metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
        
        if not os.path.exists(scaler_path) or not os.path.exists(metadata_path):
            logger.error("✗ Scaler files not found.")
            return False
            
        try:
            self.scaler = joblib.load(scaler_path)
            self.fitted = True
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load: {str(e)}")
            return False

    def transform(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Scaler not fitted.")
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        return df_scaled

    @classmethod
    def create_scaler_artifacts_from_raw_data(
        cls, 
        raw_data_path: str = 'dataset/raw/credit_card_fraud_10k.csv', 
        columns_to_scale: List[str] = None, 
        save_dir: str = 'artifacts/scale'
    ) -> bool:
        if not os.path.exists(raw_data_path):
            return False
        if columns_to_scale is None:
            columns_to_scale = ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']
        try:
            df = pd.read_csv(raw_data_path)
            scaler_instance = cls()
            scaler_instance.scaler.fit(df[columns_to_scale])
            scaler_instance.fitted = True
            return scaler_instance.save_scaler(columns_to_scale, save_dir)
        except Exception as e:
            return False


# Typo-resilient aliases to support legacy codebase / boilerplate usage
MinMaxScalingStratergy = MinMaxScalingStrategy
StandardScalingStratergy = StandardScalingStrategy


class FeatureScalingPipeline:
    """
    Pipeline that runs a scaling strategy on a set of columns in a DataFrame.
    """
    def __init__(self, strategy: FeatureScalingStrategy, columns_to_scale: List[str]):
        self.strategy = strategy
        self.columns_to_scale = columns_to_scale

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature scaling pipeline...")
        processed_df = self.strategy.scale(df, self.columns_to_scale)
        logger.info("✓ Feature scaling pipeline complete.")
        return processed_df


def create_default_scaling_pipeline() -> FeatureScalingPipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Z-score StandardScaler scaling of columns: ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']
    """
    columns = ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']
    return FeatureScalingPipeline(
        strategy=StandardScalingStrategy(),
        columns_to_scale=columns
    )


if __name__ == "__main__":
    print("🔧 Creating Scaler Artifacts for Inference")
    success = StandardScalingStrategy.create_scaler_artifacts_from_raw_data()
    if success:
        print("🎉 Scaler artifacts created successfully!")
    else:
        print("❌ Failed to create scaler artifacts.")
        exit(1)