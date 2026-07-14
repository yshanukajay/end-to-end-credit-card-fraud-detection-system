import os
import logging
import pandas as pd
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class DataSplittingStrategy(ABC):
    """
    Abstract Base Class for data splitting strategies.
    """
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Strategy to split data into train and test sets using simple random sampling.
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"SimpleTrainTestSplitStrategy initialized with test_size={test_size}")

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"\n{'='*60}")
        logger.info("SIMPLE DATA SPLITTING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting simple data splitting with target column: '{target_column}'")
        logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Log target distribution
        target_dist = y.value_counts()
        logger.info("\nTarget Variable Distribution:")
        for value, count in target_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y)*100:.2f}%)")
            
        # Perform simple split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Log split results
        logger.info("\nSplit Results:")
        logger.info(f"  ✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  ✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        train_dist = y_train.value_counts()
        test_dist = y_test.value_counts()
        logger.info("\nTarget Distribution in Training Set:")
        for value, count in train_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y_train)*100:.2f}%)")
        logger.info("\nTarget Distribution in Test Set:")
        for value, count in test_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y_test)*100:.2f}%)")
            
        logger.info(f"{'='*60}\n")
        return X_train, X_test, y_train, y_test


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Strategy to split data into train and test sets using stratification to preserve class ratios.
    Aligns with notebooks/data_pipeline/5_handle_imbalance_smote.ipynb.
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"StratifiedTrainTestSplitStrategy initialized with test_size={test_size}")

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"\n{'='*60}")
        logger.info("STRATIFIED DATA SPLITTING")
        logger.info(f"{'='*60}")
        logger.info(f"Starting stratified data splitting with target column: '{target_column}'")
        logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Log target distribution
        target_dist = y.value_counts()
        logger.info("\nTarget Variable Distribution:")
        for value, count in target_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y)*100:.2f}%)")
            
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Log split results
        logger.info("\nSplit Results:")
        logger.info(f"  ✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  ✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        train_dist = y_train.value_counts()
        test_dist = y_test.value_counts()
        logger.info("\nTarget Distribution in Training Set:")
        for value, count in train_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y_train)*100:.2f}%)")
        logger.info("\nTarget Distribution in Test Set:")
        for value, count in test_dist.items():
            logger.info(f"  {value}: {count} ({count/len(y_test)*100:.2f}%)")
            
        logger.info(f"{'='*60}\n")
        return X_train, X_test, y_train, y_test


class SMOTENCOversampler:
    """
    Oversampler strategy using SMOTENC (Synthetic Minority Over-sampling Technique for Nominal Continuous).
    """
    def __init__(self, continuous_cols: List[str], random_state: int = 42):
        self.continuous_cols = continuous_cols
        self.random_state = random_state
        logger.info(f"SMOTENCOversampler initialized with continuous columns: {continuous_cols}")

    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"\n{'='*60}")
        logger.info("CLASS IMBALANCE HANDLING - SMOTENC")
        logger.info(f"{'='*60}")
        logger.info(f"Starting SMOTENC oversampling. Original shape: {X.shape}")
        
        # Identify categorical columns for SMOTENC
        categorical_cols = [col for col in X.columns if col not in self.continuous_cols]
        categorical_features = [X.columns.get_loc(col) for col in categorical_cols]
        
        if not categorical_features:
            logger.error("✗ No categorical columns detected for SMOTENC.")
            raise ValueError("No categorical columns were detected for SMOTENC.")
            
        logger.info(f"  Detected continuous columns: {self.continuous_cols}")
        logger.info(f"  Detected categorical columns: {categorical_cols}")
        
        smote = SMOTENC(categorical_features=categorical_features, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Cast resampled data back to pandas objects with correct types/columns
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        logger.info(f"✓ SMOTENC complete. Resampled shape: {X_resampled.shape}")
        logger.info("\nResampled Target Distribution:")
        dist = y_resampled.value_counts()
        for val, count in dist.items():
            logger.info(f"  {val}: {count} ({count/len(y_resampled)*100:.2f}%)")
        logger.info(f"{'='*60}\n")
        
        return X_resampled, y_resampled


class SplitAndResampleStrategy(DataSplittingStrategy):
    """
    Composite strategy that first splits the data using a splitting strategy,
    then applies SMOTENC oversampling only to the training set to resolve class imbalance.
    """
    def __init__(self, splitter: DataSplittingStrategy, oversampler: SMOTENCOversampler):
        self.splitter = splitter
        self.oversampler = oversampler
        logger.info("SplitAndResampleStrategy initialized")

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # 1. Split the data
        X_train, X_test, y_train, y_test = self.splitter.split_data(df, target_column)
        
        # 2. Resample the training set only (prevent leakage to test set)
        X_train_resampled, y_train_resampled = self.oversampler.resample(X_train, y_train)
        
        return X_train_resampled, X_test, y_train_resampled, y_test


def create_default_resampled_splitter() -> SplitAndResampleStrategy:
    """
    Creates a pre-configured split-and-resample pipeline matching the notebook behavior:
    - Stratified splitting (80% train, 20% test)
    - SMOTENC oversampling on the training set
    """
    splitter = StratifiedTrainTestSplitStrategy(test_size=0.2, random_state=42)
    continuous_cols = ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age']
    oversampler = SMOTENCOversampler(continuous_cols=continuous_cols, random_state=42)
    return SplitAndResampleStrategy(splitter=splitter, oversampler=oversampler)