import os
import sys
import json
import yaml
import time
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator

# Resolve relative paths against project root so imports and config loading
# work regardless of which working directory the script is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_binning import CustomBinningStrategy
from src.feature_encoding import OrdinalEncodingStrategy
from src.feature_scaling import StandardScalingStrategy
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ModelInference:
    """
    Handles end-to-end model inference: input preprocessing, feature binning,
    encoding, scaling, and predicting fraud probability using the trained model.
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model inference system.
        """
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        # Load config
        self.config = load_config()
        
        # Determine model path
        if model_path is None:
            model_cfg = self.config.get('model', {})
            model_path = model_cfg.get('model_path', 'artifacts/model/random_forest_cv_model.pkl')
            
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
            
        self.model_path = model_path
        self.encoders = {}
        self.model = None
        
        logger.info(f"Model Path: {self.model_path}")
        
        try:
            # Load model
            self.load_model()
            
            # Load config sections
            self.binning_config = self.config.get('feature_binning', {})
            self.encoding_config = self.config.get('feature_encoding', {})
            self.scaling_config = self.config.get('feature_scaling', {})
            
            # Initialize and load standard scaler
            self.scaler = StandardScalingStrategy()
            scale_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'scale')
            scaler_loaded = self.scaler.load_scaler(scale_dir)
            if not scaler_loaded:
                logger.warning("⚠ Could not load scaler artifacts. Feature scaling will be skipped during inference.")
            
            # Load categorical encoders
            encode_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'encode')
            self.load_encoders(encode_dir)
            
            logger.info("✓ Model inference system initialized successfully")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize model inference: {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Load the trained model from disk.
        """
        logger.info(f"Loading trained model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.error(f"✗ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.model = joblib.load(self.model_path)
            file_size = os.path.getsize(self.model_path) / (1024**2)  # MB
            logger.info("✓ Model loaded successfully:")
            logger.info(f"  • Model Type: {type(self.model).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} MB")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_encoders(self, encoders_dir: str) -> None:
        """
        Load feature encoder configurations from directory.
        """
        logger.info(f"Loading encoders from: {encoders_dir}")
        if not os.path.exists(encoders_dir):
            logger.warning(f"⚠ Encoders directory not found: {encoders_dir}. Using defaults.")
            # Default categories for fallback
            self.encoders['merchant_category'] = {
                'categories': ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel'],
                'encoding_type': 'one_hot'
            }
            return
            
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            if not encoder_files:
                logger.warning("⚠ No encoder files found. Using default categories.")
                self.encoders['merchant_category'] = {
                    'categories': ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel'],
                    'encoding_type': 'one_hot'
                }
                return
                
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(encoders_dir, file)
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                logger.info(f"  ✓ Loaded encoder for '{feature_name}': {encoder_data.get('categories', [])}")
        except Exception as e:
            logger.error(f"✗ Failed to load encoders: {str(e)}")
            raise

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocesses a raw input sample dict into the format expected by the model.
        """
        logger.info(f"\n{'='*50}")
        logger.info("PREPROCESSING INFERENCE INPUT")
        logger.info(f"{'='*50}")
        
        if not data:
            raise ValueError("Input data cannot be empty.")
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            logger.info(f"Raw input data keys: {list(df.columns)}")
            
            # 1. Feature Binning (device_trust_score)
            if 'device_trust_score' in df.columns:
                logger.info("Applying feature binning for device_trust_score...")
                device_trust_bins = {
                    "Poor": [0.0, 25.0],
                    "Fair": [25.0, 50.0],
                    "Good": [50.0, 75.0],
                    "Excellent": [75.0, 100.0]
                }
                bin_strategy = CustomBinningStrategy(bin_definitions=device_trust_bins)
                df = bin_strategy.bin_feature(df, 'device_trust_score')
            else:
                logger.warning("⚠ 'device_trust_score' not found in input data. Binning skipped.")
                
            # 2. Feature Encoding (one-hot & ordinal)
            logger.info("Applying feature encoding...")
            
            # Nominal/One-hot encoding of merchant_category
            col = 'merchant_category'
            if col in df.columns:
                original_value = df[col].iloc[0]
                encoder_data = self.encoders.get(col, {})
                categories = encoder_data.get('categories', ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel'])
                
                for cat in categories:
                    new_col_name = f"{col}_{cat}"
                    df[new_col_name] = (df[col] == cat).astype(int)
                    
                df = df.drop(columns=[col])
                logger.info(f"  ✓ One-hot encoded '{col}': {original_value} → columns generated for {categories}")
            else:
                # If merchant category columns are already present in mock or missing
                logger.warning("⚠ 'merchant_category' not found in input data.")
                
            # Ordinal mapping of device_trust_score_binned
            ordinal_mappings = {
                "device_trust_score_binned": {
                    "Poor": 0,
                    "Fair": 1,
                    "Good": 2,
                    "Excellent": 3
                }
            }
            ord_strategy = OrdinalEncodingStrategy(ordinal_mappings=ordinal_mappings)
            df = ord_strategy.encode(df)
            
            # 3. Feature Scaling
            if hasattr(self, 'scaler') and self.scaler.fitted:
                logger.info("Applying feature scaling...")
                columns_to_scale = self.scaling_config.get('columns_to_scale', ['amount', 'transaction_hour', 'velocity_last_24h', 'cardholder_age'])
                df = self.scaler.transform(df, columns_to_scale)
            else:
                logger.warning("⚠ Scaler not loaded or fitted. Scaling skipped.")
                
            # 4. Drop unnecessary columns (like transaction_id)
            if 'transaction_id' in df.columns:
                df = df.drop(columns=['transaction_id'])
                
            # Drop target if present
            if 'is_fraud' in df.columns:
                df = df.drop(columns=['is_fraud'])
                
            # 5. Reorder features to match exact expectation of model
            expected_order = [
                'amount', 'transaction_hour', 'foreign_transaction', 'location_mismatch',
                'velocity_last_24h', 'cardholder_age', 'device_trust_score_binned',
                'merchant_category_Clothing', 'merchant_category_Electronics',
                'merchant_category_Food', 'merchant_category_Grocery', 'merchant_category_Travel'
            ]
            
            # Fill missing columns with 0 if any encoding didn't generate them
            for col in expected_order:
                if col not in df.columns:
                    df[col] = 0
                    
            df = df[expected_order]
            logger.info(f"✓ Preprocessed shape: {df.shape}")
            logger.info(f"Final features: {list(df.columns)}")
            logger.info(f"{'='*50}\n")
            
            return df
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            raise

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud status and probability for an input data dict.
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL INFERENCE PREDICTION")
        logger.info(f"{'='*60}")
        
        if self.model is None:
            raise ValueError("Model has not been loaded successfully.")
            
        try:
            # Preprocess the sample
            processed_data = self.preprocess_input(data)
            
            # Predict
            pred = self.model.predict(processed_data)[0]
            
            prob = 0.0
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(processed_data)[0][1]
                
            status = 'Fraud' if pred == 1 else 'Legitimate'
            confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)
            
            result = {
                "Prediction": int(pred),
                "Status": status,
                "Probability": float(prob),
                "Confidence": f"{confidence}%"
            }
            
            logger.info(f"✓ Prediction status: {status} (Probability: {prob:.4f}, Confidence: {confidence}%)")
            logger.info(f"{'='*60}\n")
            return result
        except Exception as e:
            logger.error(f"✗ Prediction run failed: {str(e)}")
            raise