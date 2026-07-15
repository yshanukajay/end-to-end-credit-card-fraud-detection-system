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

from utils.config import (
    get_config,
    get_model_config,
    get_binning_config,
    get_encoding_config,
    get_scaling_config,
)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


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
        self.config = get_config()
        
        # Determine model path
        if model_path is None:
            model_cfg = get_model_config()
            model_path = model_cfg.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl')
            
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
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            self.scaling_config = get_scaling_config()
            
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
            return
            
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(encoders_dir, file)
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                logger.info(f"  ✓ Loaded encoder for '{feature_name}'")
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
            df = pd.DataFrame([data])
            
            # Map standard raw names if present
            raw_rename = {
                'amt': 'amount',
                'category': 'merchant_category',
                'city_pop': 'city_population'
            }
            df = df.rename(columns={k: v for k, v in raw_rename.items() if k in df.columns})

            # Ensure all standard baseline values are present or fallback
            if 'amount' not in df.columns:
                df['amount'] = 0.0
            if 'city_population' not in df.columns:
                df['city_population'] = 1000.0
            if 'velocity_last_24h' not in df.columns:
                df['velocity_last_24h'] = 0.0

            # 1. Feature Engineering: Datetimes & Age
            if 'trans_date_trans_time' in df.columns:
                trans_dt = pd.to_datetime(df['trans_date_trans_time'])
                df['day_of_week'] = trans_dt.dt.dayofweek.astype(int)
                df['is_weekend'] = trans_dt.dt.dayofweek.isin([5, 6]).astype(int)
                df['transaction_hour'] = trans_dt.dt.hour.astype(int)
            else:
                df['day_of_week'] = 0
                df['is_weekend'] = 0
                df['transaction_hour'] = 12

            if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
                trans_dt = pd.to_datetime(df['trans_date_trans_time'])
                dob_dt = pd.to_datetime(df['dob'])
                df['customer_age'] = ((trans_dt - dob_dt).dt.days // 365).astype(int)
            else:
                df['customer_age'] = 35

            # 2. Distance Calculations
            if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
                df['distance_to_merchant'] = haversine_distance(
                    df['lat'].iloc[0], df['long'].iloc[0],
                    df['merch_lat'].iloc[0], df['merch_long'].iloc[0]
                )
                df['location_mismatch'] = (df['distance_to_merchant'] > 80).astype(int)
                df['foreign_transaction'] = (df['distance_to_merchant'] > 150).astype(int)
            else:
                df['distance_to_merchant'] = 10.0
                df['location_mismatch'] = 0
                df['foreign_transaction'] = 0

            # 3. Log Transformations
            df['amount_log'] = np.log1p(df['amount'])
            df['velocity_last_24h_log'] = np.log1p(df['velocity_last_24h'])
            df['city_population_log'] = np.log1p(df['city_population'])

            # 4. Binning Mappings
            # customer_age_binned
            age = df['customer_age'].iloc[0]
            if age <= 25:
                df['customer_age_binned'] = 0
            elif age <= 50:
                df['customer_age_binned'] = 1
            elif age <= 75:
                df['customer_age_binned'] = 2
            else:
                df['customer_age_binned'] = 3

            # transaction_hour_binned
            hour = df['transaction_hour'].iloc[0]
            if hour >= 22 or hour <= 6:
                df['transaction_hour_binned'] = 0
            elif hour <= 12:
                df['transaction_hour_binned'] = 1
            elif hour <= 17:
                df['transaction_hour_binned'] = 2
            else:
                df['transaction_hour_binned'] = 3

            # distance_to_merchant_binned
            dist = df['distance_to_merchant'].iloc[0]
            if dist <= 10:
                df['distance_to_merchant_binned'] = 0
            elif dist <= 50:
                df['distance_to_merchant_binned'] = 1
            elif dist <= 100:
                df['distance_to_merchant_binned'] = 2
            else:
                df['distance_to_merchant_binned'] = 3

            # 5. One-hot Nominals Encoding
            # Merchant Category Encoding
            if 'merchant_category' not in df.columns:
                df['merchant_category'] = 'travel'
            category_val = str(df['merchant_category'].iloc[0]).lower().strip().replace(' ', '_').replace('&', '_').replace('/', '_')
            
            categories = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]
            for cat in categories:
                df[f"merchant_category_{cat}"] = int(category_val == cat)

            # Gender Encoding
            if 'gender' not in df.columns:
                df['gender'] = 'F'
            gender_val = str(df['gender'].iloc[0]).upper().strip()
            for gen in ["F", "M"]:
                df[f"gender_{gen}"] = int(gender_val == gen)

            # 6. Feature Scaling
            if hasattr(self, 'scaler') and self.scaler.fitted:
                logger.info("Applying feature scaling...")
                columns_to_scale = [
                    'amount', 'amount_log', 
                    'velocity_last_24h', 'velocity_last_24h_log', 
                    'city_population', 'city_population_log'
                ]
                df = self.scaler.transform(df, columns_to_scale)
            else:
                logger.warning("⚠ Scaler not loaded or fitted. Scaling skipped.")

            # 7. Order & Expected final feature set (29 columns)
            expected_order = [
                "day_of_week", "is_weekend", "location_mismatch", "velocity_last_24h", 
                "foreign_transaction", "amount", "city_population", "amount_log", 
                "velocity_last_24h_log", "city_population_log", "customer_age_binned", 
                "transaction_hour_binned", "distance_to_merchant_binned", 
                "merchant_category_entertainment", "merchant_category_food_dining", 
                "merchant_category_gas_transport", "merchant_category_grocery_net", 
                "merchant_category_grocery_pos", "merchant_category_health_fitness", 
                "merchant_category_home", "merchant_category_kids_pets", 
                "merchant_category_misc_net", "merchant_category_misc_pos", 
                "merchant_category_personal_care", "merchant_category_shopping_net", 
                "merchant_category_shopping_pos", "merchant_category_travel", 
                "gender_F", "gender_M"
            ]

            df = df[expected_order]
            logger.info(f"✓ Preprocessed shape: {df.shape}")
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