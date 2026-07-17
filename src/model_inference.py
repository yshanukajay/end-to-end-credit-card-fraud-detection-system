import json
import logging
import os
import joblib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Resolve relative paths against project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_binning import CustomBinningStrategy
from src.feature_encoding import OrdinalEncodingStrategy
from src.feature_scaling import StandardScalingStrategy, MinMaxScalingStrategy
from utils.logger import get_logger
from utils.config import (
    get_config,
    get_model_config,
    get_binning_config,
    get_encoding_config,
    get_scaling_config,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


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
    Enhanced model inference class supporting both pandas and PySpark preprocessing.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_spark: bool = False, spark: Optional[SparkSession] = None):
        """
        Initialize the model inference system.
        
        Args:
            model_path: Path to the trained model file
            use_spark: Whether to use PySpark for preprocessing
            spark: Optional SparkSession instance
        """
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        self.config = get_config()
        self.use_spark = use_spark
        
        if use_spark:
            from utils.spark_session import get_or_create_spark_session
            self.spark = spark or get_or_create_spark_session()
        else:
            self.spark = None
            
        # Determine model path
        if model_path is None:
            model_cfg = get_model_config()
            model_path = model_cfg.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl')
            
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
            
        self.model_path = model_path
        self.encoders = {}
        self.model = None
        self.scaler = None
        self.scaler_columns = []
        
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Processing Engine: {'PySpark' if use_spark else 'Pandas'}")
        
        try:
            # Load model and configurations
            self.load_model()
            self.binning_config = get_binning_config()
            self.encoding_config = get_encoding_config()
            self.scaling_config = get_scaling_config()
            
            # Load scaler if available
            scale_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'scale')
            self.load_scaler(scale_dir)
            
            # Load thresholds metadata
            metadata_path = self.model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.decision_threshold = self.metadata.get('decision_threshold', 0.5)
                self.review_threshold = self.metadata.get('review_threshold', 0.2)
                logger.info(f"✓ Loaded decision threshold: {self.decision_threshold:.4f}, review threshold: {self.review_threshold:.4f}")
            else:
                self.decision_threshold = 0.5
                self.review_threshold = 0.2
                logger.warning("⚠ Threshold metadata not found, using default 0.5 decision / 0.2 review thresholds")
            
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
        Load the trained model from disk with validation.
        """
        logger.info(f"Loading trained model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.error(f"✗ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            file_size = os.path.getsize(self.model_path) / (1024**2)  # MB
            
            logger.info(f"✓ Model loaded successfully:")
            logger.info(f"  • Model Type: {type(self.model).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_encoders(self, encoders_dir: str) -> None:
        """
        Load feature encoders from directory with validation and logging.
        """
        if not os.path.isabs(encoders_dir):
            encoders_dir = os.path.abspath(os.path.join(PROJECT_ROOT, encoders_dir))
            
        logger.info(f"Loading encoders from: {encoders_dir}")
        if not os.path.exists(encoders_dir):
            logger.warning(f"⚠ Encoders directory not found: {encoders_dir}. Using defaults.")
            return
        
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            
            if not encoder_files:
                logger.warning("⚠ No encoder files found in directory")
                return
            
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
    
    def load_scaler(self, scaler_dir: str = 'artifacts/scale') -> None:
        """
        Load the fitted scaler for inference.
        """
        if not os.path.isabs(scaler_dir):
            scaler_dir = os.path.abspath(os.path.join(PROJECT_ROOT, scaler_dir))
            
        logger.info(f"Loading scaler from: {scaler_dir}")
        try:
            metadata_path = os.path.join(scaler_dir, 'scaling_metadata.json')
            
            if not os.path.exists(metadata_path):
                logger.warning("⚠ Scaler metadata not found - scaling will not be applied")
                return
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # For pandas/sklearn inference, create an sklearn scaler from metadata
            if not self.use_spark:
                if metadata.get('scaling_type') == 'standard':
                    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
                    self.scaler = SklearnStandardScaler()
                    self.scaler.mean_ = np.array(metadata['mean'])
                    self.scaler.scale_ = np.array(metadata['std'])
                    self.scaler.var_ = self.scaler.scale_ ** 2
                    self.scaler.n_features_in_ = metadata['n_features']
                    self.scaler_columns = metadata['columns_to_scale']
                    logger.info(f"✓ Loaded sklearn StandardScaler from PySpark metadata for columns: {self.scaler_columns}")
                elif metadata.get('scaling_type') == 'minmax':
                    self.scaler = MinMaxScaler()
                    self.scaler.n_features_in_ = metadata['n_features']
                    self.scaler.data_min_ = np.array(metadata['data_min'])
                    self.scaler.data_max_ = np.array(metadata['data_max'])
                    self.scaler.data_range_ = self.scaler.data_max_ - self.scaler.data_min_
                    self.scaler.scale_ = 1.0 / self.scaler.data_range_
                    self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
                    self.scaler_columns = metadata['columns_to_scale']
                    logger.info(f"✓ Loaded sklearn MinMaxScaler from PySpark metadata for columns: {self.scaler_columns}")
            else:
                # Load PySpark Scaler Strategy
                if metadata.get('scaling_type') == 'standard':
                    self.scaler = StandardScalingStrategy(spark=self.spark)
                else:
                    self.scaler = MinMaxScalingStrategy(spark=self.spark)
                
                self.scaler.load_scaler(scaler_dir)
                self.scaler_columns = metadata['columns_to_scale']
                logger.info(f"✓ Loaded PySpark Scaler for columns: {self.scaler_columns}")
                
        except Exception as e:
            logger.warning(f"⚠ Failed to load scaler: {str(e)} - scaling will not be applied")

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for model prediction with comprehensive logging.
        """
        if self.use_spark:
            return self.preprocess_input_spark(data)
        else:
            return self.preprocess_input_pandas(data)

    def preprocess_input_spark(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data using PySpark.
        """
        logger.info("Preprocessing input data using PySpark...")
        try:
            spark_df = self.spark.createDataFrame([data])
            
            # Map standard raw names if present
            raw_rename = {
                'amt': 'amount',
                'category': 'merchant_category',
                'city_pop': 'city_population',
                'unix_time': 'transaction_unix_time'
            }
            for k, v in raw_rename.items():
                if k in spark_df.columns:
                    spark_df = spark_df.withColumnRenamed(k, v)
            
            # Preprocess features using spark_utils pipeline logic
            from utils.spark_utils import preprocess_credit_card_data
            spark_df_engineered = preprocess_credit_card_data(spark_df)
            
            # Apply scaling if available
            if hasattr(self, 'scaler') and self.scaler is not None:
                spark_df_engineered = self.scaler.transform(spark_df_engineered, self.scaler_columns)
                
            # One-hot encoding of categorical variables
            # For category & gender, map index mappings saved by StringIndexer
            for col, encoder_data in self.encoders.items():
                if col in spark_df_engineered.columns:
                    categories = encoder_data['categories']
                    for category in categories:
                        new_col_name = f"{col}_{category}"
                        spark_df_engineered = spark_df_engineered.withColumn(
                            new_col_name,
                            F.when(F.col(col) == category, 1).otherwise(0)
                        )
                    spark_df_engineered = spark_df_engineered.drop(col)
            
            # Expected final features
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
            
            # Ensure all expected columns are present, fill with 0 if missing
            for col in expected_order:
                if col not in spark_df_engineered.columns:
                    spark_df_engineered = spark_df_engineered.withColumn(col, F.lit(0))
                    
            spark_df_final = spark_df_engineered.select(expected_order)
            
            # Convert to Pandas
            from utils.spark_utils import spark_to_pandas
            pandas_df = spark_to_pandas(spark_df_final)
            return pandas_df
            
        except Exception as e:
            logger.error(f"✗ Spark preprocessing failed: {str(e)}")
            raise

    def preprocess_input_pandas(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data using pandas (Standard/sklearn flow).
        """
        logger.info("Preprocessing input data using pandas...")
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
            age = df['customer_age'].iloc[0]
            if age <= 25:
                df['customer_age_binned'] = 0
            elif age <= 50:
                df['customer_age_binned'] = 1
            elif age <= 75:
                df['customer_age_binned'] = 2
            else:
                df['customer_age_binned'] = 3

            hour = df['transaction_hour'].iloc[0]
            if hour >= 22 or hour <= 6:
                df['transaction_hour_binned'] = 0
            elif hour <= 12:
                df['transaction_hour_binned'] = 1
            elif hour <= 17:
                df['transaction_hour_binned'] = 2
            else:
                df['transaction_hour_binned'] = 3

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
            if 'merchant_category' not in df.columns:
                df['merchant_category'] = 'travel'
            category_val = str(df['merchant_category'].iloc[0]).lower().strip().replace(' ', '_').replace('&', '_').replace('/', '_')
            
            categories = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]
            for cat in categories:
                df[f"merchant_category_{cat}"] = int(category_val == cat)

            if 'gender' not in df.columns:
                df['gender'] = 'F'
            gender_val = str(df['gender'].iloc[0]).upper().strip()
            for gen in ["F", "M"]:
                df[f"gender_{gen}"] = int(gender_val == gen)

            # 6. Feature Scaling
            if hasattr(self, 'scaler') and self.scaler is not None:
                cols = self.scaler_columns if hasattr(self, 'scaler_columns') and self.scaler_columns else [
                    'amount_log', 
                    'velocity_last_24h_log', 
                    'city_population_log'
                ]
                if hasattr(self.scaler, 'n_features_in_'):  # Sklearn Scaler loaded from metadata
                    df[cols] = self.scaler.transform(df[cols])
                else:  # Legacy Strategy class if loaded
                    df = self.scaler.transform(df, cols)

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
            return df
            
        except Exception as e:
            logger.error(f"✗ Pandas preprocessing failed: {str(e)}")
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
            
            # Align feature column order to match model's training schema
            if hasattr(self.model, "feature_names_in_"):
                processed_data = processed_data[list(self.model.feature_names_in_)]
            
            prob = 0.0
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(processed_data)[0][1])
            
            # Apply tiered decision system:
            # - auto-approve: prob < self.review_threshold
            # - manual review: self.review_threshold <= prob < self.decision_threshold
            # - auto-block: prob >= self.decision_threshold
            if prob >= self.decision_threshold:
                pred = 1
                status = 'Fraud'
                action = 'Auto-Block'
                confidence = round(prob * 100, 2)
            elif prob >= self.review_threshold:
                pred = 1 # Flagged for review
                status = 'Fraud'
                action = 'Manual Review'
                confidence = round(prob * 100, 2)
            else:
                pred = 0
                status = 'Legitimate'
                action = 'Auto-Approve'
                confidence = round((1 - prob) * 100, 2)
            
            result = {
                "Prediction": int(pred),
                "Status": status,
                "Action": action,
                "Probability": float(prob),
                "Confidence": f"{confidence}%"
            }
            
            logger.info(f"✓ Prediction status: {status} (Probability: {prob:.4f}, Confidence: {confidence}%)")
            logger.info(f"{'='*60}\n")
            return result
        except Exception as e:
            logger.error(f"✗ Prediction run failed: {str(e)}")
            raise