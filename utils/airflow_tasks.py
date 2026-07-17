import os
import sys
import logging
from typing import Dict, Any

# Resolve project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def validate_input_data() -> None:
    """Validate that raw data file exists and is not empty."""
    from utils.config import get_data_paths
    
    logger.info("Validating input raw data...")
    data_paths = get_data_paths()
    raw_data_path = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
    
    if not os.path.exists(raw_data_path):
        logger.error(f"✗ Raw data file not found at {raw_data_path}")
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    
    if os.path.getsize(raw_data_path) == 0:
        logger.error(f"✗ Raw data file at {raw_data_path} is empty")
        raise ValueError(f"Raw data file at {raw_data_path} is empty")
        
    logger.info(f"✓ Input data is valid. File size: {os.path.getsize(raw_data_path)} bytes.")


def run_data_pipeline() -> None:
    """Run data preprocessing pipeline."""
    from pipelines.data_pipeline import data_pipeline
    from utils.config import get_data_paths
    
    logger.info("Starting data pipeline execution...")
    data_paths = get_data_paths()
    raw_data = os.path.join(PROJECT_ROOT, data_paths.get('raw_data', 'dataset/raw/fraudTrain.csv'))
    
    # We pass force_rebuild=True to ensure pipeline runs preprocessing logic from scratch.
    data_pipeline(data_path=raw_data, force_rebuild=True)
    logger.info("✓ Data pipeline completed successfully.")


def validate_processed_data() -> None:
    """Validate processed training data splits exist and are non-empty."""
    from utils.config import get_data_paths
    
    logger.info("Validating processed data splits...")
    data_paths = get_data_paths()
    required_keys = ['X_train', 'X_test', 'Y_train', 'Y_test']
    
    for key in required_keys:
        filepath = os.path.join(PROJECT_ROOT, data_paths.get(key))
        if not os.path.exists(filepath):
            logger.error(f"✗ Processed dataset '{key}' not found at {filepath}")
            raise FileNotFoundError(f"Processed dataset '{key}' not found at {filepath}")
        if os.path.getsize(filepath) == 0:
            logger.error(f"✗ Processed dataset '{key}' at {filepath} is empty")
            raise ValueError(f"Processed dataset '{key}' at {filepath} is empty")
            
    logger.info("✓ All processed data splits are valid and present.")


def run_training_pipeline() -> None:
    """Run machine learning model training pipeline."""
    from pipelines.train_pipeline import training_pipeline
    
    logger.info("Starting machine learning training pipeline...")
    training_pipeline()
    logger.info("✓ Model training pipeline completed successfully.")


def validate_trained_model() -> None:
    """Validate that the trained model exists and is not empty."""
    from utils.config import get_model_config
    
    logger.info("Validating trained model...")
    model_cfg = get_model_config()
    model_path = os.path.join(PROJECT_ROOT, model_cfg.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl'))
    
    if not os.path.exists(model_path):
        logger.error(f"✗ Trained model not found at {model_path}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")
        
    if os.path.isdir(model_path):
        if not os.listdir(model_path):
            logger.error(f"✗ Trained model directory at {model_path} is empty")
            raise ValueError(f"Trained model directory at {model_path} is empty")
        logger.info(f"✓ Trained model is valid. Directory contains: {os.listdir(model_path)}")
    else:
        if os.path.getsize(model_path) == 0:
            logger.error(f"✗ Trained model file at {model_path} is empty")
            raise ValueError(f"Trained model file at {model_path} is empty")
        logger.info(f"✓ Trained model is valid. File size: {os.path.getsize(model_path)} bytes.")


def run_inference_pipeline() -> None:
    """Run model inference tracking on sample data."""
    from pipelines.streaming_inference_pipeline import initialize_inference_system, streaming_inference, inference_tracker
    from utils.config import get_model_config
    
    logger.info("Starting model inference pipeline execution...")
    model_config = get_model_config()
    model_path = model_config.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl')
    
    inference = initialize_inference_system(model_path=model_path)
    if inference is None:
        logger.error("✗ Failed to initialize model inference system")
        raise RuntimeError("Failed to initialize model inference system")
        
    # Sample transaction payload for testing
    data = {
        "trans_date_trans_time": "2020-06-21 12:14:00",
        "cc_num": 123456789012345,
        "amt": 450.75,
        "category": "travel",
        "gender": "F",
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000.0,
        "dob": "1988-12-15",
        "merch_lat": 40.7500,
        "merch_long": -73.9900,
        "velocity_last_24h": 5.0
    }
    
    try:
        pred = streaming_inference(inference, data)
        logger.info(f"✓ Inference sample run succeeded. Result: {pred}")
    finally:
        inference_tracker.end_inference_run()
