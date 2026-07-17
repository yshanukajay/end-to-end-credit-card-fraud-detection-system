import os
import sys
import json
import pandas as pd
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.model_inference import ModelInference
from utils.config import get_model_config, get_inference_config
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
import mlflow


class InferenceTracker:
    """Track inference requests and model performance for MLflow."""
    
    def __init__(self):
        self.predictions_batch = []
        self.batch_size = 100  # Log every 100 predictions
        self.mlflow_tracker = None
        self.current_run = None
        
    def start_inference_run(self):
        """Start an MLflow run for inference tracking."""
        try:
            self.mlflow_tracker = MLflowTracker()
            run_tags = create_mlflow_run_tags('inference_pipeline', {
                'inference_type': 'streaming',
                'batch_size': str(self.batch_size)
            })
            self.current_run = self.mlflow_tracker.start_run(
                run_name='streaming_inference', tags=run_tags
            )
            logger.info("✓ Inference tracking run started")
        except Exception as e:
            logger.error(f"✗ Failed to start inference tracking: {str(e)}")
    
    def track_prediction(self, input_data: Dict[str, Any], prediction_result: Dict[str, str], 
                        inference_time: float):
        """Track individual prediction with metadata."""
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'input_data': input_data,
                'prediction': prediction_result,
                'inference_time_ms': inference_time * 1000,
                'fraud_probability': prediction_result.get('Probability', 0.0),
                'predicted_class': 1 if prediction_result.get('Prediction', 0) == 1 else 0
            }
            
            self.predictions_batch.append(prediction_record)
            
            # Log batch when it reaches the specified size
            if len(self.predictions_batch) >= self.batch_size:
                self._log_prediction_batch()
                
        except Exception as e:
            logger.error(f"✗ Failed to track prediction: {str(e)}")
    
    def _log_prediction_batch(self):
        """Log a batch of predictions to MLflow."""
        try:
            if not self.predictions_batch or not self.mlflow_tracker:
                return
            
            # Calculate batch statistics
            batch_stats = self._calculate_batch_stats()
            
            # Log batch metrics
            mlflow.log_metrics(batch_stats['metrics'])
            
            # Save batch predictions as artifact
            batch_file = f"inference_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            batch_path = os.path.join(PROJECT_ROOT, 'artifacts', 'inference_batches', batch_file)
            os.makedirs(os.path.dirname(batch_path), exist_ok=True)
            
            with open(batch_path, 'w') as f:
                json.dump(self.predictions_batch, f, indent=2, default=str)
            
            mlflow.log_artifact(batch_path, "inference_batches")
            
            logger.info(f"✓ Logged batch of {len(self.predictions_batch)} predictions")
            
            # Clear batch
            self.predictions_batch = []
            
        except Exception as e:
            logger.error(f"✗ Failed to log prediction batch: {str(e)}")
    
    def _calculate_batch_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the current batch."""
        if not self.predictions_batch:
            return {'metrics': {}}
        
        # Extract key metrics
        inference_times = [p['inference_time_ms'] for p in self.predictions_batch]
        fraud_probs = [p['fraud_probability'] for p in self.predictions_batch]
        predicted_classes = [p['predicted_class'] for p in self.predictions_batch]
        
        metrics = {
            f'batch_size': len(self.predictions_batch),
            f'avg_inference_time_ms': np.mean(inference_times),
            f'max_inference_time_ms': np.max(inference_times),
            f'min_inference_time_ms': np.min(inference_times),
            f'avg_fraud_probability': np.mean(fraud_probs),
            f'std_fraud_probability': np.std(fraud_probs),
            f'fraud_predictions_count': np.sum(predicted_classes),
            f'legitimate_predictions_count': len(predicted_classes) - np.sum(predicted_classes),
            f'high_risk_predictions': np.sum([p > 0.7 for p in fraud_probs]),
            f'medium_risk_predictions': np.sum([(p > 0.5) & (p <= 0.7) for p in fraud_probs]),
            f'low_risk_predictions': np.sum([p <= 0.5 for p in fraud_probs])
        }
        
        return {'metrics': metrics}
    
    def end_inference_run(self):
        """End the inference tracking run."""
        try:
            # Log any remaining predictions
            if self.predictions_batch:
                self._log_prediction_batch()
            
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
                logger.info("✓ Inference tracking run ended")
        except Exception as e:
            logger.error(f"✗ Failed to end inference tracking: {str(e)}")


# Global inference tracker
inference_tracker = InferenceTracker()


def initialize_inference_system(
    model_path: str = 'artifacts/models/xgboost_tuned_model.pkl',
    encoders_path: str = 'artifacts/encode'
) -> ModelInference:
    """
    Initialize the inference system with comprehensive logging and error handling.
    
    Args:
        model_path: Path to the trained model
        encoders_path: Path to the encoders directory
        
    Returns:
        Initialized ModelInference instance
        
    Raises:
        Exception: If initialization fails
    """
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING STREAMING INFERENCE SYSTEM")
    logger.info(f"{'='*80}")
    
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.isabs(encoders_path):
        encoders_path = os.path.join(PROJECT_ROOT, encoders_path)

    try:
        # Initialize model inference
        logger.info("Creating ModelInference instance...")
        inference = ModelInference(model_path)
        
        # Load encoders if directory exists
        if os.path.exists(encoders_path):
            logger.info(f"Loading encoders from: {encoders_path}")
            inference.load_encoders(encoders_path)
        else:
            logger.warning(f"⚠ Encoders directory not found: {encoders_path}")
            logger.info("Proceeding without encoders (may affect prediction accuracy)")
        
        logger.info("✓ Streaming inference system initialized successfully")
        
        # Start inference tracking
        inference_tracker.start_inference_run()
        
        logger.info(f"{'='*80}\n")
        
        return inference
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize inference system: {str(e)}")
        raise


def streaming_inference(
    inference: ModelInference, 
    data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Perform streaming inference with comprehensive logging and error handling.
    
    Args:
        inference: Initialized ModelInference instance
        data: Input data dictionary for prediction
        
    Returns:
        Prediction result dictionary
        
    Raises:
        ValueError: If input parameters are invalid
        Exception: For any prediction errors
    """
    logger.info(f"\n{'='*70}")
    logger.info("STREAMING INFERENCE REQUEST")
    logger.info(f"{'='*70}")
    
    # Input validation
    if inference is None:
        logger.error("✗ ModelInference instance cannot be None")
        raise ValueError("ModelInference instance cannot be None")
    
    if not data or not isinstance(data, dict):
        logger.error("✗ Input data must be a non-empty dictionary")
        raise ValueError("Input data must be a non-empty dictionary")
    
    try:
        logger.info("Processing inference request...")
        logger.info(f"Input data keys: {list(data.keys())}")
        
        # Time the inference
        start_time = time.time()
        
        # Perform prediction
        prediction_result = inference.predict(data)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Track the prediction
        inference_tracker.track_prediction(data, prediction_result, inference_time)
        
        logger.info("✓ Streaming inference completed successfully")
        logger.info(f"Result: {prediction_result}")
        logger.info(f"Inference time: {inference_time*1000:.2f}ms")
        logger.info(f"{'='*70}\n")
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"✗ Streaming inference failed: {str(e)}")
        raise


# Initialize the global inference system
try:
    logger.info("Initializing global inference system...")
    model_config = get_model_config()
    model_path = model_config.get('model_path', 'artifacts/models/xgboost_tuned_model.pkl')
    inference = initialize_inference_system(model_path=model_path)
except Exception as e:
    logger.error(f"Failed to initialize global inference system: {str(e)}")
    inference = None


if __name__ == '__main__':
    # Sample transaction payload for Credit Card Fraud Detection (Raw transaction)
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
        print("\nPrediction Results:")
        print(pred)
        
        # Stop tracker
        inference_tracker.end_inference_run()
    except Exception as e:
        logger.error(f"Streaming inference execution failed: {e}")
        sys.exit(1)