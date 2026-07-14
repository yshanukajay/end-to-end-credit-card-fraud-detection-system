import os
import sys
import json
from typing import Dict, Any

# Resolve relative paths against project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model_inference import ModelInference
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# Try to import mlflow to track experiments
try:
    import mlflow
    from utils.mlflow_utils import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠ mlflow package not found. MLflow tracking will be skipped.")

def run_streaming_inference(sample_path: str = "sample_transaction.json") -> Dict[str, Any]:
    """Run inference on a sample JSON transaction."""
    sample_full_path = os.path.join(PROJECT_ROOT, sample_path)
    
    # Create a default sample transaction if it doesn't exist
    if not os.path.exists(sample_full_path):
        logger.info(f"Creating default sample transaction at {sample_path}...")
        default_sample = {
            "amount": 450.75,
            "transaction_hour": 14,
            "merchant_category": "Travel",
            "foreign_transaction": 1,
            "location_mismatch": 1,
            "device_trust_score": 12.5,
            "velocity_last_24h": 5,
            "cardholder_age": 29
        }
        with open(sample_full_path, "w") as f:
            json.dump(default_sample, f, indent=4)
        logger.info("Default sample transaction created.")
        
    # Load sample transaction
    logger.info(f"Loading sample transaction from: {sample_full_path}")
    with open(sample_full_path, "r") as f:
        transaction = json.load(f)
        
    logger.info(f"Transaction data: {json.dumps(transaction, indent=2)}")
    
    # Run inference
    inference_system = ModelInference()
    result = inference_system.predict(transaction)
    
    # Print results nicely
    print("\n" + "=" * 60)
    print("[STREAMING INFERENCE RESULT]")
    print("=" * 60)
    print(f"Status      : {result.get('Status')}")
    print(f"Probability : {result.get('Probability'):.4f}")
    print(f"Confidence  : {result.get('Confidence')}")
    print("=" * 60 + "\n")
    
    # Log streaming inference metrics to MLflow
    if MLFLOW_AVAILABLE:
        try:
            tracker = MLflowTracker()
            tracker.start_run(run_name="streaming_inference")
            
            predictions = np.array([result.get('Prediction', 0)])
            probabilities = np.array([result.get('Probability', 0.0)])
            
            tracker.log_inference_metrics(
                predictions=predictions,
                probabilities=probabilities,
                input_data_info={
                    'inference_type': 'streaming',
                    'transaction_amount': float(transaction.get('amount', 0.0)),
                    'merchant_category': str(transaction.get('merchant_category', 'Unknown')),
                    'velocity_last_24h': int(transaction.get('velocity_last_24h', 0))
                }
            )
            tracker.end_run()
            logger.info("✓ Streaming inference metrics successfully logged to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log inference metrics to MLflow: {e}")
            
    return result

if __name__ == "__main__":
    try:
        run_streaming_inference()
    except Exception as e:
        logger.error(f"Streaming inference failed: {e}")
        sys.exit(1)
