import os
import yaml
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config() -> dict:
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        return {}


def get_data_paths() -> dict:
    config = load_config()
    return config.get('data_paths', {})


def get_columns() -> dict:
    config = load_config()
    return config.get('columns', {})


def get_missing_values_config() -> dict:
    config = load_config()
    return config.get('missing_values', {})


def get_outlier_config() -> dict:
    config = load_config()
    return config.get('outlier_detection', {})


def get_binning_config() -> dict:
    config = load_config()
    return config.get('feature_binning', {})


def get_encoding_config() -> dict:
    config = load_config()
    return config.get('feature_encoding', {})


def get_scaling_config() -> dict:
    config = load_config()
    return config.get('feature_scaling', {})


def get_splitting_config() -> dict:
    config = load_config()
    return config.get('data_splitting', {})


def get_training_config() -> dict:
    config = load_config()
    return config.get('training', {})


def get_model_config() -> dict:
    config = load_config()
    return config.get('model', {})


def get_evaluation_config() -> dict:
    config = load_config()
    return config.get('evaluation', {})


def get_deployment_config() -> dict:
    config = load_config()
    return config.get('deployment', {})


def get_logging_config() -> dict:
    config = load_config()
    return config.get('logging', {})


def get_environment_config() -> dict:
    config = load_config()
    return config.get('environment', {})


def get_pipeline_config() -> dict:
    config = load_config()
    return config.get('pipeline', {})


def get_inference_config() -> dict:
    config = load_config()
    return config.get('inference', {})


# AWS S3 Configuration Functions
def get_aws_config() -> Dict[str, Any]:
    """Get AWS configuration from config.yaml"""
    config = load_config()
    aws_config = config.get('aws', {})
    
    # Fallback to environment variables if not in config.yaml
    return {
        'region': aws_config.get('region', os.getenv('AWS_REGION', 'ap-south-1')),
        'bucket': aws_config.get('s3_bucket', os.getenv('S3_BUCKET')),
        'kms_key_arn': aws_config.get('s3_kms_key_arn', os.getenv('S3_KMS_KEY_ARN')),
        'force_s3_io': aws_config.get('force_s3_io', os.getenv('FORCE_S3_IO', 'false').lower() in ('true', '1', 'yes'))
    }


def get_aws_region() -> str:
    """Get AWS region from config.yaml or environment variables"""
    aws_config = get_aws_config()
    return aws_config['region']


def get_s3_bucket() -> str:
    """Get S3 bucket name from config.yaml or environment variables (required)"""
    aws_config = get_aws_config()
    bucket = aws_config['bucket']
    if not bucket:
        raise ValueError(
            "S3 bucket is required. Please set 'aws.s3_bucket' in config.yaml "
            "or S3_BUCKET environment variable."
        )
    return bucket


def get_s3_kms_arn() -> Optional[str]:
    """Get S3 KMS key ARN from config.yaml or environment variables"""
    aws_config = get_aws_config()
    return aws_config['kms_key_arn']


def force_s3_io() -> bool:
    """Check if S3-only I/O is enforced from config.yaml or environment variables"""
    aws_config = get_aws_config()
    return aws_config['force_s3_io']


def get_mlflow_config() -> Dict[str, Any]:
    """Get MLflow configuration from config.yaml"""
    config = load_config()
    mlflow_config = config.get('mlflow', {})
    
    # Environment variables take priority over config.yaml
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI') or mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db'),
        'artifact_root': os.getenv('MLFLOW_DEFAULT_ARTIFACT_ROOT') or mlflow_config.get('artifact_root'),
        'experiment_name': mlflow_config.get('experiment_name', 'Credit Card Fraud Detection'),
        'artifact_location': mlflow_config.get('artifact_location')
    }


def get_s3_config() -> Dict[str, Any]:
    """Get complete S3 configuration (legacy function for compatibility)"""
    config = load_config()
    return config.get('aws', {})



def get_config() -> Dict[str, Any]:
    return load_config()


def get_data_config() -> Dict[str, Any]:
    return get_data_paths()


def get_preprocessing_config() -> Dict[str, Any]:
    return {
        'missing_values': get_missing_values_config(),
        'outlier_detection': get_outlier_config(),
        'feature_binning': get_binning_config(),
        'feature_encoding': get_encoding_config(),
        'feature_scaling': get_scaling_config()
    }


def get_selected_model_config() -> Dict[str, Any]:
    model_config = get_model_config()
    training_config = get_training_config()
    selected_model = model_config.get('model_type', training_config.get('default_model_type', 'random_forest'))
    model_types = model_config.get('model_types', {})
    
    return {
        'model_type': selected_model,
        'model_config': model_types.get(selected_model, {}),
        'training_strategy': model_config.get('training_strategy', training_config.get('default_training_strategy', 'cv')),
        'cv_folds': training_config.get('cv_folds', 5),
        'random_state': training_config.get('random_state', 42)
    }


def get_available_models() -> List[str]:
    model_config = get_model_config()
    return list(model_config.get('model_types', {}).keys())


def update_config(updates: Dict[str, Any]) -> None:
    config_path = CONFIG_FILE
    config = get_config()
    for key, value in updates.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def create_default_config() -> None:
    config_path = CONFIG_FILE
    if not os.path.exists(config_path):
        default_config = {
            'data_paths': {
                'raw_data': 'dataset/raw/fraudTrain.csv',
                'processed_data': 'dataset/processed/credit_card_fraud_null_handled.csv',
                'imputed_data': 'dataset/processed/credit_card_fraud_null_handled.csv',
                'processed_dir': 'dataset/processed',
                'artifacts_dir': 'artifacts',
                'data_artifacts_dir': 'artifacts/data',
                'model_artifacts_dir': 'artifacts/models',
                'X_train': 'artifacts/data/credit_card_fraud_X_train.npz',
                'X_test': 'artifacts/data/credit_card_fraud_X_test.npz',
                'Y_train': 'artifacts/data/credit_card_fraud_y_train.npz',
                'Y_test': 'artifacts/data/credit_card_fraud_y_test.npz'
            },
            'columns': {
                'target': 'is_fraud',
                'drop_columns': ['transaction_id'],
                'critical_columns': [],
                'outlier_columns': ['customer_age', 'distance_to_merchant'],
                'nominal_columns': ['merchant_category', 'gender'],
                'numeric_columns': ['amount', 'customer_age', 'distance_to_merchant', 'city_population']
            },
            'missing_values': {
                'strategy': 'fill',
                'methods': {
                    'amount': {'strategy': 'fill', 'method': 'median', 'relevant_column': 'amount'},
                    'customer_age': {'strategy': 'fill', 'method': 'median', 'relevant_column': 'customer_age'},
                    'distance_to_merchant': {'strategy': 'fill', 'method': 'median', 'relevant_column': 'distance_to_merchant'},
                    'city_population': {'strategy': 'fill', 'method': 'median', 'relevant_column': 'city_population'},
                    'merchant_category': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'merchant_category'},
                    'foreign_transaction': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'foreign_transaction'},
                    'location_mismatch': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'location_mismatch'},
                    'transaction_hour': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'transaction_hour'},
                    'velocity_last_24h': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'velocity_last_24h'},
                    'gender': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'gender'},
                    'day_of_week': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'day_of_week'},
                    'is_weekend': {'strategy': 'fill', 'method': 'mode', 'relevant_column': 'is_weekend'}
                }
            },
            'outlier_detection': {
                'detection_method': '3_sigma',
                'handling_method': 'cap',
                'z_score_threshold': 3.0
            },
            'feature_binning': {
                'customer_age_bins': {'Youth': [0, 25], 'Adult': [25, 50], 'Senior': [50, 75], 'Elderly': [75, 100]},
                'customer_age_mapping': {'Youth': 0, 'Adult': 1, 'Senior': 2, 'Elderly': 3}
            },
            'feature_encoding': {
                'nominal_columns': ['merchant_category', 'gender'],
                'ordinal_mappings': {
                    'customer_age_binned': {'Youth': 0, 'Adult': 1, 'Senior': 2, 'Elderly': 3}
                }
            },
            'feature_scaling': {
                'scaling_type': 'standard',
                'columns_to_scale': ['amount', 'customer_age', 'distance_to_merchant', 'city_population']
            },
            'data_splitting': {
                'split_type': 'stratified',
                'test_size': 0.2,
                'random_state': 42,
                'n_splits': 6
            },
            'training': {
                'default_model_type': 'xgboost',
                'default_training_strategy': 'cv',
                'cv_folds': 6,
                'random_state': 42
            }
        }
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        logger.info(f'Created default configuration file: {config_path}')


create_default_config()
