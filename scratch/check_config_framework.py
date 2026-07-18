import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import get_config, get_model_config

config = get_config()
model_cfg = get_model_config()

print("Framework from get_config():", config.get('model', {}).get('framework'))
print("Framework from get_model_config():", model_cfg.get('framework'))
print("Model Path from get_model_config():", model_cfg.get('model_path'))
