import os
import joblib
from abc import ABC, abstractmethod
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class BaseModelBuilder(ABC):
    """
    Abstract Base Class for model builders.
    Defines the contract for building, saving, and loading classifier models.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        logger.info(f"{self.model_name} initialized with params: {self.model_params}")

    @abstractmethod
    def build_model(self):
        """Instantiate and return the underlying classifier."""
        pass

    def save_model(self, filepath: str) -> None:
        """
        Persist the fitted model to disk using joblib.

        Args:
            filepath: Absolute or relative path (including filename) to save the model.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError(
                f"[{self.model_name}] No model to save. Call build_model() first."
            )

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"[{self.model_name}] Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model from disk.

        Args:
            filepath: Path to the saved model file.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[{self.model_name}] Cannot load model — file not found: {filepath}"
            )

        self.model = joblib.load(filepath)
        logger.info(f"[{self.model_name}] Model loaded from: {filepath}")


class LogisticRegressionModelBuilder(BaseModelBuilder):
    """
    Builder for a Logistic Regression baseline model.
    Aligns with notebooks/model_pipeline/1_base_model_training.ipynb.
    """
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
        }
        default_params.update(kwargs)
        super().__init__('LogisticRegression', **default_params)

    def build_model(self) -> LogisticRegression:
        self.model = LogisticRegression(**self.model_params)
        logger.info(f"[{self.model_name}] Model built.")
        return self.model


class DecisionTreeModelBuilder(BaseModelBuilder):
    """
    Builder for a Decision Tree classifier.
    Aligns with notebooks/model_pipeline/3_multi_model_training.ipynb.
    """
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42,
        }
        default_params.update(kwargs)
        super().__init__('DecisionTree', **default_params)

    def build_model(self) -> DecisionTreeClassifier:
        self.model = DecisionTreeClassifier(**self.model_params)
        logger.info(f"[{self.model_name}] Model built.")
        return self.model


class RandomForestModelBuilder(BaseModelBuilder):
    """
    Builder for a Random Forest classifier.
    Aligns with notebooks/model_pipeline/3_multi_model_training.ipynb.
    """
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)

    def build_model(self) -> RandomForestClassifier:
        self.model = RandomForestClassifier(**self.model_params)
        logger.info(f"[{self.model_name}] Model built.")
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    """
    Builder for an XGBoost classifier.
    Aligns with notebooks/model_pipeline/3_multi_model_training.ipynb.
    """
    def __init__(self, **kwargs):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    def build_model(self) -> XGBClassifier:
        self.model = XGBClassifier(**self.model_params)
        logger.info(f"[{self.model_name}] Model built.")
        return self.model


def get_model_builder(model_type: str, **kwargs) -> BaseModelBuilder:
    """
    Factory function to retrieve a model builder by name.

    Args:
        model_type: One of 'logistic_regression', 'decision_tree',
                    'random_forest', or 'xgboost'.
        **kwargs:   Optional hyperparameter overrides forwarded to the builder.

    Returns:
        An instance of the appropriate BaseModelBuilder subclass.

    Raises:
        ValueError: If an unsupported model_type string is provided.
    """
    registry = {
        'logistic_regression': LogisticRegressionModelBuilder,
        'decision_tree':       DecisionTreeModelBuilder,
        'random_forest':       RandomForestModelBuilder,
        'xgboost':             XGBoostModelBuilder,
    }

    key = model_type.lower().replace(' ', '_').replace('-', '_')
    if key not in registry:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Choose from: {list(registry.keys())}"
        )

    return registry[key](**kwargs)