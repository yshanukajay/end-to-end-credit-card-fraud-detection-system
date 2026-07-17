import os
import time
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate
from utils.logger import get_logger

# Retrieve logger configured with file and console handlers
logger = get_logger(__name__)


class ModelTrainingStrategy(ABC):
    """
    Abstract Base Class for model training strategies.
    """
    @abstractmethod
    def train(
        self,
        model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Train a model and return the fitted estimator and a metrics summary dict.
        """
        pass


class SimpleTrainingStrategy(ModelTrainingStrategy):
    """
    Trains the model once on the full training set.
    Aligns with notebooks/model_pipeline/1_base_model_training.ipynb.
    """

    def train(
        self,
        model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL TRAINING - SIMPLE FIT")
        logger.info(f"{'='*60}")
        logger.info(f"  Model       : {type(model).__name__}")
        logger.info(f"  Train shape : {np.array(X_train).shape}")

        _validate_inputs(X_train, y_train)

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        train_score = model.score(X_train, y_train)

        metrics = {
            'train_score': round(train_score, 6),
            'training_time_s': round(elapsed, 3),
        }

        logger.info(f"  Training time : {elapsed:.2f}s")
        logger.info(f"  Train score   : {train_score:.4f}")
        logger.info(f"{'='*60}\n")

        return model, metrics


class StratifiedKFoldTrainingStrategy(ModelTrainingStrategy):
    """
    Trains the model with Stratified K-Fold cross-validation and keeps the
    best-fold estimator (highest F1 on the validation fold).
    Aligns with notebooks/model_pipeline/2_kfold_validation.ipynb and
    notebooks/model_pipeline/3_multi_model_training.ipynb.

    Parameters
    ----------
    n_splits    : Number of CV folds (default 6, matching both notebooks).
    scoring     : Sklearn scoring string for fold selection (default 'f1').
    random_state: Seed for StratifiedKFold shuffle.
    """

    def __init__(
        self,
        n_splits: int = 6,
        scoring: str = 'f1',
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        logger.info(
            f"StratifiedKFoldTrainingStrategy initialized: "
            f"n_splits={n_splits}, scoring='{scoring}'"
        )

    def train(
        self,
        model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL TRAINING - STRATIFIED K-FOLD CV")
        logger.info(f"{'='*60}")
        logger.info(f"  Model       : {type(model).__name__}")
        logger.info(f"  Train shape : {np.array(X_train).shape}")
        logger.info(f"  Folds       : {self.n_splits}")
        logger.info(f"  Scoring     : {self.scoring}")

        _validate_inputs(X_train, y_train)

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        start = time.time()
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=self.scoring,
            return_estimator=True,
            return_train_score=False,
        )
        elapsed = time.time() - start

        test_scores: np.ndarray = cv_results['test_score']
        best_index: int = int(np.argmax(test_scores))
        best_estimator: BaseEstimator = cv_results['estimator'][best_index]

        metrics = {
            'cv_scores': test_scores.tolist(),
            'cv_mean': round(float(np.mean(test_scores)), 6),
            'cv_std': round(float(np.std(test_scores)), 6),
            'best_fold': best_index,
            'best_fold_score': round(float(test_scores[best_index]), 6),
            'training_time_s': round(elapsed, 3),
        }

        logger.info(f"  CV {self.scoring} scores : {[round(s, 4) for s in test_scores]}")
        logger.info(f"  Mean {self.scoring:<10} : {metrics['cv_mean']:.4f}  (+/- {metrics['cv_std']:.4f})")
        logger.info(f"  Best fold          : {best_index}  (score: {metrics['best_fold_score']:.4f})")
        logger.info(f"  Total training time: {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")

        return best_estimator, metrics


class SparkTrainingStrategy(ModelTrainingStrategy):
    """
    Trains PySpark MLlib models on PySpark DataFrames.
    """
    def __init__(self, label_col: str = "is_fraud"):
        self.label_col = label_col
        logger.info(f"SparkTrainingStrategy initialized for label: '{label_col}'")

    def train(
        self,
        model: Any,
        X_train: Any,
        y_train: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL TRAINING - PYSPARK MLLIB PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"  Pipeline Type: {type(model).__name__}")
        logger.info(f"  Rows count   : {X_train.count()}")
        
        start = time.time()
        # Train PySpark ML Pipeline
        fitted_model = model.fit(X_train)
        elapsed = time.time() - start
        
        # Calculate metric (Area under ROC) on training data
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        predictions = fitted_model.transform(X_train)
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        
        metrics = {
            'train_auc': round(auc, 6),
            'training_time_s': round(elapsed, 3),
        }
        
        logger.info(f"  Training time : {elapsed:.2f}s")
        logger.info(f"  Train AreaUnderROC : {auc:.4f}")
        logger.info(f"{'='*60}\n")
        
        return fitted_model, metrics


class ModelTrainer:
    """
    Orchestrates model training using a configurable training strategy,
    plus save/load helpers.

    Parameters
    ----------
    strategy : A ModelTrainingStrategy instance. Defaults to
               StratifiedKFoldTrainingStrategy (matching the notebooks).
    """

    def __init__(self, strategy: Optional[ModelTrainingStrategy] = None):
        self.strategy = strategy or StratifiedKFoldTrainingStrategy()
        logger.info(
            f"ModelTrainer initialized with strategy: "
            f"{type(self.strategy).__name__}"
        )

    def train(
        self,
        model: Any,
        X_train: Any,
        y_train: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Delegate training to the chosen strategy.

        Returns
        -------
        (fitted_estimator, metrics_dict)
        """
        return self.strategy.train(model, X_train, y_train)

    def save_model(self, model: Any, filepath: str) -> None:
        """
        Persist a fitted model to disk. Support both PySpark Pipelines and scikit-learn models.

        Args:
            model    : The trained estimator or PipelineModel.
            filepath : Destination path (parent directories are created automatically).

        Raises:
            ValueError: If model is None or filepath is empty.
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL SAVING")
        logger.info(f"{'='*60}")

        if model is None:
            raise ValueError("Cannot save a None model.")
        if not filepath or not isinstance(filepath, str):
            raise ValueError("Invalid filepath provided.")

        parent = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(parent, exist_ok=True)

        start = time.time()
        # Check if the model is from PySpark
        if type(model).__module__.startswith("pyspark.ml"):
            logger.info("  Framework  : PySpark MLlib")
            model.write().overwrite().save(filepath)
            elapsed = time.time() - start
            logger.info(f"  Model type : {type(model).__name__}")
            logger.info(f"  Saved to   : {filepath} (Spark PipelineModel directory)")
            logger.info(f"  Save time  : {elapsed:.2f}s")
            logger.info(f"{'='*60}\n")
            return

        # Scikit-learn / XGBoost
        joblib.dump(model, filepath)
        elapsed = time.time() - start
        size_mb = os.path.getsize(filepath) / (1024 ** 2)

        logger.info(f"  Model type : {type(model).__name__}")
        logger.info(f"  Saved to   : {filepath}")
        logger.info(f"  File size  : {size_mb:.2f} MB")
        logger.info(f"  Save time  : {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")

    def load_model(self, filepath: str, is_spark: bool = False) -> Any:
        """
        Load a previously saved model from disk.

        Args:
            filepath : Path to the model file/directory.
            is_spark : Force loading as a PySpark PipelineModel.

        Returns:
            The deserialised estimator.

        Raises:
            FileNotFoundError : If the file does not exist.
            ValueError        : If filepath is empty or not a string.
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL LOADING")
        logger.info(f"{'='*60}")

        if not filepath or not isinstance(filepath, str):
            raise ValueError("Invalid filepath provided.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        start = time.time()
        
        # Check if directory contains a Spark PipelineModel structure
        is_spark_dir = os.path.isdir(filepath) and (
            os.path.exists(os.path.join(filepath, "metadata")) or
            os.path.exists(os.path.join(filepath, "stages"))
        )
        
        if is_spark_dir or is_spark:
            from pyspark.ml import PipelineModel
            logger.info("  Framework  : PySpark MLlib")
            model = PipelineModel.load(filepath)
            elapsed = time.time() - start
            logger.info(f"  Model type : {type(model).__name__}")
            logger.info(f"  Loaded from: {filepath} (Spark PipelineModel directory)")
            logger.info(f"  Load time  : {elapsed:.2f}s")
            logger.info(f"{'='*60}\n")
            return model

        # Scikit-learn
        model = joblib.load(filepath)
        elapsed = time.time() - start
        size_mb = os.path.getsize(filepath) / (1024 ** 2)

        logger.info(f"  Model type : {type(model).__name__}")
        logger.info(f"  Loaded from: {filepath}")
        logger.info(f"  File size  : {size_mb:.2f} MB")
        logger.info(f"  Load time  : {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")

        return model


# ---------------------------------------------------------------------------
# Helpers & Factory Functions
# ---------------------------------------------------------------------------

def _validate_inputs(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> None:
    """Raise informative errors for common data problems before training."""
    if X is None or y is None:
        raise ValueError("Training data (X or y) cannot be None.")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty.")
    if len(X) != len(y):
        raise ValueError(
            f"Feature/target length mismatch: X has {len(X)} rows, "
            f"y has {len(y)} rows."
        )


def create_default_trainer() -> ModelTrainer:
    """
    Creates a ModelTrainer pre-configured with the default Stratified K-Fold Strategy.
    Uses 6 splits matching the model validation notebooks.
    """
    strategy = StratifiedKFoldTrainingStrategy(n_splits=6, scoring='f1', random_state=42)
    return ModelTrainer(strategy=strategy)


def create_trainer_from_config(config_path: Optional[str] = None) -> ModelTrainer:
    """
    Creates a ModelTrainer using parameters from config.yaml.
    """
    import yaml
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config.yaml')
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        model_cfg = config.get('model', {})
        framework = model_cfg.get('framework', 'scikit-learn')
        
        if framework == 'pyspark':
            target_col = config.get('columns', {}).get('target', 'is_fraud')
            strategy = SparkTrainingStrategy(label_col=target_col)
        else:
            training_cfg = config.get('training', {})
            strategy_type = training_cfg.get('default_training_strategy', 'cv')
            random_state = training_cfg.get('random_state', 42)
            
            if strategy_type == 'cv':
                n_splits = training_cfg.get('cv_folds', 5)
                strategy = StratifiedKFoldTrainingStrategy(
                    n_splits=n_splits,
                    scoring='f1',
                    random_state=random_state
                )
            else:
                strategy = SimpleTrainingStrategy()
            
        return ModelTrainer(strategy=strategy)
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}, falling back to default trainer: {e}")
        return create_default_trainer()