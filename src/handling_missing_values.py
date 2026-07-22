import logging
from enum import Enum
from typing import Optional, List, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
import pandas as pd  # Keep for educational comparison
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from utils.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def handle(self, df: DataFrame) -> DataFrame:
        """Handle missing values in the DataFrame."""
        pass


class DropDuplicatesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to drop duplicate rows in the PySpark DataFrame.
    """
    def handle(self, df: DataFrame) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("DUPLICATE REMOVAL (PySpark)")
        logger.info(f"{'='*60}")
        
        ############### PANDAS CODES ###########################
        # initial_count = len(df)
        # df_cleaned = df.drop_duplicates().reset_index(drop=True)
        # final_count = len(df_cleaned)
        
        ############### PYSPARK CODES ###########################
        initial_count = df.count()
        df_cleaned = df.dropDuplicates()
        final_count = df_cleaned.count()
        
        n_dropped = initial_count - final_count
        logger.info(f"Number of duplicate rows dropped: {n_dropped}")
        logger.info(f"✓ Shape went from {initial_count} rows to {final_count} rows")
        logger.info(f"{'='*60}\n")
        return df_cleaned


class RawFeatureEngineeringStrategy(MissingValueHandlingStrategy):
    """
    Strategy to engineer raw Kaggle credit card transactions dataset into engineered features.
    Saves and renames columns to match the recommended 13 columns.
    """
    def handle(self, df: DataFrame) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("RAW FEATURE ENGINEERING (PySpark)")
        logger.info(f"{'='*60}")
        
        if 'distance_to_merchant' in df.columns and 'customer_age' in df.columns:
            logger.info("✓ Columns are already engineered. Skipping raw feature engineering.")
            logger.info(f"{'='*60}\n")
            return df
            
        logger.info("Starting feature engineering on raw transaction columns...")
        
        ############### PANDAS CODES ###########################
        # pandas code ...
        
        ############### PYSPARK CODES ###########################
        from utils.spark_utils import preprocess_credit_card_data
        df_engineered = preprocess_credit_card_data(df)
        
        # Keep only the recommended columns
        recommended_cols = [
            'distance_to_merchant',
            'customer_age',
            'transaction_hour',
            'day_of_week',
            'is_weekend',
            'location_mismatch',
            'velocity_last_24h',
            'foreign_transaction',
            'amount',
            'merchant_category',
            'city_population',
            'gender',
        ]
        if 'is_fraud' in df_engineered.columns:
            recommended_cols.append('is_fraud')
            
        existing_cols = [col for col in recommended_cols if col in df_engineered.columns]
        df_final = df_engineered.select(existing_cols)
        
        logger.info(f"✓ Engineered raw features using PySpark.")
        logger.info(f"{'='*60}\n")
        return df_final


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Strategy to drop rows with missing values in critical columns."""
    
    def __init__(self, critical_columns: List[str] = None, spark: Optional[SparkSession] = None):
        """
        Initialize the drop strategy.
        
        Args:
            critical_columns: List of column names where nulls are not allowed
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.critical_columns = critical_columns or []
        logger.info(f"Initialized DropMissingValuesStrategy for columns: {self.critical_columns}")

    def handle(self, df: DataFrame) -> DataFrame:
        """
        Drop rows with missing values in critical columns.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with rows dropped
        """
        ############### PANDAS CODES ###########################
        # initial_count = len(df)
        
        ############### PYSPARK CODES ###########################
        initial_count = df.count()
        
        if self.critical_columns:
            ############### PANDAS CODES ###########################
            # df_cleaned = df.dropna(subset=self.critical_columns)
            
            ############### PYSPARK CODES ###########################
            df_cleaned = df.dropna(subset=self.critical_columns)

        else:
            # Drop rows with any null values
            df_cleaned = df.dropna()
        
        ############### PANDAS CODES ###########################
        # final_count = len(df_cleaned)
        # n_dropped = initial_count - final_count
        
        ############### PYSPARK CODES ###########################
        final_count = df_cleaned.count()
        n_dropped = initial_count - final_count

        logger.info(f"✓ Dropped {n_dropped} rows with missing values")
        logger.info(f"  • Initial rows: {initial_count}")
        logger.info(f"  • Final rows: {final_count}")
        
        return df_cleaned


class Gender(str, Enum):
    """Gender enumeration."""
    MALE = 'Male'
    FEMALE = 'Female'


class GenderPrediction(BaseModel):
    """Gender prediction model."""
    firstname: str
    lastname: str
    pred_gender: Gender


class GenderImputer:
    """Imputer that uses Groq API to predict gender based on names."""
    
    def __init__(self):
        """Initialize with Groq client."""
        self.groq_client = groq.Groq()
        self._predictions_cache = {}

    def _predict_gender(self, firstname: str, lastname: str) -> str:
        """
        Predict gender using Groq API.
        
        Args:
            firstname: First name
            lastname: Last name
            
        Returns:
            Predicted gender ('Male' or 'Female')
        """
        # Check cache first
        cache_key = f"{firstname}_{lastname}".lower()
        if cache_key in self._predictions_cache:
            return self._predictions_cache[cache_key]
        
        try:
            prompt = f"""
                What is the most likely gender (Male or Female) for someone with the first name '{firstname}'
                and last name '{lastname}' ?

                Your response only consists of one word: Male or Female
                """
            
            response = self.groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{"role": "user", "content": prompt}],
            )
            
            predicted_gender = response.choices[0].message.content.strip()
            
            # Validate prediction
            prediction = GenderPrediction(
                                        firstname=firstname, 
                                        lastname=lastname, 
                                        pred_gender=predicted_gender
                                        )
            
            # Cache the result
            self._predictions_cache[cache_key] = prediction.pred_gender
            
            logger.info(f'Predicted gender for {firstname} {lastname}: {prediction.pred_gender}')
            return prediction.pred_gender
            
        except Exception as e:
            logger.error(f"Error predicting gender for {firstname} {lastname}: {str(e)}")
            return None
    
    def impute(self, df: DataFrame) -> DataFrame:
        """
        Impute missing gender values using API predictions.
        
        Args:
            df: PySpark DataFrame with Gender, Firstname, and Lastname columns
            
        Returns:
            DataFrame with imputed gender values
        """
        ############### PANDAS CODES ###########################
        # missing_gender_index = df['Gender'].isnull()
        # for idx in df[missing_gender_index].index:
        #     first_name = df.loc[idx, 'Firstname']
        #     last_name = df.loc[idx, 'Lastname']
        #     gender = self._predict_gender(first_name, last_name)
        #     if gender:
        #         df.loc[idx, 'Gender'] = gender
        
        ############### PYSPARK CODES ###########################
        # Create a UDF (User Defined Functions) for gender prediction
        predict_gender_udf = F.udf(self._predict_gender, StringType())
        missing_gender_df= df.filter(
                    F.col('Gender').isNull() | (F.col('Gender') == '')
                    ).select('Firstname', 'Lastname')

        missing_count = missing_gender_df.count()
        logger.info(f"Imputing {missing_count} missing gender values...")

        predictions_df = missing_gender_df.withColumn(
                                                    'PredictedGender',
                                                    predict_gender_udf(F.col('Firstname'), F.col('Lastname'))
                                                    )

        df_with_predictions = df.join(
            predictions_df,
            on=['Firstname', 'Lastname'],
            how='left'
        )
        
        # Fill missing gender with predictions
        df_imputed = df_with_predictions.withColumn(
            'Gender',
            F.when(
                F.col('Gender').isNull() | (F.col('Gender') == ''),
                F.col('PredictedGender')
            ).otherwise(F.col('Gender'))
        ).drop('PredictedGender')

        return df_imputed
        


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to fill missing values using various methods.
    Supports mean/median/mode filling and custom imputers.
    """
    
    def __init__(
        self, 
        method: str = 'mean', 
        fill_value: Optional[Union[str, float, int]] = None, 
        relevant_column: Optional[str] = None, 
        is_custom_imputer: bool = False,
        custom_imputer: Optional[object] = None,
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize the fill strategy.
        
        Args:
            method: Method to use ('mean', 'median', 'mode', 'constant')
            fill_value: Value to use for constant filling
            relevant_column: Column to fill (if None, fills all numeric columns)
            is_custom_imputer: Whether to use a custom imputer
            custom_imputer: Custom imputer object (must have impute method)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df: DataFrame) -> DataFrame:
        """
        Fill missing values based on the configured strategy.
        
        Args:
            df: PySpark DataFrame
            
        Returns:
            DataFrame with filled values
        """
        if self.is_custom_imputer and self.custom_imputer:
            return self.custom_imputer.impute(df)
        
        if self.relevant_column:
            # Fill specific column
            if self.method == 'mean':
                ############### PANDAS CODES ###########################
                # mean_value = df[self.relevant_column].mean()
                # df_filled = df.fillna({self.relevant_column: mean_value})
                
                ############### PYSPARK CODES ###########################
                mean_value = df.select(F.mean(F.col(self.relevant_column))).collect()[0][0]
                df_filled = df.fillna({self.relevant_column: mean_value})
                
            elif self.method == 'median':
                ############### PANDAS CODES ###########################
                # median_value = df[self.relevant_column].median()
                # df_filled = df.fillna({self.relevant_column: median_value})
                
                ############### PYSPARK CODES ###########################
                median_value = df.approxQuantile(self.relevant_column, [0.5], 0.01)[0]
                df_filled = df.fillna({self.relevant_column: median_value})
                
                
            elif self.method == 'mode':
                ############### PANDAS CODES ###########################
                # mode_value = df[self.relevant_column].mode()[0]
                # df_filled = df.fillna({self.relevant_column: mode_value})
                
                ############### PYSPARK CODES ###########################
                mode_value = df.groupBy(self.relevant_column).count().orderBy(F.desc('count')).first()[0]
                df_filled = df.fillna({self.relevant_column: mode_value})
                
            elif self.method == 'constant' and self.fill_value is not None:
                df_filled = df.fillna({self.relevant_column: self.fill_value})
                logger.info(f'✓ Filled missing values in {self.relevant_column} with constant: {self.fill_value}')
                
            else:
                raise ValueError(f"Invalid method '{self.method}' or missing fill_value")
                
        else:
            # Fill all columns based on method
            if self.method == 'constant' and self.fill_value is not None:
                df_filled = df.fillna(self.fill_value)
                logger.info(f'✓ Filled all missing values with constant: {self.fill_value}')
            else:
                # Use Spark ML Imputer for mean/median on all numeric columns
                numeric_cols = [field.name for field in df.schema.fields 
                              if field.dataType.typeName() in ['integer', 'long', 'float', 'double']]
                
                if numeric_cols:
                    imputer = Imputer(
                        inputCols=numeric_cols,
                        outputCols=[f"{col}_imputed" for col in numeric_cols],
                        strategy=self.method if self.method in ['mean', 'median'] else 'mean'
                    )
                    
                    model = imputer.fit(df)
                    df_imputed = model.transform(df)
                    
                    # Replace original columns with imputed ones
                    for col in numeric_cols:
                        df_imputed = df_imputed.withColumn(col, F.col(f"{col}_imputed")) \
                            .drop(f"{col}_imputed")
                    
                    df_filled = df_imputed
                    logger.info(f'✓ Filled missing values in numeric columns using {self.method}')
                else:
                    df_filled = df
                    logger.warning('No numeric columns found for imputation')
        
        return df_filled


class MedianImputationStrategy(MissingValueHandlingStrategy):
    """
    Strategy to impute continuous numerical columns with their median in PySpark.
    """
    def __init__(self, columns: List[str], spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.columns = columns

    def handle(self, df: DataFrame) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("MEDIAN IMPUTATION (Continuous Numerical Features - PySpark)")
        logger.info(f"{'='*60}")
        
        df_imputed = df
        for col in self.columns:
            if col in df_imputed.columns:
                n_missing = df_imputed.filter(F.col(col).isNull() | F.isnan(col)).count()
                if n_missing > 0:
                    ############### PANDAS CODES ###########################
                    # median_val = df[col].median()
                    # df_imputed = df_imputed.fillna({col: median_val})
                    
                    ############### PYSPARK CODES ###########################
                    median_val = df_imputed.approxQuantile(col, [0.5], 0.01)[0]
                    df_imputed = df_imputed.fillna({col: median_val})
                    logger.info(f"✓ Column [{col}]: Filled {n_missing} missing values with median = {median_val:.4f}")
                else:
                    logger.info(f"  Column [{col}]: No missing values found.")
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"{'='*60}\n")
        return df_imputed


class ModeImputationStrategy(MissingValueHandlingStrategy):
    """
    Strategy to impute binary, count, or categorical columns with their mode in PySpark.
    """
    def __init__(self, columns: List[str], spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.columns = columns

    def handle(self, df: DataFrame) -> DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info("MODE IMPUTATION (Categorical Features - PySpark)")
        logger.info(f"{'='*60}")
        
        df_imputed = df
        for col in self.columns:
            if col in df_imputed.columns:
                # Type-safe check for missing values depending on column data type
                col_type = dict(df_imputed.dtypes)[col]
                if col_type in ("string", "char", "varchar"):
                    n_missing = df_imputed.filter(F.col(col).isNull() | (F.col(col) == "")).count()
                elif col_type in ("double", "float"):
                    n_missing = df_imputed.filter(F.col(col).isNull() | F.isnan(col)).count()
                else:
                    n_missing = df_imputed.filter(F.col(col).isNull()).count()
                if n_missing > 0:
                    ############### PANDAS CODES ###########################
                    # mode_val = df[col].mode()[0]
                    # df_imputed = df_imputed.fillna({col: mode_val})
                    
                    ############### PYSPARK CODES ###########################
                    mode_val = df_imputed.groupBy(col).count().orderBy(F.desc('count')).first()[0]
                    df_imputed = df_imputed.fillna({col: mode_val})
                    logger.info(f"✓ Column [{col}]: Filled {n_missing} missing values with mode = {mode_val}")
                else:
                    logger.info(f"  Column [{col}]: No missing values found.")
            else:
                logger.warning(f"⚠ Column [{col}] not found in the DataFrame.")
                
        logger.info(f"{'='*60}\n")
        return df_imputed


class MissingValuePipeline:
    """
    Pipeline that runs a sequence of missing value handling/cleaning strategies using PySpark.
    """
    def __init__(self, strategies: List[MissingValueHandlingStrategy]):
        self.strategies = strategies

    def execute(self, df: DataFrame) -> DataFrame:
        logger.info("Starting PySpark missing value handling pipeline...")
        processed_df = df
        for strategy in self.strategies:
            processed_df = strategy.handle(processed_df)
            
        # Count remaining missing values in PySpark in a type-safe way
        remaining_missing = 0
        dtypes_dict = dict(processed_df.dtypes)
        for c in processed_df.columns:
            col_type = dtypes_dict[c]
            if col_type in ("string", "char", "varchar"):
                col_missing = processed_df.filter(F.col(c).isNull() | (F.col(c) == "")).count()
            elif col_type in ("double", "float"):
                col_missing = processed_df.filter(F.col(c).isNull() | F.isnan(c)).count()
            else:
                col_missing = processed_df.filter(F.col(c).isNull()).count()
            remaining_missing += col_missing
        logger.info(f"Remaining missing values in dataset: {remaining_missing}")
        if remaining_missing == 0:
            logger.info("✓ All missing values successfully resolved.")
        else:
            logger.warning(f"⚠ Warning: {remaining_missing} missing values still remain in the dataset.")
            
        return processed_df


def create_default_missing_value_pipeline(spark: Optional[SparkSession] = None) -> MissingValuePipeline:
    """
    Creates a pre-configured pipeline matching the strategy used in the Jupyter Notebook:
    - Removes duplicate rows
    - Imputes continuous columns with median
    - Imputes binary, count, and categorical columns with mode
    """
    continuous_cols = ['amount', 'customer_age', 'distance_to_merchant', 'city_population']
    discrete_cols = [
        'foreign_transaction', 'location_mismatch', 'is_fraud',
        'transaction_hour', 'velocity_last_24h', 'merchant_category',
        'gender', 'day_of_week', 'is_weekend'
    ]
    
    spark_sess = spark or get_or_create_spark_session()
    
    return MissingValuePipeline([
        DropDuplicatesStrategy(spark_sess),
        RawFeatureEngineeringStrategy(spark_sess),
        MedianImputationStrategy(columns=continuous_cols, spark=spark_sess),
        ModeImputationStrategy(columns=discrete_cols, spark=spark_sess)
    ])