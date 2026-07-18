import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd  # Keep pandas import for educational purposes
from pyspark.sql import DataFrame, SparkSession
from utils.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion supporting both pandas and PySpark."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize DataIngestor with a SparkSession.
        
        Args:
            spark: Optional SparkSession. If not provided, will create/get one.
        """
        self.spark = spark or get_or_create_spark_session()
        
    def _configure_s3(self, file_path: str) -> str:
        """Configure Spark Hadoop configuration for S3 access if path is S3."""
        if file_path.startswith("s3://") or file_path.startswith("s3a://"):
            s3a_path = file_path.replace("s3://", "s3a://")
            
            # Load credentials/region using our helper config functions
            from utils.config import get_aws_config
            aws_cfg = get_aws_config()
            region = aws_cfg.get('region', 'ap-south-1')
            
            # Extract keys from system environment first, then config
            access_key = os.environ.get('AWS_ACCESS_KEY_ID') or aws_cfg.get('aws_access_key_id')
            secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or aws_cfg.get('aws_secret_access_key')
            
            # Fallback to local profile check if boto3 is available
            if not access_key or not secret_key:
                try:
                    import boto3
                    # Pop env overrides first
                    for env_var in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE']:
                        val = os.getenv(env_var)
                        if val and not os.path.exists(val):
                            del os.environ[env_var]
                    session = boto3.Session()
                    creds = session.get_credentials()
                    if creds:
                        access_key = creds.access_key
                        secret_key = creds.secret_key
                except Exception:
                    pass

            # Configure Hadoop S3A settings
            sc = self.spark.sparkContext
            hadoop_conf = sc._jsc.hadoopConfiguration()
            
            hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            if access_key and secret_key:
                hadoop_conf.set("fs.s3a.access.key", access_key)
                hadoop_conf.set("fs.s3a.secret.key", secret_key)
            else:
                hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
                
            hadoop_conf.set("fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
            hadoop_conf.set("fs.s3a.connection.ssl.enabled", "true")
            hadoop_conf.set("fs.s3a.fast.upload", "true")
            
            logger.info(f"✓ Configured Spark Hadoop context for S3 access (Region: {region}, Path: {s3a_path})")
            return s3a_path
        return file_path
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> DataFrame:
        """
        Ingest data from the specified path.
        
        Args:
            file_path_or_link: Path to the data file
            
        Returns:
            DataFrame (PySpark or pandas depending on implementation)
        """
        pass


class DataIngestorCSV(DataIngestor):
    """CSV data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest CSV data using PySpark.
        
        Args:
            file_path_or_link: Path to the CSV file
            **options: Additional options for CSV reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - CSV (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting CSV data ingestion from: {file_path_or_link}")
        
        try:
            # Default CSV options
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)
            
            ############### PANDAS CODES ###########################
            # df = pd.read_csv(file_path_or_link)
            
            ############### PYSPARK CODES ###########################
            file_path_or_link = self._configure_s3(file_path_or_link)
            df = self.spark.read.options(**csv_options).csv(file_path_or_link)
            
            logger.info(f"✓ CSV data loaded successfully")
            logger.info(f"  • Rows: {df.count()}")
            logger.info(f"  • Columns: {len(df.columns)}")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load CSV data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorExcel(DataIngestor):
    """Excel data ingestion implementation."""
    
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        """
        Ingest Excel data using PySpark.
        Note: This implementation converts Excel to CSV format internally as PySpark
        doesn't have native Excel support. For production use, consider using
        spark-excel library.
        
        Args:
            file_path_or_link: Path to the Excel file
            sheet_name: Name of the sheet to read (optional)
            **options: Additional options
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - EXCEL (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Excel data ingestion from: {file_path_or_link}")
        
        try:
            # For Excel files, we need to use pandas as an intermediary
            # In production, consider using spark-excel library
            logger.info("⚠ Note: Using pandas for Excel reading, then converting to PySpark")
            
            ############### PANDAS CODES ###########################
            # df = pd.read_excel(file_path_or_link)
            
            ############### PYSPARK CODES ###########################
            pandas_df = pd.read_excel(file_path_or_link)
            df = self.spark.createDataFrame(pandas_df)
            
            logger.info(f"✓ Excel data loaded successfully")
            logger.info(f"  • Rows: {df.count()}")
            logger.info(f"  • Columns: {len(df.columns)}")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Excel data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorParquet(DataIngestor):
    """PySpark Parquet data ingestion implementation (new for PySpark)."""
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest Parquet data using PySpark.
        Note: Parquet is a columnar format optimized for big data processing.
        
        Args:
            file_path_or_link: Path to the Parquet file or directory
            **options: Additional options for Parquet reading
            
        Returns:
            PySpark DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA INGESTION - PARQUET (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Parquet data ingestion from: {file_path_or_link}")
        
        try:
            # Read Parquet file(s)
            file_path_or_link = self._configure_s3(file_path_or_link)
            df = self.spark.read.parquet(file_path_or_link)
            
            logger.info(f"✓ Parquet data loaded successfully")
            logger.info(f"  • Rows: {df.count()}")
            logger.info(f"  • Columns: {len(df.columns)}")
            logger.info(f"{'='*60}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Parquet data from {file_path_or_link}: {str(e)}")
            logger.info(f"{'='*60}\n")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional[SparkSession] = None) -> DataIngestor:
        """
        Get appropriate data ingestor based on file extension.
        
        Args:
            file_path: Path to the data file
            spark: Optional SparkSession
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")