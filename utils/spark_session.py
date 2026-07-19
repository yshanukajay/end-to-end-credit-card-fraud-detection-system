"""
Centralized SparkSession management for the credit card fraud detection pipeline.
Provides consistent Spark configuration across all modules.
"""

import os
import sys
import logging
from typing import Optional, List
from pyspark.sql import SparkSession

# Ensure correct python workers are launched using the current python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Modular open-packages options for Java 17+ / Java 21 compatibility with PyArrow/PySpark
REQUIRED_JAVA_OPTS = [
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.misc=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.text=ALL-UNNAMED",
    "--add-opens=java.base/java.util.regex=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
]


def _merge_java_options(existing: Optional[str], required_opts: List[str]) -> str:
    """Merge required JVM options with existing options without duplication."""
    if not existing:
        return " ".join(required_opts)
    tokens = set(existing.split())
    for opt in required_opts:
        tokens.add(opt)
    return " ".join(sorted(tokens))


# Set/merge JDK_JAVA_OPTIONS and JAVA_TOOL_OPTIONS environment variables
os.environ["JDK_JAVA_OPTIONS"] = _merge_java_options(
    os.environ.get("JDK_JAVA_OPTIONS"), REQUIRED_JAVA_OPTS
)
os.environ["JAVA_TOOL_OPTIONS"] = _merge_java_options(
    os.environ.get("JAVA_TOOL_OPTIONS"), REQUIRED_JAVA_OPTS
)

logger = logging.getLogger(__name__)


def create_spark_session(
                        app_name: str = "CreditCardFraudDetectionPipeline",
                        master: Optional[str] = None,
                        config_options: Optional[dict] = None
                        ) -> SparkSession:
    """
    Create or get an existing SparkSession with optimized configuration.
    
    Args:
        app_name: Name of the Spark application
        master: Spark master URL (default: local[4] mode to prevent oversubscribing container thread memory)
        config_options: Additional Spark configuration options
        
    Returns:
        SparkSession: Configured SparkSession instance
    """
    master = master or os.getenv("SPARK_MASTER", "local[4]")
    try:
        # Determine extra JVM options (merge if existing in config_options)
        existing_driver_opts = config_options.get("spark.driver.extraJavaOptions") if config_options else None
        existing_executor_opts = config_options.get("spark.executor.extraJavaOptions") if config_options else None

        driver_java_opts = _merge_java_options(existing_driver_opts, REQUIRED_JAVA_OPTS)
        executor_java_opts = _merge_java_options(existing_executor_opts, REQUIRED_JAVA_OPTS)

        # Base configuration for optimal performance
        builder = SparkSession.builder \
                                    .appName(app_name) \
                                    .master(master) \
                                    .config("spark.driver.memory", "3g") \
                                    .config("spark.executor.memory", "2g") \
                                    .config("spark.driver.extraJavaOptions", driver_java_opts) \
                                    .config("spark.executor.extraJavaOptions", executor_java_opts) \
                                    .config("spark.network.timeout", "800s") \
                                    .config("spark.executor.heartbeatInterval", "60s") \
                                    .config("spark.sql.adaptive.enabled", "true") \
                                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                                    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                                    .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
                                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                                    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
                                    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
                                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                                    .config("spark.sql.shuffle.partitions", "200") \
                                    .config("spark.sql.parquet.compression.codec", "snappy") \
                                    .config("spark.sql.parquet.mergeSchema", "false") \
                                    .config("spark.sql.parquet.filterPushdown", "true") \
                                    .config("spark.sql.csv.parser.columnPruning.enabled", "true") \
                                    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        
        # Apply additional configuration if provided
        if config_options:
            for key, value in config_options.items():
                builder = builder.config(key, value)
        
        # Create or get existing session
        spark = builder.getOrCreate()
        
        # Set log level to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"✓ SparkSession created/retrieved: {app_name}")
        logger.info(f"  • Spark Version: {spark.version}")
        logger.info(f"  • Master: {spark.sparkContext.master}")
        logger.info(f"  • Default Parallelism: {spark.sparkContext.defaultParallelism}")
        
        return spark
        
    except Exception as e:
        logger.error(f"✗ Failed to create SparkSession: {str(e)}")
        raise


def stop_spark_session(spark: SparkSession) -> None:
    """
    Safely stop a SparkSession.
    
    Args:
        spark: SparkSession instance to stop
    """
    try:
        if spark and hasattr(spark, 'stop'):
            spark.stop()
            logger.info("✓ SparkSession stopped successfully")
    except Exception as e:
        logger.error(f"✗ Error stopping SparkSession: {str(e)}")


def get_spark_session_info(spark: SparkSession) -> dict:
    """
    Get information about the current SparkSession.
    
    Args:
        spark: Active SparkSession instance
        
    Returns:
        dict: Session information including version, config, etc.
    """
    try:
        info = {
            "version": spark.version,
            "app_name": spark.conf.get("spark.app.name"),
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "executor_memory": spark.conf.get("spark.executor.memory", "default"),
            "executor_cores": spark.conf.get("spark.executor.cores", "default"),
            "adaptive_enabled": spark.conf.get("spark.sql.adaptive.enabled"),
            "arrow_enabled": spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")
        }
        return info
    except Exception as e:
        logger.error(f"✗ Error getting SparkSession info: {str(e)}")
        return {}


def configure_spark_for_ml(spark: SparkSession) -> SparkSession:
    """
    Configure SparkSession specifically for ML workloads.
    
    Args:
        spark: Existing SparkSession instance
        
    Returns:
        SparkSession: Configured SparkSession
    """
    try:
        # ML-specific optimizations
        spark.conf.set("spark.ml.tuning.parallelism", "4")
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        
        logger.info("✓ SparkSession configured for ML workloads")
        return spark
        
    except Exception as e:
        logger.error(f"✗ Error configuring Spark for ML: {str(e)}")
        return spark


# Global session management (optional)
_global_spark_session = None


def get_or_create_spark_session(
    app_name: str = "CreditCardFraudDetectionPipeline",
    **kwargs
) -> SparkSession:
    """
    Get existing global SparkSession or create a new one.
    
    Args:
        app_name: Name of the Spark application
        **kwargs: Additional arguments for create_spark_session
        
    Returns:
        SparkSession: Active SparkSession instance
    """
    global _global_spark_session
    
    if _global_spark_session is None or _global_spark_session._jsc is None:
        _global_spark_session = create_spark_session(app_name, **kwargs)
    
    return _global_spark_session
