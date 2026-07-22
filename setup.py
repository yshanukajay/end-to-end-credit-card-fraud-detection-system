from setuptools import setup, find_packages

setup(
    name="credit-card-fraud-detection",
    version="1.0.0",
    author="yshanukajay",
    description=(
        "End-to-end credit card fraud detection system with PySpark preprocessing, "
        "XGBoost training, Kafka streaming inference, MLflow tracking, and Airflow orchestration."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    python_requires=">=3.11",
    packages=find_packages(include=["src", "src.*", "utils", "utils.*"]),
    install_requires=[
        # Core ML & Data Science
        "numpy>=1.23.2,<2.0",
        "pandas>=1.5.3",
        "scipy>=1.9.2",
        "scikit-learn>=1.2.0",
        "imbalanced-learn>=0.10.1",
        # Model Libraries
        "xgboost>=1.7.3",
        "lightgbm>=3.3.5",
        # Configuration & Utilities
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        # AWS
        "boto3>=1.26.0",
        "botocore>=1.29.0",
        # API & Web Services
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0,<2.11.2",
        # Experiment Tracking
        "mlflow>=2.3.0",
        # Big Data
        "pyspark==3.5.6",
        "pyarrow==14.0.2",
        # Kafka
        "confluent-kafka==2.6.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
        ],
        "airflow": [
            "apache-airflow==2.10.4",
            "apache-airflow-providers-apache-spark>=4.1.5,<6.0.0",
            "apache-airflow-providers-http>=4.7.0",
            "apache-airflow-providers-standard>=1.0.0",
            "airflow-provider-mlflow>=0.1.0",
            "apache-airflow-providers-apache-kafka==1.6.1",
            "pendulum>=3.0.0,<4.0.0",
        ],
        "viz": [
            "matplotlib>=3.6.2",
            "seaborn>=0.12.2",
            "plotly>=5.11.0",
        ],
        "llm": [
            "groq>=0.11.0",
            "wandb>=0.15.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "run-data-pipeline=pipelines.data_pipeline:main",
            "run-train-pipeline=pipelines.train_pipeline:main",
            "run-inference=pipelines.streaming_inference_pipeline:main",
        ],
    },
)
