# Model Pipeline Service
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/yshanukajay/end-to-end-credit-card-fraud-detection-system"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Model Training Pipeline with PySpark, MLflow and S3 integration"

# Install Java and system dependencies (creating man directory to prevent Debian post-install failure)
RUN mkdir -p /usr/share/man/man1 \
 && apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless curl wget \
    && rm -rf /var/lib/apt/lists/*

# Create group/user and home dir with proper permissions
ARG UID=1001
ARG GID=1001
RUN groupadd --system --gid $GID app \
 && useradd --system --uid $UID --gid app --home-dir /home/appuser --shell /bin/bash appuser \
 && mkdir -p /home/appuser /opt/app /tmp/spark /tmp/hadoop \
 && chown -R appuser:app /home/appuser /opt/app /tmp/spark /tmp/hadoop

# Set working directory
WORKDIR /opt/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code with proper ownership
COPY --chown=appuser:app src/ ./src/
COPY --chown=appuser:app utils/ ./utils/
COPY --chown=appuser:app pipelines/ ./pipelines/
COPY --chown=appuser:app config.yaml .

# Copy entrypoint script
COPY docker/model-pipeline/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Environment variables (consolidated)
ENV HOME=/home/appuser \
    JAVA_HOME=/usr/lib/jvm/default-java \
    JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.base/java.util.regex=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED" \
    JDK_JAVA_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.base/java.util.regex=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED" \
    SPARK_LOCAL_DIRS=/tmp/spark \
    HADOOP_OPTS="-Djava.io.tmpdir=/tmp/hadoop" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    MPLCONFIGDIR=/tmp/matplotlib \
    GIT_PYTHON_REFRESH=quiet

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=2 \
    CMD python3 -c "import boto3, mlflow, os; boto3.client('s3').head_bucket(Bucket=os.environ.get('S3_BUCKET', 'credit-card-fraud-data-bucket'))" || exit 1

# Use entrypoint script
ENTRYPOINT ["bash", "/entrypoint.sh"]
