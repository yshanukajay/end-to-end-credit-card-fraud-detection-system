#!/usr/bin/env bash
set -euo pipefail

echo "🔮 Starting Airflow Service for Credit Card Fraud Detection Pipeline..."
echo "whoami: $(whoami)"
echo "HOME: $HOME"

# Extract AWS credentials from mounted file if available
if [ -f "/home/airflow/.aws/credentials" ]; then
    echo "🔑 Setting up AWS credentials for PySpark & S3..."
    export AWS_ACCESS_KEY_ID=$(grep -A 10 "^\[default\]" /home/airflow/.aws/credentials | grep "aws_access_key_id" | cut -d'=' -f2 | tr -d ' \r')
    export AWS_SECRET_ACCESS_KEY=$(grep -A 10 "^\[default\]" /home/airflow/.aws/credentials | grep "aws_secret_access_key" | cut -d'=' -f2 | tr -d ' \r')
    echo "✅ AWS credentials extracted for Airflow PySpark S3A"
elif [ -f "/aws/credentials" ]; then
    echo "🔑 Setting up AWS credentials from /aws/credentials..."
    export AWS_ACCESS_KEY_ID=$(grep -A 10 "^\[default\]" /aws/credentials | grep "aws_access_key_id" | cut -d'=' -f2 | tr -d ' \r')
    export AWS_SECRET_ACCESS_KEY=$(grep -A 10 "^\[default\]" /aws/credentials | grep "aws_secret_access_key" | cut -d'=' -f2 | tr -d ' \r')
    echo "✅ AWS credentials extracted from /aws/credentials"
fi

# Configure Spark S3A inside Airflow container
export SPARK_CONF_DIR=/tmp/spark-conf
mkdir -p $SPARK_CONF_DIR /tmp/spark /tmp/hadoop /tmp/ivy-cache

cat > $SPARK_CONF_DIR/spark-defaults.conf << EOF
spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain
spark.hadoop.fs.s3a.endpoint=s3.${AWS_DEFAULT_REGION:-us-east-1}.amazonaws.com
spark.hadoop.fs.s3a.path.style.access=false
spark.hadoop.fs.s3a.connection.ssl.enabled=true
spark.hadoop.fs.s3a.fast.upload=true
EOF

echo "🔧 Airflow Spark S3A configuration completed"

# Ensure artifact & log directories are writable by the airflow user.
# Windows Docker Desktop mounts volumes as root-owned; chmod 777 makes them
# accessible to uid 50000 (airflow) without requiring root privileges here.
for dir in /opt/app/artifacts /opt/app/logs /opt/app/dataset; do
    if [ -d "$dir" ]; then
        chmod -R 777 "$dir" 2>/dev/null || true
        echo "🔓 Ensured write access: $dir"
    fi
done

# Delegate execution to official Airflow entrypoint
exec /entrypoint "$@"
