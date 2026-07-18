#!/usr/bin/env bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Ignore comments and empty lines
        if [[ ! "$line" =~ ^# ]] && [[ "$line" =~ = ]]; then
            # Clean carriage return (\r) for Windows compatibility
            cleaned_line=$(echo "$line" | tr -d '\r')
            export "$cleaned_line"
        fi
    done < .env
    echo "⚙️ Loaded environment variables from .env"
fi

unset MLFLOW_TRACKING_URI
unset MLFLOW_DEFAULT_ARTIFACT_ROOT

# Set local MLflow tracking URI and Python path
export MLFLOW_TRACKING_URI="http://localhost:5001"
export PYTHONPATH="."
export AWS_PROFILE="${AWS_PROFILE:-default}"  # Use default AWS profile

# Set AWS credentials from environment or credentials file
if [ -z "$AWS_ACCESS_KEY_ID" ] && [ -f ~/.aws/credentials ]; then
    # Try using python to resolve credentials first for maximum robustness
    CREDENTIALS=$(python -c "
import boto3
try:
    session = boto3.Session()
    creds = session.get_credentials()
    if creds:
        print(f'AWS_ACCESS_KEY_ID={creds.access_key}')
        print(f'AWS_SECRET_ACCESS_KEY={creds.secret_key}')
        print(f'AWS_REGION={session.region_name or \"\"}')
except Exception:
    pass
" 2>/dev/null)
    
    if [ ! -z "$CREDENTIALS" ]; then
        cleaned_creds=$(echo "$CREDENTIALS" | tr -d '\r')
        eval "$cleaned_creds"
        echo "🔑 Loaded AWS credentials via boto3"
    else
        # Fallback to grep extraction if python parsing fails
        AWS_ACCESS_KEY_ID=$(grep -A2 "\\[$AWS_PROFILE\\]" ~/.aws/credentials 2>/dev/null | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ' | tr -d '\r')
        AWS_SECRET_ACCESS_KEY=$(grep -A2 "\\[$AWS_PROFILE\\]" ~/.aws/credentials 2>/dev/null | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ' | tr -d '\r')
        export AWS_ACCESS_KEY_ID
        export AWS_SECRET_ACCESS_KEY
        echo "🔑 Extracted AWS credentials from ~/.aws/credentials"
    fi
fi

# Set S3 folder configs to avoid profile conflicts
if [ -f ~/.aws/config ] && [ -f ~/.aws/credentials ]; then
    export AWS_CONFIG_FILE="$HOME/.aws/config"
    export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
fi

# Fallback values if region is not set
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

# Clean up invalid AWS environment overrides mapping in .env (e.g. docker-only paths)
if [ ! -z "$AWS_SHARED_CREDENTIALS_FILE" ] && [ ! -f "$AWS_SHARED_CREDENTIALS_FILE" ]; then
    unset AWS_SHARED_CREDENTIALS_FILE
fi
if [ ! -z "$AWS_CONFIG_FILE" ] && [ ! -f "$AWS_CONFIG_FILE" ]; then
    unset AWS_CONFIG_FILE
fi

echo "🧹 Cleaned environment variables"
echo "Target MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "PYTHONPATH: $PYTHONPATH"
if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "AWS credentials: ${AWS_ACCESS_KEY_ID:0:5}***"
fi

# Activate virtual environment
if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
    echo "✓ Activated virtual environment (.venv/Scripts)"
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
    echo "✓ Activated virtual environment (.venv/bin)"
else
    echo "⚠️  Virtual environment not found (.venv), using global/conda Python"
fi

# Run the command passed as argument
if [ $# -gt 0 ]; then
    exec "$@"
fi
