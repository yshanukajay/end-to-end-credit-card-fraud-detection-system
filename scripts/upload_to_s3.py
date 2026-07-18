#!/usr/bin/env python3
import os
import sys
import argparse
import threading
from dotenv import load_dotenv

# Try importing boto3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    print("[ERROR] 'boto3' is not installed. Please run: pip install boto3")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)


class ProgressPercentage(object):
    """Callback class to display upload progress percentage in console."""
    
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                f"\rUploading {os.path.basename(self._filename)}: {self._seen_so_far / (1024*1024):.2f}MB / {self._size / (1024*1024):.2f}MB ({percentage:.2f}%)"
            )
            sys.stdout.flush()


def upload_file_to_s3(local_path, bucket_name, s3_key=None):
    """
    Upload a local file to an S3 bucket with progress updates.
    """
    # Ensure local path is absolute relative to project root
    if not os.path.isabs(local_path):
        local_path = os.path.abspath(os.path.join(PROJECT_ROOT, local_path))

    if not os.path.exists(local_path):
        print(f"[ERROR] Local file does not exist: {local_path}")
        return False

    # Default S3 key to the relative file path if not provided
    if s3_key is None:
        # e.g., 'dataset/raw/fraudTrain.csv' -> 'raw/fraudTrain.csv'
        rel_path = os.path.relpath(local_path, PROJECT_ROOT).replace('\\', '/')
        if rel_path.startswith('dataset/'):
            s3_key = rel_path.replace('dataset/', '')
        else:
            s3_key = os.path.basename(local_path)

    # Clean up invalid AWS environment overrides loaded from .env (e.g. docker-only paths)
    for env_var in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE']:
        val = os.getenv(env_var)
        if val and not os.path.exists(val):
            del os.environ[env_var]

    # Initialize S3 Client
    session_profile = os.getenv('AWS_PROFILE', 'default')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print(f"[INFO] Initializing S3 client (Profile: '{session_profile}', Region: '{region}')...")
    try:
        session = boto3.Session(profile_name=session_profile, region_name=region)
        s3 = session.client('s3')
    except Exception as e:
        print(f"[ERROR] Failed to initialize boto3 session: {str(e)}")
        print("Hint: Run 'aws configure' or check your credentials.")
        return False

    # Check/Validate bucket
    print(f"[INFO] Validating S3 bucket '{bucket_name}'...")
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ('404', 404):
            print(f"[WARNING] Bucket '{bucket_name}' does not exist. Attempting to create it...")
            try:
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"[SUCCESS] Successfully created bucket: {bucket_name}")
            except Exception as ce:
                print(f"[ERROR] Failed to create bucket: {str(ce)}")
                return False
        elif error_code in ('403', 403):
            print(f"[ERROR] Access Denied: You do not have permission to access bucket '{bucket_name}'.")
            return False
        else:
            print(f"[ERROR] Failed to access bucket: {str(e)}")
            return False

    print(f"[INFO] Uploading {local_path} to s3://{bucket_name}/{s3_key}...")
    try:
        progress_callback = ProgressPercentage(local_path)
        
        # Perform S3 Upload
        s3.upload_file(
            Filename=local_path,
            Bucket=bucket_name,
            Key=s3_key,
            Callback=progress_callback
        )
        print(f"\n[SUCCESS] Upload completed successfully!")
        return True

    except FileNotFoundError:
        print(f"\n[ERROR] Local file not found: {local_path}")
        return False
    except NoCredentialsError:
        print("\n[ERROR] AWS credentials not found. Run 'aws configure' or set AWS env vars.")
        return False
    except ClientError as e:
        print(f"\n[ERROR] S3 client error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload dataset files to S3 bucket.")
    parser.add_argument('--file', type=str, default='dataset/raw/fraudTrain.csv',
                        help='Path to local file to upload (default: dataset/raw/fraudTrain.csv)')
    parser.add_argument('--bucket', type=str, default=os.getenv('S3_BUCKET'),
                        help='Target S3 bucket name (defaults to S3_BUCKET from .env)')
    parser.add_argument('--key', type=str, default=None,
                        help='S3 object key (defaults to file structure within raw/processed)')

    args = parser.parse_args()

    # Fallback/Validation for bucket name
    bucket_name = args.bucket
    if not bucket_name:
        print("[ERROR] S3 Bucket name not specified.")
        print("Hint: Set 'S3_BUCKET' in your .env file or pass it using the '--bucket' flag.")
        sys.exit(1)

    # Perform upload
    success = upload_file_to_s3(args.file, bucket_name, args.key)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
