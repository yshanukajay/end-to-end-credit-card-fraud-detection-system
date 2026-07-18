SHELL := /usr/bin/env bash
.ONESHELL:

.PHONY: all clean install setup-dirs train-pipeline data-pipeline batch-inference run-all clean-kill help \
         kafka-format kafka-start kafka-start-bg kafka-stop kafka-topics kafka-cleanup-topics \
         kafka-producer-stream kafka-producer-batch kafka-consumer kafka-consumer-continuous \
         kafka-check kafka-monitor kafka-sample-scored kafka-reset kafka-help \
         airflow-init airflow-start airflow-kill airflow-reset \
         docker-build docker-up docker-down docker-data-pipeline docker-model-pipeline docker-inference-pipeline docker-run-all docker-status \
         s3-upload-data s3-list s3-clean s3-delete-prefix s3-smoke

# Default Python interpreter
PYTHON = python
MLFLOW_PORT ?= 5001

# Get the directory of this Makefile (always ends with a slash)
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Airflow home directory setup locally
export AIRFLOW_HOME := $(abspath $(ROOT_DIR)airflow)

# Setup virtual environment path depending on OS
ifeq ($(OS),Windows_NT)
  VENV_DIR = .venv
  VENV_PYTHON = $(VENV_DIR)/Scripts/python.exe
  VENV_MLFLOW = $(VENV_DIR)/Scripts/mlflow.exe
  VENV_AIRFLOW = $(VENV_DIR)/Scripts/airflow.exe
  KAFKA_HOME ?= C:\kafka_2.12-3.7.0
  KAFKA_BIN_DIR = $(KAFKA_HOME)/bin/windows/
  KAFKA_SCRIPT_EXT = .bat
  KAFKA_CONFIG_FILE ?= $(ROOT_DIR)kafka/server.properties
else
  VENV_DIR = .venv
  VENV_PYTHON = $(VENV_DIR)/bin/python
  VENV_MLFLOW = $(VENV_DIR)/bin/mlflow
  VENV_AIRFLOW = $(VENV_DIR)/bin/airflow
  KAFKA_HOME ?= /opt/kafka
  KAFKA_BIN_DIR = $(KAFKA_HOME)/bin/
  KAFKA_SCRIPT_EXT = .sh
  KAFKA_CONFIG_FILE ?= $(ROOT_DIR)kafka/server.properties
endif

# Fallback to system / active environment (e.g. Conda) if .venv directory does not exist
ifeq ($(wildcard $(VENV_DIR)),)
  RUN_PYTHON = python
  RUN_MLFLOW = mlflow
  RUN_AIRFLOW = airflow
else
  RUN_PYTHON = $(VENV_PYTHON)
  RUN_MLFLOW = $(VENV_MLFLOW)
  RUN_AIRFLOW = $(VENV_AIRFLOW)
endif

# Cross-platform directory creation tool
MKDIR = python -c "import os, sys; [os.makedirs(d, exist_ok=True) for d in sys.argv[1:]]"

# Default target
all: help

# Help target
help:
	@echo "🚀 Credit Card Fraud ML System with Kafka Streaming"
	@echo "============================================="
	@echo ""
	@echo "📦 Setup Commands:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make setup-dirs          - Create necessary directories for pipelines"
	@echo "  make clean               - Clean up artifacts"
	@echo ""
	@echo "🔄 ML Pipeline Commands:"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline (XGBoost)"
	@echo "  make batch-inference     - Run the streaming inference pipeline"
	@echo ""
	@echo "📊 MLflow Commands:"
	@echo "  make mlflow-ui           - Launch MLflow UI (port $(MLFLOW_PORT))"
	@echo "  make stop-all            - Stop all MLflow servers"
	@echo ""
	@echo "🌊 Kafka Streaming Commands:"
	@echo "  make kafka-validate      - Validate Kafka installation"
	@echo "  make kafka-format        - Format Kafka storage (first time)"
	@echo "  make kafka-start         - Start native Kafka broker"
	@echo "  make kafka-start-bg      - Start broker in background"
	@echo "  make kafka-stop          - Stop native Kafka broker"
	@echo "  make kafka-topics        - Create fraud detection topics"
	@echo "  make kafka-cleanup-topics - Remove unused topics"
	@echo "  make kafka-flush-messages - Flush all messages from topics"
	@echo ""
	@echo "📊 Data Production Commands:"
	@echo "  make kafka-producer-stream - Stream transactions (1/sec for 5 mins)"
	@echo "  make kafka-producer-batch  - Batch produce 100 transaction events"
	@echo ""
	@echo "🤖 ML Processing Commands:"
	@echo "  make kafka-consumer        - Batch ML consumer (process all messages)"
	@echo "  make kafka-consumer-continuous - Real-time continuous ML consumer"
	@echo ""
	@echo "🔧 Monitoring Commands:"
	@echo "  make kafka-check         - Check broker status"
	@echo "  make kafka-monitor       - Monitor cluster health / consumer lag"
	@echo "  make kafka-sample-scored - Show prediction analytics & statistics"
	@echo "  make kafka-reset         - Reset all Kafka log data"
	@echo "  make kafka-help          - Show all Kafka commands help"
	@echo ""
	@echo "🔧 Airflow Orchestration Commands:"
	@echo "  make airflow-init        - Initialize Apache Airflow"
	@echo "  make airflow-start       - Start Airflow in standalone mode"
	@echo "  make airflow-kill        - Kill all Airflow processes"
	@echo "  make airflow-reset       - Reset Airflow database"
	@echo "  make clean-kill          - Kill all processes and clean logs/data"
	@echo ""
	@echo "🐳 Docker Services Commands:"
	@echo "  make docker-build        - Build all Docker images"
	@echo "  make docker-up           - Start all Docker services"
	@echo "  make docker-down         - Stop all Docker services"
	@echo "  make docker-data-pipeline    - Run data pipeline in Docker"
	@echo "  make docker-model-pipeline   - Run model pipeline in Docker"
	@echo "  make docker-inference-pipeline - Run inference pipeline in Docker"
	@echo "  make docker-run-all      - Run all pipelines in Docker"
	@echo "  make docker-status       - Show Docker service status"
	@echo ""
	@echo "🌐 S3 Commands:"
	@echo "  make s3-upload-data              - Upload data/raw & data/processed to S3 (one-time)"
	@echo "  make s3-list PREFIX=<prefix>     - List S3 keys with prefix"
	@echo "  make s3-clean                    - Clean project S3 artifacts (safe)"
	@echo "  make s3-delete-prefix PREFIX=<>  - Delete S3 keys with prefix (careful!)"
	@echo "  make s3-smoke                    - Test S3 connectivity"
	@echo ""
	@echo "💡 Quick Start (Batch Processing):"
	@echo "  1. make install && make setup-dirs"
	@echo "  2. make kafka-start-bg && make kafka-topics"
	@echo "  3. make kafka-producer-batch"
	@echo "  4. make kafka-consumer"
	@echo ""
	@echo "🔄 Real-time Streaming Demo:"
	@echo "  1. Terminal 1: make kafka-consumer-continuous"
	@echo "  2. Terminal 2: make kafka-producer-stream"
	@echo "  3. Watch real-time ML processing!"
	@echo "  4. Terminal 3: make kafka-sample-scored (view analytics)"

# ========================================================================================
# SETUP AND ENVIRONMENT COMMANDS
# ========================================================================================

# Install project dependencies and set up environment
install:
	@echo "📦 Installing project dependencies and setting up environment..."
	@echo "Activating and installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(ROOT_DIR)requirements.txt --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.11.txt
	@echo "✅ Installation completed successfully!"

# Create necessary directories
setup-dirs:
	@echo "📁 Creating necessary directories..."
	@$(MKDIR) artifacts/data artifacts/models artifacts/encode
	@$(MKDIR) artifacts/mlflow_run_artifacts artifacts/mlflow_training_artifacts
	@$(MKDIR) artifacts/inference_batches
	@$(MKDIR) dataset/processed dataset/raw runtime/kafka-logs runtime/pids
	@echo "✅ Directories created successfully!"

# Clean up
clean:
	@echo "🧹 Cleaning up artifacts..."
	rm -rf artifacts/models/* artifacts/evaluation/* artifacts/predictions/* artifacts/encode/* mlruns
	@echo "✅ Cleanup completed!"

# ========================================================================================
# ML PIPELINE COMMANDS
# ========================================================================================

# Run data pipeline
data-pipeline: setup-dirs
	@echo "🔄 Start running data pipeline..."
	@$(RUN_PYTHON) pipelines/data_pipeline.py
	@echo "✅ Data pipeline completed successfully!"

# Run training pipeline
train-pipeline: setup-dirs
	@echo "🎯 Running XGBoost model training pipeline..."
	@$(RUN_PYTHON) pipelines/train_pipeline.py
	@echo "✅ Training pipeline completed successfully!"

# Run streaming inference pipeline
batch-inference: setup-dirs
	@echo "🔮 Running batch/streaming inference pipeline..."
	@$(RUN_PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "✅ Batch/streaming inference completed successfully!"

# Run the entire pipeline end-to-end
run-all: data-pipeline train-pipeline batch-inference
	@echo "✅ Entire ML pipeline (Ingestion -> Split -> Features -> Scale -> Model -> Evaluator) run completed successfully!"

# Comprehensive cleanup and kill command
clean-kill:
	@echo "🧹 Comprehensive cleanup and kill operation..."
	@echo "=============================================="
	@echo "⚠️  This will kill all processes and remove logs/data (NOT code)"
	@python -c "import sys; val = input('Continue? (y/N): '); sys.exit(0 if val.strip().lower() == 'y' else 1)"
	@echo ""
	@echo "🛑 Killing all processes..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%airflow%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%kafka%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%mlflow%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
else
	@pkill -f kafka || echo "No Kafka processes found"
	@pkill -f airflow || echo "No Airflow processes found"
	@pkill -f mlflow || echo "No MLflow processes found"
	@pkill -f spark || echo "No Spark processes found"
	@pkill -f java.*kafka || echo "No Java Kafka processes found"
endif
	@echo ""
	@echo "🗑️  Removing logs and data directories..."
	@rm -rf runtime/kafka-logs/ || echo "Kafka logs not found"
	@rm -rf runtime/pids/ || echo "PID files not found"
	@rm -rf runtime/kafka.log || echo "Kafka log file not found"
	@echo "🗃️  Completely removing Airflow directory (clears all execution history)..."
	@rm -rf airflow/ || echo "Airflow directory not found"
	@rm -rf mlruns/ || echo "MLflow runs not found"
	@rm -rf artifacts/mlflow_*/ || echo "MLflow artifacts not found"
	@rm -rf artifacts/data/streaming_checkpoints/ || echo "Streaming checkpoints not found"
	@find . -path "./.venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./.venv" -prune -o -name "*.pyc" -delete 2>/dev/null || true
	@echo ""
	@echo "✅ Cleanup completed successfully!"

# ========================================================================================
# MLFLOW COMMANDS
# ========================================================================================

mlflow-ui:
	@echo "📊 Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@$(RUN_MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns --host 127.0.0.1 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo "🛑 Stopping all MLflow servers..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort $(MLFLOW_PORT) -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
else
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
endif
	@echo "✅ All MLflow servers have been stopped"

# ========================================================================================
# NATIVE KAFKA STREAMING COMMANDS
# ========================================================================================

# Format native Kafka storage (KRaft mode)
kafka-format:
	@echo "🔧 Formatting native Kafka storage (KRaft mode)..."
	@python -c "import os, sys; sys.exit(0) if 'KAFKA_HOME' in os.environ else (print('❌ KAFKA_HOME not set. Please install Kafka natively and set KAFKA_HOME'), sys.exit(1))"
	@$(MKDIR) runtime/kafka-logs runtime/pids
	@echo "🔑 Generating cluster UUID..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "\
		$$clusterId = & '$(KAFKA_BIN_DIR)kafka-storage.bat' random-uuid; \
		Write-Host 'Using Cluster ID:' $$clusterId; \
		& '$(KAFKA_BIN_DIR)kafka-storage.bat' format -t $$clusterId -c '$(KAFKA_CONFIG_FILE)' \
	"
else
	@CLUSTER_ID=$$($(KAFKA_BIN_DIR)kafka-storage.sh random-uuid); \
	echo "Using Cluster ID: $$CLUSTER_ID"; \
	$(KAFKA_BIN_DIR)kafka-storage.sh format -t $$CLUSTER_ID -c "$(KAFKA_CONFIG_FILE)"
endif
	@echo "✅ Native Kafka storage formatted successfully"

kafka-validate:
	@echo "🔍 Validating native Kafka installation..."
	@$(RUN_PYTHON) -m utils.kafka_utils validate

kafka-start:
	@echo "🚀 Starting native Kafka broker in foreground..."
	$(KAFKA_BIN_DIR)kafka-server-start$(KAFKA_SCRIPT_EXT) $(KAFKA_CONFIG_FILE)

kafka-start-bg:
	@echo "🚀 Starting native Kafka broker in background..."
	@$(MKDIR) runtime/pids
ifeq ($(OS),Windows_NT)
	@powershell -Command "Start-Process -FilePath '$(KAFKA_BIN_DIR)kafka-server-start.bat' -ArgumentList '$(KAFKA_CONFIG_FILE)' -WindowStyle Hidden"
	@echo "✅ Kafka broker started in background (Windows WindowStyle Hidden)"
else
	@nohup $(KAFKA_BIN_DIR)kafka-server-start.sh "$(KAFKA_CONFIG_FILE)" > runtime/kafka.log 2>&1 & \
	echo $$! > runtime/pids/kafka.pid
	@echo "✅ Kafka broker started in background (PID: $$(cat runtime/pids/kafka.pid))"
	@echo "📄 Logs: runtime/kafka.log"
endif

kafka-stop:
	@echo "🛑 Stopping native Kafka broker..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%kafka%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
	@echo "✅ Kafka broker stopped"
else
	@if [ -f "runtime/pids/kafka.pid" ]; then \
		PID=$$(cat runtime/pids/kafka.pid); \
		echo "🔍 Found Kafka PID: $$PID"; \
		kill $$PID || true; \
		rm -f runtime/pids/kafka.pid; \
		echo "✅ Kafka broker stopped"; \
	else \
		echo "⚠️ PID file not found, trying graceful shutdown..."; \
		$(KAFKA_BIN_DIR)kafka-server-stop.sh || true; \
	fi
endif

kafka-topics:
	@echo "📋 Creating fraud detection topics on native broker..."
	@$(RUN_PYTHON) -m utils.kafka_utils setup-topics

kafka-producer-stream:
	@echo "🌊 Starting Kafka streaming producer (real data sampling)..."
	@$(RUN_PYTHON) pipelines/kafka_producer.py --mode streaming --rate 1 --duration 300

kafka-producer-batch:
	@echo "📦 Starting Kafka batch producer (real data sampling)..."
	@$(RUN_PYTHON) pipelines/kafka_producer.py --mode batch --num-events 100

kafka-consumer:
	@echo "🌊 Starting Kafka batch consumer with ML predictions..."
	@$(RUN_PYTHON) pipelines/kafka_batch_consumer.py

kafka-consumer-continuous:
	@echo "🔄 Starting continuous Kafka consumer monitoring..."
	@echo "📡 Monitoring for NEW messages (real-time ML processing)"
	@echo "🛑 Press Ctrl+C to stop monitoring"
	@$(RUN_PYTHON) pipelines/kafka_batch_consumer.py --continuous --poll-interval 5
	
kafka-check:
	@echo "🔍 Checking native Kafka broker status..."
	@$(RUN_PYTHON) -m utils.kafka_utils validate
	
kafka-sample-scored:
	@echo "📊 Analyzing fraud prediction results..."
	@$(RUN_PYTHON) scripts/kafka_analytics.py

kafka-monitor:
	@echo "📊 Monitoring Kafka lag..."
	@$(RUN_PYTHON) -c " \
import utils.kafka_utils as ku; \
import json; \
lag = ku.monitor_consumer_lag('fraud_detection_group', 'raw_transactions'); \
print(json.dumps(lag, indent=2)); \
"

kafka-cleanup-topics:
	@echo "🧹 Cleaning up unused Kafka topics..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "\
		$$topics = & '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --list; \
		foreach ($$t in $$topics) { \
			if ($$t -ne 'raw_transactions' -and $$t -ne 'fraud_predictions' -and -not $$t.StartsWith('__')) { \
				Write-Host '🗑️ Deleting topic:' $$t; \
				& '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --delete --topic $$t; \
			} \
		} \
	"
else
	@for topic in $$( $(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --list ); do \
		if [ "$$topic" != "raw_transactions" ] && [ "$$topic" != "fraud_predictions" ] && [[ ! "$$topic" =~ ^__ ]]; then \
			echo "🗑️ Deleting topic: $$topic"; \
			$(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic "$$topic"; \
		fi; \
	done
endif
	@echo "✅ Topic cleanup completed"

kafka-flush-messages:
	@echo "🗑️ Flushing all messages from Kafka topics..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "\
		& '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --delete --topic raw_transactions; \
		& '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --delete --topic fraud_predictions; \
		Start-Sleep -Seconds 2; \
		& '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --create --topic raw_transactions --partitions 1 --replication-factor 1; \
		& '$(KAFKA_BIN_DIR)kafka-topics.bat' --bootstrap-server localhost:9092 --create --topic fraud_predictions --partitions 1 --replication-factor 1; \
	"
else
	@$(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic raw_transactions 2>/dev/null || true
	@$(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic fraud_predictions 2>/dev/null || true
	@sleep 2
	@$(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --create --topic raw_transactions --partitions 1 --replication-factor 1
	@$(KAFKA_BIN_DIR)kafka-topics.sh --bootstrap-server localhost:9092 --create --topic fraud_predictions --partitions 1 --replication-factor 1
endif
	@echo "✅ All messages flushed - topics are now empty"

kafka-reset:
	@echo "🧹 Resetting Kafka data (destructive operation)..."
	@$(MAKE) kafka-stop
	@sleep 2
	@rm -rf runtime/kafka-logs
	@rm -f runtime/pids/kafka.pid
	@echo "✅ Kafka reset completed. Run 'make kafka-format' to reinitialize"

kafka-help:
	@echo "🔧 Native Kafka Commands Help"
	@echo "=================================================="
	@echo "Setup Commands:"
	@echo "  kafka-format     - Format Kafka storage (first time)"
	@echo "  kafka-start      - Start native Kafka broker"
	@echo "  kafka-start-bg   - Start broker in background"
	@echo "  kafka-stop       - Stop native Kafka broker"
	@echo "  kafka-topics     - Create fraud detection topics"
	@echo "  kafka-cleanup-topics - Remove unused topics"
	@echo ""
	@echo "Data Commands:"
	@echo "  kafka-producer-stream  - Start streaming producer (real data)"
	@echo "  kafka-producer-batch   - Start batch producer (real data)"
	@echo "  kafka-consumer         - Start batch ML consumer"
	@echo "  kafka-consumer-continuous - Start continuous ML consumer"
	@echo ""
	@echo "Monitoring Commands:"
	@echo "  kafka-check      - Check broker status"
	@echo "  kafka-monitor    - Monitor cluster health / consumer lag"
	@echo "  kafka-sample-scored - Show prediction analytics & statistics"
	@echo "  kafka-reset      - Reset all Kafka data"
	@echo "  kafka-help       - Show this help"

# ========================================================================================
# APACHE AIRFLOW ORCHESTRATION COMMANDS
# ========================================================================================

airflow-init:
	@echo "🔧 Initializing Apache Airflow..."
	@export PYTHONPATH="$(ROOT_DIR):$$PYTHONPATH" && \
	$(RUN_AIRFLOW) db migrate && \
	$(RUN_AIRFLOW) users create -u admin -p admin -r Admin -e admin@example.com -f Admin -l User && \
	$(MKDIR) $(AIRFLOW_HOME)/dags && find dags -name "*.py" -exec cp {} $(AIRFLOW_HOME)/dags/ \;
	@echo "✅ Airflow initialized successfully!"

airflow-start:
	@echo "🚀 Starting Airflow in standalone mode..."
	@echo "Checking for port conflicts..."
	@echo "Ensuring DAGs are copied..."
	@find dags -name "*.py" -exec cp {} $(AIRFLOW_HOME)/dags/ \; 2>/dev/null || true
	@echo "Starting Airflow standalone..."
	@export PYTHONPATH="$(ROOT_DIR):$$PYTHONPATH" && \
	$(RUN_AIRFLOW) standalone

airflow-kill:
	@echo "🛑 Killing all Airflow processes..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%airflow%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
	@powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
else
	@pkill -f airflow || echo "No Airflow processes found"
endif
	@echo "✅ All Airflow processes killed successfully!"

airflow-reset:
	@echo "🔄 Resetting Airflow database..."
	@$(MAKE) airflow-kill
	@rm -rf $(AIRFLOW_HOME)/airflow.db $(AIRFLOW_HOME)/logs/*
	@export PYTHONPATH="$(ROOT_DIR):$$PYTHONPATH" && \
	$(RUN_AIRFLOW) db migrate && \
	$(RUN_AIRFLOW) users create -u admin -p admin -r Admin -e admin@example.com -f Admin -l User
	@echo "✅ Airflow reset complete!"

# ========================================================================================
# DOCKER SERVICES COMMANDS
# ========================================================================================

docker-build:
	@echo "🐳 Building all Docker images..."
	docker compose build

docker-up:
	@echo "🐳 Starting all Docker services..."
	docker compose up -d mlflow-tracking

docker-down:
	@echo "🐳 Stopping all Docker services..."
	docker compose down

docker-data-pipeline:
	@echo "🐳 Running data pipeline in Docker..."
	docker compose run --rm data-pipeline

docker-model-pipeline:
	@echo "🐳 Running model pipeline in Docker..."
	docker compose run --rm model-pipeline

docker-inference-pipeline:
	@echo "🐳 Running inference pipeline in Docker..."
	docker compose run --rm inference-pipeline

docker-run-all:
	@echo "🐳 Running all pipelines in Docker sequentially..."
	docker compose run --rm data-pipeline
	docker compose run --rm model-pipeline
	docker compose run --rm inference-pipeline

docker-status:
	@echo "🐳 Showing Docker service status..."
	docker compose ps

# ========================================================================================
# S3 ORCHESTRATION COMMANDS
# ========================================================================================

s3-upload-data:
	@python -c "import os, sys, subprocess; [os.environ.update({l.split('=', 1)[0].strip(): l.split('=', 1)[1].strip()}) for l in open('.env') if '=' in l and not l.startswith('#')] if os.path.exists('.env') else None; [os.environ.pop(e) for e in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'] if e in os.environ and not os.path.exists(os.environ[e])]; print('[INFO] Uploading raw data to S3...'); raw_file = 'dataset/raw/fraudTrain.csv'; subprocess.run([sys.executable, 'scripts/upload_to_s3.py', '--file', raw_file]) if os.path.exists(raw_file) else print('[WARNING] Raw file not found: ' + raw_file); print('[INFO] Uploading processed data artifacts to S3...'); path = 'artifacts/data'; [subprocess.run([sys.executable, 'scripts/upload_to_s3.py', '--file', os.path.join(path, f)]) for f in os.listdir(path) if f.endswith('.parquet') or f.endswith('.csv')] if os.path.exists(path) else None"

s3-list:
	@python -c "import os, sys, boto3; [os.environ.update({l.split('=', 1)[0].strip(): l.split('=', 1)[1].strip()}) for l in open('.env') if '=' in l and not l.startswith('#')] if os.path.exists('.env') else None; [os.environ.pop(e) for e in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'] if e in os.environ and not os.path.exists(os.environ[e])]; bucket = os.getenv('S3_BUCKET'); prefix = '$(PREFIX)'; res = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=prefix); print(f'Objects in s3://{bucket}/{prefix}:'); [print(f\"  - {obj['Key']} ({obj['Size'] / (1024*1024):.2f} MB)\") for obj in res.get('Contents', [])]"

s3-clean:
	@python -c "import os, sys, boto3; [os.environ.update({l.split('=', 1)[0].strip(): l.split('=', 1)[1].strip()}) for l in open('.env') if '=' in l and not l.startswith('#')] if os.path.exists('.env') else None; [os.environ.pop(e) for e in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'] if e in os.environ and not os.path.exists(os.environ[e])]; bucket = os.getenv('S3_BUCKET'); b = boto3.resource('s3').Bucket(bucket); [print(f'[INFO] Deleting s3://{bucket}/{k}...') or b.objects.filter(Prefix=k).delete() for k in ['predictions/', 'mlflow-artifacts/']]"

s3-delete-prefix:
	@python -c "import os, sys, boto3; [os.environ.update({l.split('=', 1)[0].strip(): l.split('=', 1)[1].strip()}) for l in open('.env') if '=' in l and not l.startswith('#')] if os.path.exists('.env') else None; [os.environ.pop(e) for e in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'] if e in os.environ and not os.path.exists(os.environ[e])]; prefix = '$(PREFIX)'; (print('[ERROR] PREFIX is required. Usage: make s3-delete-prefix PREFIX=my-prefix') or sys.exit(1)) if not prefix else None; bucket = os.getenv('S3_BUCKET'); print(f'[INFO] Deleting all keys under s3://{bucket}/{prefix}...'); boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix).delete(); print('[SUCCESS] Done!')"

s3-smoke:
	@python -c "import os, sys, boto3; [os.environ.update({l.split('=', 1)[0].strip(): l.split('=', 1)[1].strip()}) for l in open('.env') if '=' in l and not l.startswith('#')] if os.path.exists('.env') else None; [os.environ.pop(e) for e in ['AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE'] if e in os.environ and not os.path.exists(os.environ[e])]; bucket = os.getenv('S3_BUCKET'); boto3.client('s3').head_bucket(Bucket=bucket); print(f'[SUCCESS] S3 Connection Successful! Bucket: s3://{bucket}')"
