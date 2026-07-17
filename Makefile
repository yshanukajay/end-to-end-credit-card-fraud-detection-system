.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help mlflow-ui stop-all \
         airflow-init airflow-webserver airflow-scheduler airflow-stop

# Get the directory of this Makefile (always ends with a slash)
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Airflow home directory setup locally
export AIRFLOW_HOME := $(abspath $(ROOT_DIR)airflow)

# Default Python interpreter
PYTHON = python
MLFLOW_PORT ?= 5001

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies in active conda environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up artifacts"
	@echo "  make mlflow-ui           - Launch MLflow UI"
	@echo "  make stop-all            - Stop running MLflow servers"
	@echo "  make airflow-init        - Initialize local Airflow DB and create admin user"
	@echo "  make airflow-webserver   - Start Airflow Webserver on port 8080"
	@echo "  make airflow-scheduler   - Start Airflow Scheduler"
	@echo "  make airflow-stop        - Stop running Airflow servers"

# Install project dependencies in the active conda environment
install:
	@echo "Installing project dependencies in the active conda environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(ROOT_DIR)requirements.txt
	@echo "Installation completed successfully!"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	$(PYTHON) -c "import shutil, os; [shutil.rmtree(os.path.join('$(ROOT_DIR)', p), ignore_errors=True) for p in ['artifacts/models', 'artifacts/evaluation', 'artifacts/predictions', 'artifacts/encode', 'mlruns']]"
	@echo "Cleanup completed!"

# Run data pipeline
data-pipeline:
	@echo "Start running data pipeline..."
	$(PYTHON) $(ROOT_DIR)pipelines/data_pipeline.py
	@echo "Data pipeline completed successfully!"

.PHONY: data-pipeline-rebuild
data-pipeline-rebuild:
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(ROOT_DIR)'); from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	$(PYTHON) $(ROOT_DIR)pipelines/train_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	$(PYTHON) $(ROOT_DIR)pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	$(PYTHON) $(ROOT_DIR)pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	$(PYTHON) $(ROOT_DIR)pipelines/train_pipeline.py
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	$(PYTHON) $(ROOT_DIR)pipelines/streaming_inference_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	mlflow ui --backend-store-uri sqlite:///$(ROOT_DIR)mlflow.db --default-artifact-root $(ROOT_DIR)mlruns --host 127.0.0.1 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
ifeq ($(OS),Windows_NT)
	@echo "Finding and stopping MLflow processes on port $(MLFLOW_PORT)..."
	@powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort $(MLFLOW_PORT) -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
else
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
endif
	@echo "✅ All MLflow servers have been stopped"

# Initialize Airflow DB
airflow-init:
	@echo "Initializing Airflow database at $(AIRFLOW_HOME)..."
	$(PYTHON) $(ROOT_DIR)utils/run_airflow.py db migrate

# Start Airflow Webserver
airflow-webserver:
	@echo "Starting Airflow Webserver on port 8080..."
	$(PYTHON) $(ROOT_DIR)utils/run_airflow.py webserver --port 8080

# Start Airflow Scheduler
airflow-scheduler:
	@echo "Starting Airflow Scheduler..."
	$(PYTHON) $(ROOT_DIR)utils/run_airflow.py scheduler

# Stop all running Airflow processes
airflow-stop:
	@echo "Stopping all Airflow processes..."
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-CimInstance Win32_Process -Filter \"CommandLine Like '%airflow%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"
	@powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue"
else
	@-pkill -f "airflow" || true
endif
	@echo "✅ All Airflow processes stopped"