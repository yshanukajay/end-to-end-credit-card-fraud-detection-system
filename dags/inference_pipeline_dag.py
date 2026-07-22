import os, sys
from airflow import DAG
from airflow.utils import timezone
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Ensure project root & /opt/app are in sys.path for Airflow container
for path in ['/opt/app', os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.airflow_tasks import run_inference_pipeline

"""

============== DAG ============================

Check Model Exists (short-circuit skip if not) -> Run Inference Pipeline Task

"""


def check_model_exists() -> bool:
    """
    Returns True if a trained model is available, False to short-circuit (skip) the DAG.
    This prevents noisy hard-failures every minute while training is pending.
    """
    import logging
    logger = logging.getLogger(__name__)

    candidate_paths = [
        '/opt/app/artifacts/models/spark_tuned_model',
        '/opt/app/artifacts/models/xgboost_tuned_model.pkl',
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            logger.info(f"✅ Model found at: {path} — proceeding with inference")
            return True

    logger.warning(
        "⏭ No trained model found in %s — skipping inference run. "
        "Run the train_pipeline_dag first.", candidate_paths
    )
    return False  # ShortCircuitOperator will skip all downstream tasks


default_arguments = {
                    'owner' : 'ML Engineering Team',
                    'depends_on_past' : False,
                    'start_date': timezone.datetime(2025, 9, 14, 10, 0),
                    'email_on_failure': False,
                    'email_on_retry': False,
                    'retries': 0,
                    }

with DAG(
        dag_id = 'inference_pipeline_dag',
        schedule_interval='* * * * *', # Every 1 minute
        catchup=False,
        max_active_runs=1,
        default_args = default_arguments,
        description='Inference Pipeline - Scheduled Every Minute',
        tags=['pyspark', 'mllib', 'mlflow', 'batch-processing']
        ) as dag:

    # Step 1 — Short-circuit: skip entire DAG run if no model is ready yet
    check_model_task = ShortCircuitOperator(
                                        task_id='check_model_exists',
                                        python_callable=check_model_exists,
                                        execution_timeout=timedelta(minutes=1)
                                        )

    # Step 2 — Run inference (only reached if model exists)
    run_inference_pipeline_task = PythonOperator(
                                            task_id='run_inference_pipeline',
                                            python_callable=run_inference_pipeline,
                                            execution_timeout=timedelta(minutes=2)
                                            )

    check_model_task >> run_inference_pipeline_task