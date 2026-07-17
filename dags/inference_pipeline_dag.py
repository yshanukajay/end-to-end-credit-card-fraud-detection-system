import os, sys
from airflow import DAG
from airflow.utils import timezone
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.airflow_tasks import validate_trained_model, run_inference_pipeline

"""

============== DAG ============================

Validate Trained Model -> Run Train Pipeline Task

"""

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
    
    # Step 1
    validate_trained_model_task = PythonOperator(
                                            task_id='validate_trained_model',
                                            python_callable=validate_trained_model,
                                            execution_timeout=timedelta(minutes=2)
                                            )

    # Step 2
    run_inference_pipeline_task = PythonOperator(
                                            task_id='run_inference_pipeline',
                                            python_callable=run_inference_pipeline,
                                            execution_timeout=timedelta(minutes=2)
                                            )

    validate_trained_model_task >> run_inference_pipeline_task