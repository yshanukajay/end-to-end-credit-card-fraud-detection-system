import os, sys
from airflow import DAG
from airflow.utils import timezone
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

# Ensure project root & /opt/app are in sys.path for Airflow container
for path in ['/opt/app', os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from utils.airflow_tasks import validate_input_data, run_data_pipeline

"""

============================ DAG ============================

Validate Input Data -> Run Data Pipeline Task

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
        dag_id = 'data_pipeline_dag',
        schedule_interval='*/5 * * * *',
        catchup=False,
        max_active_runs=1,
        default_args = default_arguments,
        description='Data Pipeline - Every 5 Minutes Sheduled',
        tags=['pyspark', 'mllib', 'mlflow', 'batch-processing']
        ) as dag:
    
    # Step 1
    validate_input_data_task = PythonOperator(
                                            task_id='validate_input_data',
                                            python_callable=validate_input_data,
                                            execution_timeout=timedelta(minutes=2)
                                            )

    # Step 2
    run_data_pipeline_task = PythonOperator(
                                            task_id='run_data_pipeline',
                                            python_callable=run_data_pipeline,
                                            execution_timeout=timedelta(minutes=15)
                                            )

    validate_input_data_task >> run_data_pipeline_task