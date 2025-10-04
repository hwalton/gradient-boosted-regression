from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import json
import os

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'ml_pipeline_with_drift_detection',
    default_args=default_args,
    description='ML Pipeline with Drift Detection and Retraining',
    schedule_interval=timedelta(hours=24),
    catchup=False,
    tags=['ml', 'drift-detection', 'retraining']
)

# Kubernetes configuration
volume_mount = {
    'name': 'shared-storage',
    'mount_path': '/app/shared'
}

volume = {
    'name': 'shared-storage',
    'persistent_volume_claim': {'claim_name': 'ml-data-pvc'}
}

def check_drift_status(**context):
    """Check drift report and decide next step"""
    drift_report_path = "/tmp/ml-data/monitoring/drift_report.json"
    
    # For local testing, use a different path
    if not os.path.exists(drift_report_path):
        drift_report_path = "shared/monitoring/drift_report.json"
    
    if not os.path.exists(drift_report_path):
        print("No drift report found, proceeding with retraining")
        return 'retrain_pipeline'
    
    try:
        with open(drift_report_path, 'r') as f:
            drift_report = json.load(f)
        
        overall_drift = drift_report.get('overall_drift', False)
        drift_score = drift_report.get('drift_score', 0)
        
        print(f"Drift detected: {overall_drift}, Drift score: {drift_score}")
        
        if overall_drift and drift_score > 0.3:  # Retrain if >30% features drifted
            return 'retrain_pipeline'
        else:
            return 'skip_retraining'
            
    except Exception as e:
        print(f"Error reading drift report: {e}")
        return 'retrain_pipeline'  # Default to retraining on error

# 1. Drift Detection Task
drift_detection_task = KubernetesPodOperator(
    task_id='detect_drift',
    name='drift-detection-pod',
    namespace='default',
    image='gbr-data:latest',  # Use same image as data processing
    cmds=['python', '-m', 'monitoring.drift_detection'],
    env_vars=[
        {'name': 'MLFLOW_TRACKING_URI', 'value': 'http://mlflow-service:5000'}
    ],
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 2. Check Drift Status (Python function)
check_drift_task = BranchPythonOperator(
    task_id='check_drift_status',
    python_callable=check_drift_status,
    dag=dag
)

# 3a. Data Processing (if drift detected)
data_processing_task = KubernetesPodOperator(
    task_id='retrain_pipeline',
    name='data-processing-pod',
    namespace='default',
    image='gbr-data:latest',
    cmds=['python', '-m', 'data.data'],
    env_vars=[
        {'name': 'MLFLOW_TRACKING_URI', 'value': 'http://mlflow-service:5000'}
    ],
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 3b. Training Task
training_task = KubernetesPodOperator(
    task_id='train_model',
    name='training-pod',
    namespace='default',
    image='gbr-train:latest',
    cmds=['python', '-m', 'training.train'],
    env_vars=[
        {'name': 'MLFLOW_TRACKING_URI', 'value': 'http://mlflow-service:5000'}
    ],
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 3c. Restart Serving (to use new model)
restart_serving_task = BashOperator(
    task_id='restart_serving',
    bash_command='kubectl rollout restart deployment/serving-deployment',
    dag=dag
)

# 4. Skip Task (if no significant drift)
skip_task = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "No significant drift detected, skipping retraining"',
    dag=dag
)

# Define task dependencies
drift_detection_task >> check_drift_task

# Branch based on drift detection
check_drift_task >> [data_processing_task, skip_task]

# If retraining, continue with training and serving restart
data_processing_task >> training_task >> restart_serving_task