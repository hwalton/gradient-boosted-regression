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
    'ml_pipeline_with_performance_monitoring',
    default_args=default_args,
    description='ML Pipeline with Performance Monitoring and Retraining',
    schedule_interval=timedelta(hours=24),
    catchup=False,
    tags=['ml', 'performance-monitoring', 'retraining']
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

def check_performance_status(**context):
    """Check performance monitoring report and decide next step"""
    performance_report_path = "/app/shared/monitoring/model_performance_report.json"
    
    # For local testing, use a different path
    if not os.path.exists(performance_report_path):
        performance_report_path = "shared/monitoring/model_performance_report.json"
    
    if not os.path.exists(performance_report_path):
        print("No performance report found, proceeding with retraining")
        return 'retrain_pipeline'
    
    try:
        with open(performance_report_path, 'r') as f:
            performance_report = json.load(f)
        
        overall_degraded = performance_report.get('overall_degraded', False)
        degraded_metrics = performance_report.get('degraded_metrics', [])
        
        print(f"Performance degraded: {overall_degraded}")
        print(f"Degraded metrics: {degraded_metrics}")
        
        if overall_degraded:
            return 'retrain_pipeline'
        else:
            return 'skip_retraining'
            
    except Exception as e:
        print(f"Error reading performance report: {e}")
        return 'retrain_pipeline'  # Default to retraining on error

# 1. Performance Monitoring Task
performance_monitoring_task = KubernetesPodOperator(
    task_id='monitor_performance',
    name='performance-monitoring-pod',
    namespace='default',
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.monitoring.monitor'],
    env_vars=[
        {'name': 'MLFLOW_TRACKING_URI', 'value': 'http://mlflow-service:5000'}
    ],
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 2. Check Performance Status (Python function)
check_performance_task = BranchPythonOperator(
    task_id='check_performance_status',
    python_callable=check_performance_status,
    dag=dag
)

# 3a. Data Processing (if retraining needed)
data_processing_task = KubernetesPodOperator(
    task_id='retrain_pipeline',
    name='data-processing-pod',
    namespace='default',
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.data.data'],
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
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.training.train'],
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

# 4. Skip Task (if performance is acceptable)
skip_task = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "Model performance acceptable, skipping retraining"',
    dag=dag
)

# Define task dependencies
performance_monitoring_task >> check_performance_task

# Branch based on performance monitoring
check_performance_task >> [data_processing_task, skip_task]

# If retraining, continue with training and serving restart
data_processing_task >> training_task >> restart_serving_task