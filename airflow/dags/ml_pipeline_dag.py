from datetime import datetime, timedelta
from airflow import DAG
# Correct import path for Airflow 3.x
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
# Updated import paths for Airflow 3.x
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
# Import Kubernetes objects
from kubernetes.client import models as k8s
import json
import os

# Helper function to replace days_ago
def days_ago(days):
    return datetime.now() - timedelta(days=days)

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

# Create DAG - Updated for Airflow 3.x
dag = DAG(
    'ml_pipeline_with_performance_monitoring',
    default_args=default_args,
    description='ML Pipeline with Performance Monitoring and Retraining',
    schedule=timedelta(hours=24),  # Changed from schedule_interval to schedule
    catchup=False,
    tags=['ml', 'performance-monitoring', 'retraining']
)

# Kubernetes configuration - Updated to use proper K8s objects
volume_mount = k8s.V1VolumeMount(
    name='shared-storage',
    mount_path='/app/shared'
)

volume = k8s.V1Volume(
    name='shared-storage',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name='ml-data-pvc'
    )
)

def check_performance_status(**context):
    """Check performance monitoring report and decide next step"""
    performance_report_path = "/app/shared/monitoring/model_performance_report.json"
    
    # For local testing, use a different path
    if not os.path.exists(performance_report_path):
        performance_report_path = "shared/monitoring/model_performance_report.json"
    
    if not os.path.exists(performance_report_path):
        print("No performance report found, proceeding with retraining")
        return 'full_retrain_pipeline'
    
    try:
        with open(performance_report_path, 'r') as f:
            performance_report = json.load(f)
        
        overall_degraded = performance_report.get('overall_degraded', False)
        degraded_metrics = performance_report.get('degraded_metrics', [])
        
        print(f"Performance degraded: {overall_degraded}")
        print(f"Degraded metrics: {degraded_metrics}")
        
        if overall_degraded:
            return 'full_retrain_pipeline'
        else:
            return 'skip_retraining'
            
    except Exception as e:
        print(f"Error reading performance report: {e}")
        return 'full_retrain_pipeline'  # Default to retraining on error

# Environment variables
env_vars = [
    k8s.V1EnvVar(name='MLFLOW_TRACKING_URI', value='http://mlflow-service:5000')
]

# 1. Generate Fresh Drifted Data (always runs first)
generate_data_task = KubernetesPodOperator(
    task_id='generate_drifted_data',
    name='data-generation-pod',
    namespace='default',
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.data.data'],
    env_vars=env_vars,
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 2. Performance Monitoring Task (uses fresh data)
performance_monitoring_task = KubernetesPodOperator(
    task_id='monitor_performance',
    name='performance-monitoring-pod',
    namespace='default',
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.monitoring.monitor'],
    env_vars=env_vars,
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 3. Check Performance Status (Python function)
check_performance_task = BranchPythonOperator(
    task_id='check_performance_status',
    python_callable=check_performance_status,
    dag=dag
)

# 4a. Full Retraining Pipeline (if performance degraded)
full_retrain_task = KubernetesPodOperator(
    task_id='full_retrain_pipeline',
    name='full-retrain-pod',
    namespace='default',
    image='gbr-ml:latest',
    cmds=['python', '-m', 'src.training.train'],
    env_vars=env_vars,
    volumes=[volume],
    volume_mounts=[volume_mount],
    is_delete_operator_pod=True,
    dag=dag
)

# 4b. Restart Serving (to use new model)
restart_serving_task = BashOperator(
    task_id='restart_serving',
    bash_command='kubectl rollout restart deployment/serving-deployment',
    dag=dag
)

# 4c. Skip Task (if performance is acceptable)
skip_task = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "Model performance acceptable, skipping retraining"',
    dag=dag
)

# TASK DEPENDENCIES:
# 1. Always generate fresh drifted data first
# 2. Monitor performance on that fresh data
# 3. Branch based on performance results
# 4. Either retrain or skip

generate_data_task >> performance_monitoring_task >> check_performance_task

# Branch based on performance monitoring
check_performance_task >> [full_retrain_task, skip_task]

# If retraining, restart serving after training
full_retrain_task >> restart_serving_task