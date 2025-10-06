from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 0,
}

with DAG(
    dag_id="manual_training_job",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["manual", "training"],
    description="Manually triggered DAG to run the training job on Kubernetes",
) as dag:

    shared_volume = k8s.V1Volume(
        name="shared-storage",
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="ml-data-pvc"),
    )
    shared_mount = k8s.V1VolumeMount(name="shared-storage", mount_path="/app/shared")

    training = KubernetesPodOperator(
        task_id="run_training",
        name="run-training",
        namespace="default",
        image="ghcr.io/hwalton/gbr-ml:latest",
        image_pull_policy="IfNotPresent",
        cmds=["python"],
        arguments=["-m", "src.training.train"],
        volumes=[shared_volume],
        volume_mounts=[shared_mount],
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=False,
    )

    training