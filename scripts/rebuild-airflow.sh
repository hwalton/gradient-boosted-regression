#!/bin/bash
# filepath: /home/harvey/Git/gradient-boosted-regression/scripts/rebuild-airflow.sh

echo "Uploading DAG to Airflow..."

# Look for Airflow pod in default namespace
POD_NAME=$(kubectl get pods -l app=airflow-standalone -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "❌ No Airflow pod found in default namespace"
    exit 1
fi

echo "Found Airflow pod: $POD_NAME"

# Create dags directory if it doesn't exist
echo "Ensuring dags directory exists..."
kubectl exec $POD_NAME -- mkdir -p /opt/airflow/dags 2>/dev/null || true

# Copy the DAG
echo "Copying DAG to Airflow pod..."
kubectl cp airflow/dags/ml_pipeline_dag.py $POD_NAME:/opt/airflow/dags/

echo "✅ DAG uploaded successfully"

# Check for import errors
echo "Checking for import errors..."
kubectl exec $POD_NAME -- airflow dags list-import-errors

# List all DAGs
echo "Listing all DAGs..."
kubectl exec $POD_NAME -- airflow dags list

echo "🌐 Access Airflow UI at: http://localhost:8080"
echo "📝 Port forward command: kubectl port-forward svc/airflow-webserver 8080:8080"