#!/bin/bash
# Create scripts/upload-dag.sh

echo "Uploading DAG to Airflow..."

POD_NAME=$(kubectl get pods -n airflow -l app=airflow-standalone -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "❌ No Airflow pod found"
    exit 1
fi

echo "Found Airflow pod: $POD_NAME"

# Create dags directory if it doesn't exist
echo "Creating dags directory..."
kubectl exec -n airflow $POD_NAME -- mkdir -p /opt/airflow/dags

# Copy the DAG
echo "Copying DAG to Airflow pod..."
kubectl cp airflow/dags/ml_pipeline_dag.py airflow/$POD_NAME:/opt/airflow/dags/

echo "✅ DAG uploaded successfully"

# Check if it was detected
echo "Checking if DAG is detected..."
sleep 5
kubectl exec -n airflow $POD_NAME -- airflow dags list | grep ml_pipeline_with_performance_monitoring || echo "DAG not detected yet, may take a few minutes..."

echo "🌐 Access Airflow UI at: http://localhost:8080"
echo "📝 Port forward command: kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow"