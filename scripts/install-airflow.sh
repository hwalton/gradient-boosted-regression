#!/bin/bash
# filepath: /home/harvey/Git/gradient-boosted-regression/scripts/install-airflow.sh

echo "Setting up Airflow in default namespace..."

# Clean up any existing airflow namespace
kubectl delete namespace airflow --ignore-not-found=true

# Clean up existing Airflow resources in default namespace
kubectl delete deployment airflow-standalone --ignore-not-found=true
kubectl delete deployment airflow-postgresql --ignore-not-found=true
kubectl delete service airflow-webserver --ignore-not-found=true
kubectl delete service airflow-postgresql --ignore-not-found=true

# Deploy Airflow in default namespace
kubectl apply -f k8s/airflow-rbac.yaml
kubectl apply -f k8s/airflow-simple.yaml

# Get the pod name
POD_NAME=$(kubectl get pods -l app=airflow-standalone -o jsonpath='{.items[0].metadata.name}')

# Initialize the database
kubectl exec $POD_NAME -- airflow db migrate > /dev/null

echo "Waiting for Airflow to be ready..."
kubectl wait --for=condition=available deployment/airflow-standalone --timeout=600s

echo "✅ Airflow is ready in default namespace!"
echo "Upload DAG: ./scripts/rebuild-airflow.sh"
echo "Port forward: kubectl port-forward svc/airflow-webserver 8080:8080"

