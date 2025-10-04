#!/bin/bash

# Set minikube docker environment
eval $(minikube docker-env)

# Build Docker images
echo "Building Docker images..."
docker build -f Dockerfile -t gbr-ml:latest .
kubectl delete job data-processing-job
kubectl delete job training-job


# Delete existing deployments (ignore errors if they don't exist)
echo "Cleaning up existing deployments..."
kubectl delete deployment mlflow-server --ignore-not-found=true
kubectl delete deployment serving-deployment --ignore-not-found=true
kubectl delete service mlflow-service --ignore-not-found=true
kubectl delete service serving-service --ignore-not-found=true

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/serve.yaml

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/mlflow-server --timeout=300s
kubectl wait --for=condition=available deployment/serving-deployment --timeout=300s

echo "-  Deployments ready!"
echo ""
echo "-  Access MLflow UI:"
echo "   kubectl port-forward service/mlflow-service 5000:5000"
echo "   Then visit: http://localhost:5000"
echo ""
echo "-  Access Serving API:"
echo "   kubectl port-forward service/serving-service 8080:8080"
echo "   Then call: curl http://localhost:8080/predict"
echo ""
echo "-  Check status:"
echo "   kubectl get pods -l 'app in (mlflow-server,serving)'"