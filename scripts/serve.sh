#!/bin/bash

# Build the serving image
eval $(minikube docker-env)
docker build -f Dockerfile.serve -t gbr-serve:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/serve.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/serving-deployment --timeout=300s

echo "Serving deployment ready!"
echo "To access the service locally:"
echo "kubectl port-forward service/serving-service 8080:8080"
echo ""
echo "Then test with:"
echo "curl http://localhost:8080/health"