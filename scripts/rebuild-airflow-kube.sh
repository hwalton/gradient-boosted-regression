#!/bin/bash
# filepath: /home/harvey/Git/gradient-boosted-regression/scripts/rebuild-airflow-kube.sh

echo "Cleaning up Airflow..."
kubectl delete namespace airflow

echo "Deploying Airflow..."
kubectl apply -f k8s/airflow-simple.yaml

echo "Creating fresh PVC in airflow namespace..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-data-pvc
  namespace: airflow
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: standard
EOF

echo "Waiting for PVC to be bound..."
kubectl wait --for=condition=bound pvc/ml-data-pvc -n airflow --timeout=60s

echo "Checking status..."
kubectl get pvc -n airflow
kubectl get pods -n airflow