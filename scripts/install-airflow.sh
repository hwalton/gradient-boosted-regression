# Clean up existing deployment
kubectl delete namespace airflow

# Deploy with Airflow 3.x
kubectl apply -f k8s/airflow-simple.yaml

# Create the PVC
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

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=airflow-standalone -n airflow --timeout=300s

# Upload your DAG
./scripts/rebuild-airflow.sh