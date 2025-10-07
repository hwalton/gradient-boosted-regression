# Gradient Boosted Regression

### Setup Kubernetes Cluster

```
minikube start
```

### Setup ML Pipelines
```
./scripts/rebuild-kube.sh
```

### Setup Airflow

```
./scripts/install-airflow.sh
./scripts/rebuild-airflow.sh
```

find the admin password with:
```
kubectl logs deployment/airflow-standalone | grep -A 5 -B 5 "admin"
```

### Run data processing:
```
./scripts/data-processing-job.sh
```

### Run training job:
```
./scripts/training-job.sh

### Run tests:
```
source venv/bin/activate
python -m pytest -q
```