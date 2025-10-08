# Gradient Boosted Regression

This project deploys a Scikit-Learn Gradient Boosted Regression model of the California housing dataset, with data processing and training pipelines, and a Flask API to call the model for predictions.

It integrates MLflow, Apache Airflow, Docker, Helm, and Kubernetes for orchestration and deployment.


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
```

### Run tests:
```
source venv/bin/activate
python -m pytest -q
```