# Gradient Boosted Regression

# Setup

1. Ensure you have a Kaggle account and have set up your API token as per [Kaggle API documentation](https://www.kaggle.com/docs/api#authentication).

2. Set up and activate Conda environment:
   ```bash
   conda env create -f conda.yml
   conda activate gradient-boosted-regressor
   ```

3. Launch MLflow UI:
   ```bash
   mlflow ui --port 5000 --host 127.0.0.1
   ```
   Access the UI at `http://localhost:5000`.

# Install airflow
https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-sources.html
## Set AIRFLOW_HOME
`export AIRFLOW_HOME=/home/${USER}/airflow`
`mkdir -p $AIRFLOW_HOME`

## Create and activate a new virtual environment
`python -m venv airvenv`
`source airvenv/bin/activate`

## Install Apache Airflow with Kubernetes provider
`pip install -r requirements.airflow.txt`

## Initialize the Airflow database
`airflow db init`

## Start Airflow webserver and scheduler
airflow webserver --port 8081
airflow scheduler

# Kubernetes Setup
./scripts/rebuild-deployments.sh
./scripts/rebuild-jobs.sh

# Run ML pipeline
kubectl apply -f k8s-jobs/data-processing-job.yaml
kubectl apply -f k8s-jobs/training-job.yaml

{"admin": "SURFhSAssN3eEHrt"}