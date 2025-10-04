#!/bin/bash

echo "Deploying Airflow DAG..."

# Set Airflow home
export AIRFLOW_HOME=/home/harvey/airflow

# Copy DAG to Airflow
mkdir -p $AIRFLOW_HOME/dags
cp -r airflow/dags/* $AIRFLOW_HOME/dags/

echo "DAG copied to $AIRFLOW_HOME/dags/"

# Restart Airflow to pick up new DAG
echo "Restarting Airflow..."
pkill -f "airflow webserver" 2>/dev/null || true
pkill -f "airflow scheduler" 2>/dev/null || true
sleep 2

# Activate environment and start services
source airvenv/bin/activate
export AIRFLOW__CORE__LOAD_EXAMPLES=False
cd $AIRFLOW_HOME

airflow webserver --port 8081 --daemon
sleep 3
airflow scheduler --daemon

echo "Airflow DAG deployed and services restarted!"
echo "Visit http://localhost:8081 to see your ML pipeline DAG"