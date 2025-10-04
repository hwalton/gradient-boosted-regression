#!/bin/bash

echo "Deploying Airflow DAG..."

# Set Airflow home
export AIRFLOW_HOME=/home/${USER}/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# Activate environment
source airvenv/bin/activate

# Test the correct import path (don't test the wrong one anymore)
echo "Testing Kubernetes provider import..."
python -c "
try:
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    print('✅ Kubernetes provider available')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Copy DAG to Airflow
mkdir -p $AIRFLOW_HOME/dags
cp -r airflow/dags/* $AIRFLOW_HOME/dags/

echo "DAG copied to $AIRFLOW_HOME/dags/"

# Test DAG import after fixing
echo "Testing DAG import..."
cd $AIRFLOW_HOME
python -c "
import sys
sys.path.append('dags')
try:
    import ml_pipeline_dag
    print('✅ DAG imports successfully')
except Exception as e:
    print(f'❌ DAG import failed: {e}')
    print('Check the import paths in your DAG file')
    exit(1)
"

# # Kill existing processes
# pkill -TERM -f airflow
# sleep 3
# pkill -9 -f airflow


# Start in standalone mode
echo "Starting Airflow in standalone mode..."
cd $AIRFLOW_HOME

# Run standalone in background
nohup airflow standalone > airflow.log 2>&1 &

echo "✅ Airflow started in standalone mode!"
echo "Visit http://localhost:8080 to see your ML pipeline DAG"
echo "Check logs: tail -f $AIRFLOW_HOME/airflow.log"

# Wait and check if it's running
sleep 5
if pgrep -f "airflow" > /dev/null; then
    echo "✅ Airflow processes are running"
else
    echo "❌ Airflow failed to start, check logs"
fi