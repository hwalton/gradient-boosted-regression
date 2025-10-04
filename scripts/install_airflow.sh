# Activate environment
source airvenv/bin/activate

# Check what's running
ps aux | grep airflow

# Kill processes manually if needed
# pkill -f airflow

# Clean up
rm -rf ~/airflow /home/harvey/airflow

# Set environment
export AIRFLOW_HOME=/home/harvey/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False
mkdir -p $AIRFLOW_HOME

# Install step by step
pip uninstall apache-airflow apache-airflow-providers-cncf-kubernetes -y
pip install apache-airflow-3.1.0-source.tar.gz
pip install apache-airflow-providers-cncf-kubernetes

# Test
python -c "from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator; print('Success!')"

# Initialize
airflow db init