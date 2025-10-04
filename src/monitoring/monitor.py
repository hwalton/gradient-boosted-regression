import mlflow

def main():
    """
    Run data pipeline and log as MLflow run (nested if called from an active run).
    """
    mlflow.set_experiment("gradient_boosted_regression")
    # choose nested behavior depending on whether a run is already active
    if mlflow.active_run() is None:
        with mlflow.start_run(run_name="data_monitoring"):
            return False

    else:
        with mlflow.start_run(run_name="data_monitoring", nested=True):
            return False

if __name__ == "__main__":
    # When running this module directly, let main() decide run nesting.
    main()