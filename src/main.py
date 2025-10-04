import logging
import mlflow

from src.training import train
from src.data import data
from src.monitoring import monitor

def main():
    mlflow.set_experiment("gradient_boosted_regression")
    # top-level run for full pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        data.main()
        retrain = monitor.main()
        print(f"Retrain flag: {retrain}")
        if retrain:
            logging.warning("Drift detected - retraining model")
            train.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()