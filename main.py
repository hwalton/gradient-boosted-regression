import logging
import mlflow

from training import train
from data import data
from monitoring import monitor

def main():
    mlflow.set_experiment("gradient_boosted_regression")
    # top-level run for full pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        data.main()
        ok = monitor.main()
        if not ok:
            logging.warning("Drift detected - retraining model")
        train.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()