import logging

from utils.utils import load_processed_data

logger = logging.getLogger(__name__)

class Cfg:
    """Configuration for training"""
    processed_dir: str = "data/processed"

def main():
    """
    Minimal training entrypoint: loads processed data and prints basic info.
    Extend this function to add model training, evaluation, and persistence.
    """

    processed_dir = Cfg.processed_dir

    try:
        X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    except FileNotFoundError as exc:
        logger.error("Could not load processed data: %s", exc)
        raise

    logger.info("Loaded processed data from %s", processed_dir)
    logger.info("X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)
    logger.info("y_train length: %d, y_test length: %d", len(y_train), len(y_test))

    # Placeholder: return loaded datasets for downstream training code
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Configure logging to show INFO messages on the console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()