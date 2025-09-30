import logging

from training import train
from data import data

def main():
    data.main()
    train.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()