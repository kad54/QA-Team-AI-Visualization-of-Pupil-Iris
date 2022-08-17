import logging
import os


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING,
    )
    return logging.getLogger(name)


LOGGER = set_logging(
    __name__
)  # define globally (used in train.py, val.py, detect.py, etc.)
