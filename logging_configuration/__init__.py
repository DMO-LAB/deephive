import logging 
import os
import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler
from decouple import config as conf
neptune_key = conf("neptune_secret_access_key", default='')

def configure_logger(name, local_only=False, level=logging.INFO):
    """
    Configure the logger.
    :param name: The name of the logger.
    :param local_only: Whether to only log locally.
    :return: The logger.
    """
    # create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create the formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create the console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create the file handler
    fh = logging.FileHandler(f'logs/{name}.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # create the neptune handler
    if not local_only:
        run = neptune.init(
            project="elo/deephive",
            api_token=neptune_key,
            source_files=['*.py', 'config.yml'])
        nh = NeptuneHandler(run=run, level=level)
        nh.setLevel(level)
        nh.setFormatter(formatter)
        logger.addHandler(nh)
        return logger, run
    return logger, None
        