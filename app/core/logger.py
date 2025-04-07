import logging
import os
from datetime import datetime


def logger(callable_name):
    LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(callable_name)
