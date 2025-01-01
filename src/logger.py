#########################################################################
#   Copyright 2022 - 2025, KorrAI                                       #
#   ALL RIGHTS RESERVED.                                                #
#   This file is subject to the full copyright and disclaimer notice    #
#   included in a separate file in this directory.                      #
#########################################################################
#                                                                       #
#   This file contains the implementation of Custom logging.            #
#                                                                       #
#########################################################################

import queue
from logging.handlers import QueueHandler, QueueListener
from datetime import datetime
import logging
import os

logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('h5py').setLevel(logging.CRITICAL)
logging.getLogger('google.cloud.storage').setLevel(logging.CRITICAL)

# Setup queue
loggerQueue = queue.Queue(-1)  # no limit on size
queueHandler = QueueHandler(loggerQueue)

# Setup logger
appLogger = logging.getLogger()
appLogger.addHandler(queueHandler)
appLogger.setLevel(logging.INFO)

# LOG FORMAT
formatter = logging.Formatter('%(asctime)s :: %(levelname)-7s - %(message)s')

# Setup console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

def get_app_listener(path,loggerQueue = loggerQueue, console_handler = console_handler):
    # Setup file handler
    os.makedirs(path, exist_ok=True)
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler('{}/{}_log.log'.format(path,datetime_str))
    file_handler.setFormatter(formatter)

    # Attach both console_handler, file_handler to the quque
    appListener = QueueListener(loggerQueue, console_handler,file_handler)
    return appListener
