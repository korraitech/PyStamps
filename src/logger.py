#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   This program is free software: you can redistribute it and/or       #
#   modify it under the terms of the European Space Agency Public       #
#   License (ESA-PL) Permissive (Type 3) - v2.4.                        #
#                                                                       #
#   This program is distributed in the hope that it will be useful,     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
#   ESA-PL Permissive (Type 3) - v2.4 for more details.                 #
#                                                                       #
#   You should have received a copy of the license along with this      #
#   program. If not, see the ESA-PL v2.4 license at:                    #
#   https://essr.esa.int/license/european-space-agency-public-license-v2-4-permissive-type-3
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
