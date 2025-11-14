#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   This program is free software: you can redistribute it and/or       #
#   modify it under the terms of the GNU General Public License as      #
#   published by the Free Software Foundation, either version 3 of      #
#   the License, or (at your option) any later version.                 #
#                                                                       #
#   This program is distributed in the hope that it will be useful,     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
#   GNU General Public License for more details.                        #
#                                                                       #
#   You should have received a copy of the GNU General Public License   #
#   along with this program. If not, see <https://www.gnu.org/licenses/>.#
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
