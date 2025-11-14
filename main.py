#########################################################################
#   Copyright 2024 - 2025, KorrAI                                       #
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
#   This file is the entry point for the service                        #
#                                                                       #
#########################################################################

import warnings
warnings.simplefilter(action='ignore')
from src.stamps import Stamps
import json

def run_service(args):
    stamps = Stamps()
    stamps.set_param(args)
    stamps.start()

def read_input():
    data = {}
    with open('input.json') as fread:
        data = json.load(fread)
    return data

if __name__ == "__main__":
    print("==========================================")
    print ("Starting Stamps Service !!!!")
    print("==========================================")
    args = read_input()
    run_service(args)
