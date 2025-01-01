#########################################################################
#   Copyright 2022 - 2023, KorrAI                                       #
#   ALL RIGHTS RESERVED.                                                #
#   This file is subject to the full copyright and disclaimer notice    #
#   included in a separate file in this directory.                      #
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
