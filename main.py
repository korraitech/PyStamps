#########################################################################
#   Copyright 2024 - 2025, KorrAI                                       #
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
