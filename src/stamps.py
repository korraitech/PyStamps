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
#   This file contains the implementation of pystamps.                  #
#                                                                       #
#########################################################################


from .logger import appLogger, get_app_listener
from .mtprep import mt_prep
from .steps import run_stamps_steps
from .step.utils import read_h5
from datetime import datetime
from tqdm import tqdm
import numpy as np
import csv
import os


class Stamps():
    def __init__(self) -> None:
        self.__TEMP_DIR:str = os.path.join(os.getcwd(),"SERVICE_LOG")

    def set_param(self, param={}) -> None:
        appLogger.info(">>>>>>>>>>>>>>>> {}::{}".format(
            type(self).__name__, "set_param"))
        self.__PARAM_STM: dict = param["Pstm"]
        self.__PARAM_RDT: str = param["Prdt"]
        self.__PARAM_INP: str = param["Pinp"]
        self.__PARAM_OUT: str = param["Pout"]
        os.makedirs(self.__PARAM_OUT, exist_ok=True)

    def __stamps_processor(self):
        appLogger.info(">>>>>>>>>>>>>>>> {}::{}".format(
            type(self).__name__, "__stamps_processor"))
        appLogger.info(self.__PARAM_STM)

        # RUN mt_prep
        mt_prep(
            master=self.__PARAM_RDT,
            datadir=self.__PARAM_INP,
            workdir=self.__PARAM_OUT,
            da_thresh=self.__PARAM_STM["ADT"],
            prg=self.__PARAM_STM["NRP"],
            paz=self.__PARAM_STM["NAP"],
            overlap_rg=self.__PARAM_STM["RPO"],
            overlap_az=self.__PARAM_STM["APO"]
        )

        # RUN stamps steps
        run_stamps_steps(self.__PARAM_OUT,self.__PARAM_STM["NRP"],self.__PARAM_STM["NAP"])

    def __hdf5_to_csv(self):
        appLogger.info(">>>>>>>>>>>>>>>> {}::{}".format(
            type(self).__name__, "__hdf5_to_csv"))
        ps_data = read_h5(os.path.join(self.__PARAM_OUT,"ps_plot.h5"))
        header = ['lng','lat','point_id','coherence','avg_deformation_velocity']
        header += [datetime.fromtimestamp(int( day * 86400)).strftime('%Y-%m-%d') for day in ps_data['day']]
        with open(os.path.join(self.__PARAM_OUT,'ground_displacement.csv'), 'w',
                    newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            lonlat = np.array(ps_data['lonlat'])
            coh_ps = np.array(ps_data['coh_ps'], dtype=float)
            ph_disp = np.round(np.array(ps_data['ph_disp'], dtype=float), 7)
            ph_mm = np.round(np.array(ps_data['ph_mm'], dtype=float), 7)
            indices = np.arange(len(lonlat)).reshape(-1, 1)
            data = np.hstack((lonlat, indices, coh_ps.reshape(-1, 1), ph_disp.reshape(-1, 1), ph_mm))
            for row in tqdm(data):
                writer.writerow(row.tolist())


    def start(self) -> None:
        appListener = get_app_listener(path = self.__TEMP_DIR)
        appListener.start()

        appLogger.info(
            "++++++++++++++++ {}::{}".format(type(self).__name__, "start"))
        self.__stamps_processor()
        self.__hdf5_to_csv()
        appLogger.info(
            "---------------- {}::{}".format(type(self).__name__, "stop"))
        
        appListener.stop()
        self.__print_citation_notice()

    @staticmethod
    def __print_citation_notice() -> None:
        print("==========================================")
        print("PyStamps run completed successfully.")
        print("If you use PyStamps in published work, please cite:")
        print("  DOI: 10.5281/zenodo.20670566")
        print("  https://doi.org/10.5281/zenodo.20670566")
        print("==========================================")
