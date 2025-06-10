#########################################################################
#   Copyright 2024 - 2025, KorrAI                                       #
#   ALL RIGHTS RESERVED.                                                #
#   This file is subject to the full copyright and disclaimer notice    #
#   included in a separate file in this directory.                      #
#########################################################################
#                                                                       #
#   This file contains the implementation of Stmaps.                    #
#                                                                       #
#########################################################################

from .logger import appLogger, get_app_listener
from .mtprep import mt_prep
from .steps import run_stamps_steps
from .step.utils import read_h5
from datetime import datetime
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
        run_stamps_steps(self.__PARAM_OUT)

    def __hdf5_to_csv(self):
        appLogger.info(">>>>>>>>>>>>>>>> {}::{}".format(
            type(self).__name__, "__hdf5_to_csv"))
        ps_data = read_h5(os.path.join(self.__PARAM_OUT,"ps_plot.h5"))
        header = ['lng','lat','point_id','coherence','Avg_Deformation_Velocity(mm/year)']
        header += [datetime.fromtimestamp(int( day * 86400)).strftime('%Y-%m-%d') for day in ps_data['day']]
        with open(os.path.join(self.__PARAM_OUT,'urbansar_result.csv'), 'w', 
                    newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for i, ll in enumerate(ps_data['lonlat']):
                row = [ll[0],ll[1],i,float(ps_data['coh_ps'][i]),float(ps_data['ph_disp'][i])]
                row += [ps_data['ph_mm'][i][j] for j in range(len(ps_data['ph_mm'][i]))]
                writer.writerow(row)


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
