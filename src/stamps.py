#########################################################################
#   Copyright 2022 - 2025, KorrAI                                       #
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
import os
import h5py
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool

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

    def __hdf5_reader(self,path:str):
        with h5py.File(path, 'r') as h5:
            if 'pm2.mat' in path:
                return h5['coh_ps'][0]
            elif "ps_plot_v-dao.mat" in path:
                return h5['ph_disp'][0]
            elif "ps_plot_ts_v-dao.mat" in path:
                return (h5["lonlat"][0],h5["lonlat"][1], 
                        h5["day"][0],np.array(h5["ph_mm"]))

    def __stamps_postprocess(self):
        appLogger.info(">>>>>>>>>>>>>>>> {}::{}".format(
            type(self).__name__, "__stamps_postprocess"))
        stampsoutpath = self.__PARAM_OUT
        ts_matfile = os.path.join(stampsoutpath, "ps_plot_ts_v-dao.mat")
        vm_matfile = os.path.join(stampsoutpath, "ps_plot_v-dao.mat")
        pm_matfile = os.path.join(stampsoutpath, "pm2.mat")
        res_csv = os.path.join(stampsoutpath, "urbansar_result.csv")

        longitude,latitude,days,phase = self.__hdf5_reader(ts_matfile)

        # start populating  df
        df = {
            'export_res_ 1':  np.insert(longitude, 0, 0),
            'export_res_ 2':  np.insert(latitude,  0, 1),
            'export_res_ 3':  np.insert(self.__hdf5_reader(vm_matfile), 0, 'NaN'),
        }

        # number of deformation Points
        # An index of rows Populated
        curIdx = 4
        for idx in range(len(days)):
            df['export_res_ {}'.format(curIdx)] = np.insert(phase[idx], 0, days[idx])
            curIdx += 1

        df = pd.DataFrame(df)
        new_headers    = list(df.iloc[0])
        new_headers[2] = 'Avg_Deformation_Velocity(mm/year)'
        df.columns = new_headers
        df = df.iloc[1:].reset_index(drop=True)

        # get name of third field
        fields = df.columns
        velField = fields[2]

        # compile dictonatry
        usrDict = {
            fields[0]: 'lng',
            fields[1]: 'lat',
            velField: 'Avg_Deformation_Velocity(mm/year)'
        }

        def dateFormatter(day):
            start = date(1, 1, 1)
            delta = timedelta(day)
            offset = start + delta - \
                relativedelta(years=1,) - timedelta(days=2,)
            return offset

        for encDate in fields[3:]:
            val = str(dateFormatter(int(float(encDate))))
            usrDict[encDate] = val

        coherence = pd.DataFrame(self.__hdf5_reader(pm_matfile))
        df.rename(columns=usrDict, inplace=True)
        idtArr = np.arange(0, len(df))
        df.insert(2, 'point_id', idtArr)
        df.insert(3, 'coherence', coherence)

        # Write df
        df.to_csv(res_csv, index=False)

    def start(self) -> None:
        appListener = get_app_listener(path = self.__TEMP_DIR)
        appListener.start()

        appLogger.info(
            "++++++++++++++++ {}::{}".format(type(self).__name__, "start"))
        self.__stamps_processor()
        #self.__stamps_postprocess()
        appLogger.info(
            "---------------- {}::{}".format(type(self).__name__, "stop"))
        
        appListener.stop()
