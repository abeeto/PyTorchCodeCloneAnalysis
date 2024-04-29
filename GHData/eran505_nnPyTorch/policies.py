import pandas as pd
from os.path import expanduser
import numpy as np
import hashlib
from os.path import expanduser
from sys import exit

# const var
from pandas._libs.lib import infer_dtype

home = expanduser("~")


# ----------------



class Qpolicy(object):

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.home = expanduser("~")
        self.Q = None
        self.rel = None
        self.map = {}
        self.matrix_f = None
        self.loader_all_data()

    def loader_all_data(self):
        names = ["S" + str(i) for i in range(1, 13)]
        names.insert(0, "id")

        df_raw = pd.read_csv("{}/Q.csv".format(self.dir_path), sep=';', error_bad_lines=False, index_col=False, )

        map_df = pd.read_csv("{}/map.csv".format(self.dir_path), sep=';', names=names)
        map_df['id'] = map_df['id'].astype('uint64')

        df_raw = pd.merge(map_df, df_raw, how='inner', on=['id'])
        ctr_df = self.get_count_state()
        df_raw = pd.merge(df_raw, ctr_df, how='left', on=['id'])
        df_raw['ctr'].fillna(0.1, inplace=True)
        self.colz = list(df_raw)
        df_raw.to_csv("{}/tmp.csv".format(self.home), index=False)
        self.matrix_f = df_raw.to_numpy()
        idz = self.matrix_f[:, 0]
        for i, id_number in enumerate(idz):
            self.map[self.hash_func(self.matrix_f[i, 1:13])] = (i, id_number)
        print("END")

    def hash_func(self, dataPoint):
        return hash(tuple(dataPoint))

    def get_count_state(self ):
        colz = ["S" + str(i) for i in range(1, 13)]
        colz.insert(0, "id")
        colz.append("ctr")

        df_last_states = pd.read_csv("{}/Last_States.csv".format(self.dir_path), names=colz, sep=';')
        df_last_states['id'] = df_last_states['id'].astype('uint64')

        return df_last_states[["id", "ctr"]]

    def get_actions_value(self,np_arr):
        h_id = self.map[self.hash_func(np_arr)]
        #print(h_id )
        #print(self.matrix_f[h_id[0],-28:-1])
        return self.matrix_f[h_id[0],-28:-1]

if __name__ == "__main__":
    p = "{}/car_model/generalization/5data".format(home)
    q = Qpolicy(p)
    pass
