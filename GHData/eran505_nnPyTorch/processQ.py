import os_util as pt

import numpy as np
import pandas as pd



def Q_upload(dir,name):
    d_Q={}
    d_map={}
    res_all_Q = pt.walk_rec(dir,[],"Q.csv")
    res_all_map = pt.walk_rec(dir, [], "map.csv")

    for itemQ in res_all_Q:
        print(itemQ)
        df = pd.read_csv(itemQ,sep=';')

        df = df[df['id'] != "id"]
        df["id"]=pd.to_numeric(df["id"])

        d_Q[str(itemQ).split('/')[-1].split('.')[0][0]]=df
    for itemMap in res_all_map:
        df = pd.read_csv(itemMap, sep=';',names=['state'],index_col=0)

        d_map[str(itemMap).split('/')[-1].split('.')[0][0]] = df.to_dict()

    res=0
    for item in d_map:
        for ky in d_map[item]:
            for enrty in d_map[item][ky]:
                value = d_map[item][ky][enrty]
                if value==name:
                    print(value)
                    res=enrty


    if res==0:
        print("cant find ->",name)
        return
    print("--------"*10)
    for Q in d_Q:
        df = d_Q[Q]

        r_res = df[df['id'] == res]
        if len(r_res)==0:
            print(Q,"\t",r_res.values)
            continue
        print(Q,"\t",r_res.values,"argMAx",np.argmax(r_res.values))

if __name__ == "__main__":
    name="0A_(0, 0, 0)_(0, 0, 0)_0|0D_(20, 20, 0)_(0, 0, 0)_4|"
    name="0A_(6, 3, 2)_(1, 1, 0)_0|0D_(20, 20, 0)_(0, 0, 0)_4|"
    name="0A_(5, 3, 2)_(2, 2, 1)_0|0D_(20, 20, 0)_(0, 0, 0)_4|"
    Q_upload("/home/eranhe/car_model/debug",name)