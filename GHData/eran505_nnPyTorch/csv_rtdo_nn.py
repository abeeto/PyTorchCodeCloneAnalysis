import os
import pandas as pd
import numpy as np
import os_util as pt
import matplotlib.pyplot as plt

def get_info(p_path="/home/eranhe/car_model/old/servers/30_03_Z0_EXP/map_exp2"):
    name_folder=str(p_path).split('/')[-1]
    if os.path.isdir(p_path) is False:
        return
    d_l=[]
    res = pt.walk_rec(p_path,[],"",False,lv=-1)
    con_res = pt.walk_rec(p_path, [], "con")
    for item in res:
        d_l.append(get_info_dir(item))

    df = pd.DataFrame(d_l)
    df_con = pd.read_csv(con_res[0])
    for col in list(df_con):
        df[str(col)]=df_con[col].iloc[0]
    df.to_csv("{}/{}_sum.csv".format(p_path,name_folder))
    return df

def get_info_dir(p_path="/home/eranhe/car_model/old/servers/30_03_Z0_EXP/map_exp2/nn",k=10):
    name = str(p_path).split('/')[-1]
    csv_filesI = pt.walk_rec(p_path,[],"_P.csv")
    csv_files = [x for x in csv_filesI if str(x).split("/")[-1].__contains__("lock") is False]
    print(csv_files)
    assert len(csv_files) == 1
    df = pd.read_csv(csv_files[0])
    d={}
    list_col = list(df)
    list_col.remove("ctr_round")
    d['mode']=name
    for item_col in list_col:
        item_col_name = str(item_col).split("_")[-1]
        if item_col =="ctr_coll":
            d[item_col_name+"_MAX"]=df[item_col].max()
        d[item_col_name+ "_AVG"] = float(np.mean(df[[item_col]].tail(k)))
    print(d)
    return d




def warpper(p="/home/eranhe/car_model/old/servers/30_03_Z0_EXP"):
    resz = pt.walk_rec(p, [], "", False, lv=-1)
    df_l=[]
    for item in resz:
        df_l.append(get_info(item))
    df = pd.concat(df_l)
    df.to_csv("{}/all.csv".format(p))

def get_all_coll(p):
    res = pt.walk_rec(p,[],"P.csv")
    d={}
    for item in res:

        if str(item).split("/")[-1].__contains__("lock"):
            continue
        print(item)
        name="_".join(str(item).split("/")[-1])
        if name.__contains__("map"):
            continue
            name+="_R7"
        if name.__contains__("smart"):
            name += "_R10"
        if name.__contains__("roni"):
            name += "_R5"
        name = "_".join(name.split("_")[1:])
        print(name)
        df = pd.read_csv(item)
        d[name]=df['ctr_coll']/500
        print(list(df))

    # Create dataframe from dic and make keys, index in dataframe
    dfObj = pd.DataFrame.from_dict(d, orient='index')
    print(dfObj)
    dfObj = dfObj.transpose()

    dfObj.plot( kind='line')
    plt.show()
    dfObj.to_csv("{}/df.csv".format(p))

if __name__ == "__main__":
   # warpper("/home/eranhe/car_model/old/servers/30_03_Z0_EXP")
    #pp_path1 = "/home/eranhe/car_model/old/servers/04_04_Z1_EXP"
    pp_path1="/home/ERANHER/car_model/results/26_04/con1"
    #warpper(pp_path1)
    get_all_coll(pp_path1)