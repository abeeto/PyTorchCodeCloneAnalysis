

import helper as hlp
import os,csv
import numpy as np
import pandas as pd
from sys import exit
import os_util as pt
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
'''
    valueType CollReward = 1000;
    valueType GoalReward = -500;
    valueType WallReward = -1000;
    valueType Step_reward = 0;
    valueType discountF=0.987;
'''


def action_dico():
    d={}
    ctr=0
    for x in range(-1,2,1):
        for y in range(-1, 2, 1):
            for z in range(-1, 2, 1):
                d[(x,y,z)]=ctr
                ctr=ctr+1
    return d

d_action = action_dico()

def string_to_reward(string_r):
    if string_r=='rG':
        return -500
    if string_r=='rW':
        return -1000
    if string_r=="rC":
        return 1000
    else:
        raise Exception("{}--not found".format(string_r))

def to_experiences(exp_arr,reward):
    experiences=[]
    index = 0
    ctr=0
    while True:
        a = np.zeros((1,29),dtype=float)
        action = d_action[tuple(exp_arr[index][1])]
        a=np.concatenate([np.array(exp_arr[index][0]).flatten(),
                   np.array([action]),
                    np.array(exp_arr[index+1][0]).flatten(),
                    np.array([0,0])],axis=0)
        experiences.append(a)
        index=index+1
        if index+1>=len(exp_arr):
            experiences.append(np.concatenate([experiences[-1][15:27],np.zeros(13),np.array([string_to_reward(reward),1]) ],axis=0))
            experiences[-2][-2]=string_to_reward(reward)*0.987
            break

    return experiences

def process_path(arr):

    experiences=[]
    k=0
    while True:
        d = arr[k].replace('\n','').replace('"','').replace(' ','').split('@')[-1].split('|')
        a = arr[k+1].replace('\n','').replace('"','').split('@')[-1].split('|')
        k = k + 2
        if len(a[-1])==0 or len(d[-1])==0:
            print("err")
            return None
        d = [eval(x) for x in d]
        a = [eval(x) for x in a]

        state = a[:]+d[:-1]
        action= d[-1]
        experiences.append((state,action))

        if k+1>=len(arr)-2:
            break

    last = np.array(arr[-2].split('_'))
    state_last = list(last[[2,3,5,6]])
    state_last = [eval(x) for x in state_last]
    experiences.append([state_last])
    reward = arr[-2][1:3]
    res = to_experiences(experiences,reward)
    return res

def load_trajectory_file(p):
    buffer=[]
    size=None
    golas = None
    with open(p, "r") as f:
        lines = f.readlines()
        size_file = len(lines)
        size=lines[0][:-1]
        golas = lines[1][:-1]
        i=2
        str_line=lines[i]
        while str_line[1:-2]!="END":
            i+=1
            str_line=lines[i]

        while i<len(lines)-1:
            i+=1
            start=i
            while lines[i][1:-2] != 'END' and i<size_file-1:
                i += 1
            end=i
            traj = process_path(lines[start:end+1])
            if traj is None:
                break
            buffer.extend(traj)
            print(i)

    return buffer

def re_process(buffer,dup=False):
    my_array = np.stack(buffer)
    if not dup:
        return np.ones(np.shape(my_array)[0]),my_array
    dt = np.dtype((np.void, my_array.dtype.itemsize * my_array.shape[1]))
    b = np.ascontiguousarray(my_array).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(my_array.dtype).reshape(-1, my_array.shape[1])
    return cnt,unq

def get_data(path_p="/home/eranhe/car_model/debug/6962_u4_L99_Traj.csv"):
    father = "/".join(str(path_p).split('/')[:-1])
    b=load_trajectory_file(path_p)
    w,data = re_process(b)
    abs_max = np.linalg.norm(data, axis=0,ord=np.inf)
    np.array(abs_max).tofile("{}/max_norm.csv".format(father), sep=',')
    data= data/abs_max
    print("data:",len(data))
    abs_max.tofile("{}/abs_max.csv".format(father), sep=',')
    df = pd.DataFrame(data=data)
    df["w"]=w
    df = df.sort_values(25)
    df.to_csv("{}/data.csv".format(father),index=False)
    return data,w

def make_mini_Traj_file(dir):
    name = os.path.join(dir,"last.csv")
    file = pt.walk_rec(dir,[],"Traj.csv")
    assert len(file)==1
    file = file[0]
    str_command="tail -n 2000 {} >> {}".format(file,name)
    os.system(str_command)
    return name

def make_test(dir):
    test_p = make_mini_Traj_file(dir)
    father = "/".join(str(test_p).split('/')[:-1])
    b=load_trajectory_file(test_p)
    w,data = re_process(b,True)
    abs_ = np.genfromtxt('{}/max_norm.csv'.format(father), delimiter=',')
    data= data/abs_
    df = pd.DataFrame(data=data)
    df["w"]=w
    df.to_csv("{}/df_test.csv".format(father),index=False)
    exit(0)


def make_critic_table(path_dir):
    df = pd.read_csv(os.path.join(path_dir,"all.csv"))

    q_indx = [x+len(list(df))-28 for x in range(27)]
    np_state_indx = [x for x in range(12)]
    idx = np_state_indx+q_indx
    df = df.iloc[:,idx]
    m = df.to_numpy()
    abs_ = np.genfromtxt('{}/max_norm.csv'.format(path_dir), delimiter=',')
    norm = np.full(39,1000)
    norm[:12]=abs_[:12]
    m=m/norm
    df = pd.DataFrame(data=m)

    #df.to_csv(path_p+"/critic.csv",index=False)
    print(list(df))
    #new_df = df[((df[0] == 1.0) & (df[2] == 0.5)) & ((df[5] == -1.0) )]
    exit(0)

if __name__ == '__main__':
    path_p = "/home/eranhe/car_model/generalization/new/4"
    file_name = "Traj.csv"
    m, w = get_data(os.path.join(path_p,file_name))
    make_test(path_p)
    make_critic_table(path_p)
