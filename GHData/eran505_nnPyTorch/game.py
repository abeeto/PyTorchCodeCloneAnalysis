from os.path import expanduser
import nn_pytorch as nnpy
import torch
import numpy as np
import os_util as pt
import csv, math
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import Loader, RegressionFeature
import RL.DQN as dqn
from socket import gethostname
from random import randrange
import policies
import xgboost as xgb
import joblib
import numba

# ptp_ = np.array([168.0, 168.0, 3.0, 299.0, 299.0, 3.0, 167.0, 260.0, 269.0, 3.0, 4.0, 4.0, 2.0, 168.0, 168.0, 3.0, 2.0, 2.0, 2.0, 260.0, 269.0, 3.0, 129.0, 139.0, 3.0])
# min_ = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 5.0, 5.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ptp_ = np.array([24.0, 24.0, 3.0, 29.0, 29.0, 3.0, 25.0, 25.0, 27.0, 3.0, 2.0, 2.0, 2.0, 24.0, 24.0, 3.0, 2.0, 2.0, 2.0, 25.0, 27.0, 3.0, 21.0, 22.0, 3.0])
# min_ = np.array([4.0, 11.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 33.0, 18.0, 0.0, -1.0, -1.0, -1.0, 5.0, 22.0, 0.0, 0.0, 8.0, 0.0])
# ptp_ = np.array([23.0, 31.0, 2.0, 49.0, 48.0, 2.0, 31.0, 51.0, 35.0, 1.0, 3.0, 3.0, 1.0, 23.0, 31.0, 2.0, 2.0, 2.0, 2.0, 51.0, 35.0, 1.0, 23.0, 31.0, 2.0])
def norm(f):
    f_norm = f / abs_[:12]
    f_norm = np.nan_to_num(f_norm)
    return f_norm


class schedulerAction(object):

    @staticmethod
    def get_move(diff):
        max_diff = max(diff)
        if max_diff == 0:
            return 1
        b = int(math.log2(max_diff)) + 1

        b = max(b - 3, 0)
        action_number = pow(2, b)

        return action_number


class Transformer(object):

    def __init__(self, dir_data):
        self.loader = None
        self.reg = None
        self.loader_init(dir_data)
        self.distance_man = lambda vec_1, vec_2: np.absolute(vec_1 - vec_2)

    def loader_init(self, dir_data):
        self.loader = Loader(dir_data)
        self.loader.load_p("{}/p.csv".format(dir_data))
        self.loader.load_game_setting("{}/con.csv".format(dir_data))
        dico_info_game = self.loader.get_config_id(0)
        attacker_paths = self.loader.get_path_object()
        self.reg = RegressionFeature(attacker_paths, dico_info_game)

    def get_F(self, np_arr):
        return self.reg.get_F(np_arr)


class AgentA(object):

    def __init__(self, csv_path):
        self.ctr_round = 0
        self.all_paths = []
        self.w_paths = []
        self.get_all_paths(csv_path)
        self.path_indexes = np.arange(start=0, stop=len(self.w_paths), step=1)
        self.path_number = -1
        self.step_t = -1

    def get_all_paths(self, csv_all_paths):
        self.read_file(csv_all_paths)

    def read_file(self, path_file):
        all_p = []
        with open(path_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                if len(row) < 2:
                    continue
                self.w_paths.append(float(row[0].split(":")[-1]))
                arr = [list(eval(x[1:-2])) for x in row[1:] if len(x) > 5]
                l = np.array(arr)
                all_p.append(l)
        self.all_paths = all_p

    def next_move(self, num_repeated_action):
        self.step_t += num_repeated_action
        if self.step_t >= self.all_paths[self.path_number].shape[0]:
            self.step_t = self.all_paths[self.path_number].shape[0] - 1
            # raise Exception("EndOfPathException")
        self.cur_state = self.all_paths[self.path_number][self.step_t, :]

    def reset(self):
        self.path_number = np.random.choice(self.path_indexes, 1, False)[0]  # , p=self.w_paths)[0]
        self.ctr_round += 1
        self.path_number = self.ctr_round % len(self.path_indexes)
        self.step_t = 0
        self.cur_state = self.all_paths[self.path_number][self.step_t, :]

    def __str__(self):
        return "A_" + str([tuple(x) for x in self.cur_state])


class AgentD(object):

    def __init__(self, data_path, nn_path, start_pos, debug_print, max_speed=1):
        self.max_speed = 1
        self.actions = None
        self.make_action_list()
        self.start_positions = start_pos
        self.nn = None
        self.load_nn(nn_path)
        self.cur_state = None
        self.reset()
        self.trans = Transformer(data_path)
        self.ctr = 0
        self.time_t = 0
        self.action_array = [1]
        self.debug_print = debug_print
        # self.real_Q=policies.Qpolicy(data_path)

    def load_nn(self, path_to_model):
        if pytoch:
            self.nn = nnpy.LR(len(abs_))
            self.nn = dqn.DQN(12,27)
            self.nn.load_state_dict(torch.load(path_to_model, map_location=device))
            # self.nn = self.nn.double()
            self.nn.eval()
        else:
            # self.nn = xgb.XGBClassifier({'nthread': 4})  # init model
            # self.nn.load_model(path_to_model)  # load data
            # load saved model
            self.nn = joblib.load(path_to_model)

    def reset(self):
        self.cur_state = np.array(self.start_positions, copy=True)

    def get_move(self, pos_A):
        v = np.zeros(27)
        f = self.get_F_D(pos_A)
        f = np.hstack((f.flatten(), np.zeros(3))).ravel()
        for i in range(27):
            f[-3:] = self.actions[i]
            expected_reward_y = self.nn(torch.tensor(norm(f)).double())
            v[i] = expected_reward_y
            print("{}:->{}".format(i, v[i]))
        print("np.argmax = {} ".format(np.argmax(v)))
        exit()
        return np.argmax(v)

    def get_features(self, pos_A, rep):
        f = self.get_F_D(pos_A, rep)
        f = np.hstack((f.flatten())).ravel()
        return f

    def get_action_xgb(self, pos_A, rep):
        f = self.get_features(pos_A, rep)
        f = np.reshape(f,(1,len(f)))

        res = self.nn.predict(f)
        return res[0]

    def get_action_xgb_reg(self, pos_A, rep):
        action_list=[]
        f = self.get_features(pos_A, rep)
        f = np.reshape(f,(1,len(f)))
        a = np.zeros((1,f.shape[-1]+1))
        a[0,:-1]=f[0,:]
        for i in range(27):
            a[0,-1] = i
            res = self.nn.predict(a)
            action_list.append(res[0])
        action_a  = np.argmax(np.array(action_list))
        return action_a

    def get_move_all(self, pos_A, rep):
        f = self.get_features(pos_A, rep)
        f = f.astype('f')
        # print("F:->",(f))
        f = torch.tensor(norm(f)).unsqueeze(0).float()
        expected_reward_y = self.nn(f)  # .double()

        arg_max_action = np.argmax(expected_reward_y.detach().numpy())
        # if self.ctr==0 :
        #     arg_max_action=self.action_array[0]
        self.ctr += 1
        if self.debug_print:
            print("A{} D{}".format(pos_A.flatten(), self.cur_state.flatten()))
            print("{}   \nargmax={} ".format(expected_reward_y.tolist(), arg_max_action))
            print('-' * 10)
        # exit()
        return arg_max_action

    def get_F_D(self, posA, rep):
        a = np.array([posA.flatten(), self.cur_state.flatten()]).flatten()
        a = np.expand_dims(a, axis=0)
        #a = self.trans.get_F(a)
        if with_time:
            a = np.append(a, np.array([self.time_t, rep]).flatten())
        return a

    def next_real_move(self, A_state, num_repeated_action):
        entry = np.append(A_state, self.cur_state)
        print("entry: ", entry)
        az = self.real_Q.get_actions_value(entry)
        arg_max = np.argmax(az)
        print("[action] {}".format(arg_max))
        self.apply_SEQ_action(num_repeated_action, arg_max)

    def next_move(self, pos_A, num_repeated_action):

        if pytoch:
            action_a_id = self.get_move_all(pos_A, num_repeated_action)
        else:
            pass
            #action_a_id = self.get_action_xgb(pos_A, num_repeated_action)
        #self.apply_SEQ_action(num_repeated_action, action_a_id)
        self.apply_SEQ_action(num_repeated_action, action_a_id)
        self.time_t += 1

    def apply_SEQ_action(self, num_repeated_action, action_a_id):
        for _ in range(num_repeated_action):
            self.apply_action(action_a_id)


    def apply_action(self, action_id):
        action_a = self.actions[action_id]
        speed = self.cur_state[1, :]
        new_speed = action_a + speed
        new_speed[new_speed > self.max_speed] = self.max_speed
        new_speed[new_speed < -self.max_speed] = -self.max_speed
        self.cur_state[0, :] = new_speed + self.cur_state[0, :]
        self.cur_state[1, :] = new_speed



    def make_action_list(self):
        l = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                for z in range(-1, 2, 1):
                    l.append(np.array([x, y, z]))
        self.actions = np.array(l)

    def __str__(self):
        return "D_" + str([tuple(x) for x in self.cur_state])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Game(object):

    def __init__(self, dir_data, model_path, debug_print=False):
        self.Traj = []
        self.folder = dir_data
        self.save_data = False
        self.home = expanduser("~")
        self.pow2diffs = [np.array([pow(2, 3 + x), pow(2, 3 + x), 4]) for x in range(12)]
        self.grid_size = None
        self.golas = None
        self.d_setting = None
        self.debug_print = debug_print
        self.D = None
        self.collision_arr = set()
        self.A = None
        self.debug_dict = {}
        self.get_info_game(dir_data)
        self.info = np.zeros(3)
        self.construct(dir_data, model_path, debug_print)

    def get_info_game(self, dir_data):
        obj = Loader(dir_data)
        obj.load_game_setting("{}/con.csv".format(dir_data))
        d = obj.get_config_id(0)
        self.grid_size = np.array([d['X'], d['Y'], d['Z']])
        self.golas = d['P_G']
        self.d_setting = d

        if self.save_data:
            self.Traj.append("size({}, {}, {}, )".format(self.grid_size[0],
                                                         self.grid_size[1], self.grid_size[2]))
            for item_g in self.golas:
                str_goal = "goal"
                str_goal += ("({}, {}, {}, )".format(item_g[0],
                                                     item_g[1], item_g[2]))
                str_goal += "_"
            self.Traj.append(str_goal[:-1])

    def save(self):
        if self.save_data:
            self.Traj.append(
                "A@({}, {}, {}, )".format(self.A.cur_state[0, 0], self.A.cur_state[0, 1], self.A.cur_state[0, 2]))
            self.Traj.append("D@({}, {}, {}, )".format(int(self.D.cur_state[0, 0]), int(self.D.cur_state[0, 1]),
                                                       int(self.D.cur_state[0, 2])))

    def construct(self, dir_data, model_path, debug_print):
        self.A = AgentA("{}/p.csv".format(dir_data))
        self.D = AgentD(dir_data, "{}".format(model_path),
                        np.array([self.d_setting['D_start'].squeeze(0), np.zeros(3)])
                        , debug_print)

    def main_loop(self, max_iter):
        d = {}
        for _ in range(max_iter):
            self.A.reset()
            self.D.reset()
            while True:
                if self.mini_game_end():
                    self.Traj.append("END")
                    if self.debug_print:
                        print("END")

                    break
                num_actions = schedulerAction.get_move(np.abs(self.A.cur_state[0, :] - self.D.cur_state[0, :]))
                self.inset_to_debug_dict(num_actions)

                if self.debug_print:
                    self.print_state(num_actions)

                self.D.next_move(self.A.cur_state, num_actions)
                self.A.next_move(num_actions)

                self.save()

            if self.debug_print:
                self.print_state(num_actions)

        self.print_info()
        self.flush()

    def mini_game_end(self):
        if self.if_A_at_goal(self.A.cur_state[0, :]):
            self.info[0] += 1
            return True
        if np.any(self.grid_size <= self.D.cur_state[0, :]) or np.any(self.D.cur_state[0, :] < 0):
            self.info[1] += 1
            return True
        if np.all(self.D.cur_state[0, :] == self.A.cur_state[0, :]):
            self.info[2] += 1
            self.collision_arr.add(tuple(self.D.cur_state[0, :]))
            return True

    def print_info(self):
        print("Goal:{}\tWall:{}\tCollision:{}".format(self.info[0], self.info[1], self.info[2]))
        acc = sum(self.debug_dict.values())
        for key in sorted(self.debug_dict.keys()):
            print("{} : {}".format(key, self.debug_dict[key] / acc))

    def inset_to_debug_dict(self, action_num):
        if action_num not in self.debug_dict:
            self.debug_dict[action_num] = 0
        self.debug_dict[action_num] += 1

    def if_A_at_goal(self, pos_A):
        for item_goal in self.golas:
            if np.array_equal(pos_A, item_goal):
                return True
        return False

    def print_state(self, num_actions):
        print("{}|{} [A]:{}".format(str(self.A), str(self.D), num_actions))

    def flush(self):
        if self.save_data is False:
            return
        with open("{}/t.csv".format(self.folder), 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter='\n')
            wr.writerow(self.Traj)


def plot_loss(array, dist):
    df = pd.DataFrame(array, columns=["coll"])
    df["coll"].plot(kind='line')
    plt.savefig('{}'.format(dist))  # save the figure to file
    plt.show()


with_time = False
pytoch = True
if __name__ == "__main__":
    from time import time
    import cProfile
    import re

    l = []
    # schedulerAction.get_move(np.array([8,0,0]))
    # exit()
    home = expanduser("~")

    data_folder = "{}/car_model/generalization/new/exp_400/12".format(home)
    data_folder="/home/eranhe/car_model/debug"
    if pytoch:
        abs_ = np.genfromtxt('{}/max_norm.csv'.format(data_folder), delimiter=',')
        #ptp_ = np.genfromtxt('{}/ptp.csv'.format(data_folder), delimiter=',')

    test_dir = pt.mkdir_system(data_folder, "test", True)
    if pytoch:
        nn_path = "{}/car_model/nn".format(home)
    else:
        nn_path = "{}/car_model/xgb".format(home)
    res = pt.walk_rec(nn_path, [], "pt")
    res = sorted(res)

    path_to_save = test_dir + "/coll.png"
    debug_print = False
    loop_number = 10
    s = time()
    for item_model in res:
        model_name = str(item_model).split('/')[-1].split('.')[0]
        print("[{}]".format(model_name))
        g = Game(data_folder, item_model, debug_print)
        g.main_loop(loop_number)
        l.append(g.info[2])
        print("collisions: {}".format(g.collision_arr))

    print("Time: ",time()-s)
    x = np.argmax(np.array(l))
    l = map(lambda x: x / loop_number, l)
    plot_loss(l, path_to_save)
