import numpy as np, os
import mod_hyper as mod, sys, math
from random import randint
from scipy.special import expit
import tensorflow as tf
from copy import deepcopy
from matplotlib import pyplot as plt

#Changeable Macros
controller_id = 1 #1 TF FF
            #2 PT FF
            #3 PT GRU-MB


#FIXED
num_input = 20
is_random_initial_state = False
num_profiles = 3
run_time = 300

plt.switch_backend('Qt4Agg')
#Load classes required to load
class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())
class Fast_Controller():  # TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())

def get_setpoints():

        desired_setpoints = np.reshape(np.zeros(run_time), (run_time, 1))

        for profile in range(num_profiles):
            multiplier = randint(1, 5)
            # print profile, multiplier
            for i in range(run_time / num_profiles):
                turbine_speed = math.sin(i * 0.2 * multiplier)
                turbine_speed *= 0.3  # Between -0.3 and 0.3
                turbine_speed += 0.5  # Between 0.2 and 0.8 centered on 0.5
                desired_setpoints[profile * run_time / num_profiles + i][
                    0] = turbine_speed

        # plt.plot(desired_setpoints, 'r--', label='Setpoints')
        # plt.show()
        return desired_setpoints

def data_preprocess(filename='ColdAir.csv', downsample_rate=25, split=1000):
        # Import training data and clear away the two top lines
        data = np.loadtxt(filename, delimiter=',', skiprows=2)

        # Splice data (downsample)
        ignore = np.copy(data)
        data = data[0::downsample_rate]
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i != data.shape[0] - 1):
                    data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                                 j].sum() / downsample_rate
                else:
                    residue = ignore.shape[0] - i * downsample_rate
                    data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

        # Normalize between 0-0.99
        normalizer = np.zeros(data.shape[1])
        min = np.zeros(len(data[0]))
        max = np.zeros(len(data[0]))
        for i in range(len(data[0])):
            min[i] = np.amin(data[:, i])
            max[i] = np.amax(data[:, i])
            normalizer[i] = max[i] - min[i] + 0.00001
            data[:, i] = (data[:, i] - min[i]) / normalizer[i]

        # Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data

def plot_controller(individual, controller_id, save_foldername, data_setpoints=False, ):

    #Load simulator
    simulator = mod.unpickle(save_foldername + 'Champion_Simulator')

    train_data, valid_data = data_preprocess()  # Get simulator data
    setpoints = get_setpoints()
    if is_random_initial_state:
        start_sim_input = np.copy(train_data[randint(0, len(train_data))])
    else:
        start_sim_input = np.reshape(np.copy(train_data[0]), (1, len(train_data[0])))
    start_controller_input = np.reshape(np.zeros(num_input), (1, num_input))
    for i in range(start_sim_input.shape[-1] - 2): start_controller_input[0][i] = start_sim_input[0][i]

    if controller_id == 2 or controller_id == 3: #PT models
        mod.pt_controller_results(individual, setpoints, start_controller_input, start_sim_input, simulator)
    else: #TF models earlier
        if data_setpoints:  # Bprop test
            setpoints = train_data[0:, 11:12]
            mod.controller_results_bprop(individual, setpoints, start_controller_input, start_sim_input, simulator,
                                         train_data[0:, 0:-2])
        else:
            mod.controller_results(individual, setpoints, start_controller_input, start_sim_input, simulator)


if __name__ == "__main__":
    print 'Running Controller Testing on Simple Reconfigurability Task with',
    if controller_id == 1: print 'TF_FF'
    elif controller_id == 2: print 'PT_FF'
    elif controller_id == 3: print 'PT_GRU-MB'

    if controller_id == 2 or controller_id == 3: save_foldername = 'R_Reconfigurable_Controller/'
    elif controller_id == 1: save_foldername = 'R_Controller/'
    else: sys.exit('Incorrect Controller ID selection')

    test_controller = mod.unpickle(save_foldername + 'Champion_Controller')
    plot_controller(test_controller, controller_id, save_foldername)














