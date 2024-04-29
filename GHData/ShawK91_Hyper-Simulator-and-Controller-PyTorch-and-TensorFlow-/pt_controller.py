import numpy as np, os
import mod_hyper as mod, sys, math
from random import randint
from scipy.special import expit
from copy import deepcopy
from matplotlib import pyplot as plt
plt.switch_backend('Qt4Agg')


class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'Controller.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/valid_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self):
        self.num_input = 20
        self.num_hnodes = 15
        self.num_output = 2

        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 10000000
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.load_seed = False #Loads a seed population from the save_foldername
                                  # IF FALSE: Runs Backpropagation, saves it and uses that
            # Determine the nerual archiecture
            self.arch_type = 1  # 1 FF
                                # 2 GRUMB
            self. output_activation = None

            #Controller choices
            self.bprop_max_gens = 1000
            self.target_sensor = 11 #Turbine speed the sensor to control
            self.run_time = 300 #Controller Run time

            #Controller noise
            self.sensor_noise = 0.0
            self.sensor_failure = None  # Options: None, [11,15] permutations
            self.actuator_noise = 0.0
            self.actuator_failure = None  # Options: None, [0,1] permutations

            # Reconfigurability parameters
            self.is_random_initial_state = False  # Start state of controller
            self.num_profiles = 3

            #SSNE stuff
            self.ssne_param = SSNE_param()
            self.total_gens = 100000
            self.num_evals = 10 #Number of independent evaluations before getting a fitness score

            if self.arch_type == 1: self.arch_type = 'FF'
            elif self.arch_type == 2: self.arch_type = 'GRU-MB'
            self.save_foldername = 'R_Reconfigurable_Controller/'

class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())

class Task_Controller: #Reconfigurable Control Task
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output

        self.train_data, self.valid_data = self.data_preprocess() #Get simulator data
        self.ssne = mod.Fast_SSNE(parameters) #Initialize SSNE engine

        # Save folder for checkpoints
        self.marker = 'TF_ANN'
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #Load simulator
        self.simulator = mod.unpickle(self.save_foldername + 'Champion_Simulator')
        #mod.simulator_results(self.simulator)


        #####Create Reconfigurable controller population
        self.pop = []
        for i in range(self.parameters.population_size):
            # Choose architecture
            if self.parameters.arch_type == "GRU-MB":
                self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, self.num_output,
                                             output_activation=self.parameters.output_activation))
            elif self.parameters.arch_type == "FF":
                self.pop.append(mod.PT_FF(self.num_input, self.num_hidden, self.num_output,
                                          output_activation=self.parameters.output_activation))
            else:
                sys.exit('Error: Invalid architecture choice')


        ###Initialize Controller Population
        if self.parameters.load_seed: #Load seed population
            self.pop[0] = mod.unpickle('R_Controller/seed_controller') #Load PT_GRUMB object
        else: #Run Backprop
            self.run_bprop()
        self.pop[0].to_fast_net()  # transcribe neurosphere to its Fast_Net

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)

    #TODO BackProp
    def run_bprop(self):
        train_x = self.train_data[0:-1,0:-2]
        sensor_target = self.train_data[1:,self.parameters.target_sensor:self.parameters.target_sensor+1]
        train_x = np.concatenate((train_x, sensor_target), axis=1) #Input training data
        train_y = self.train_data[0:-1,-2:] #Target Controller Output

        #for gen in range(self.parameters.bprop_max_gens):
            #self.sess.run(self.bprop, feed_dict={self.input: train_x, self.target: train_y})
            #print self.sess.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})
        #self.saver.save(self.sess, self.save_foldername + 'bprop_controller') #Save individual

    def plot_controller(self, individual, data_setpoints=False):
        setpoints = self.get_setpoints()
        if self.parameters.is_random_initial_state:
            start_sim_input = np.copy(self.train_data[randint(0, len(self.train_data))])
        else:
            start_sim_input = np.reshape(np.copy(self.train_data[0]), (1, len(self.train_data[0])))
        start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
        for i in range(start_sim_input.shape[-1] - 2): start_controller_input[0][i] = start_sim_input[0][i]

        if data_setpoints: #Bprop test
            setpoints = self.train_data[0:, 11:12]
            mod.controller_results_bprop(individual, setpoints, start_controller_input, start_sim_input, self.simulator, self.train_data[0:,0:-2])
        else:
            mod.controller_results(individual, setpoints, start_controller_input, start_sim_input, self.simulator)

    def get_setpoints(self):

        desired_setpoints = np.reshape(np.zeros(self.parameters.run_time), (parameters.run_time, 1))

        for profile in range(parameters.num_profiles):
            multiplier = randint(1, 5)
            #print profile, multiplier
            for i in range(self.parameters.run_time/self.parameters.num_profiles):
                turbine_speed = math.sin(i * 0.2 * multiplier)
                turbine_speed *= 0.3 #Between -0.3 and 0.3
                turbine_speed += 0.5  #Between 0.2 and 0.8 centered on 0.5
                desired_setpoints[profile * self.parameters.run_time/self.parameters.num_profiles + i][0] = turbine_speed

        #plt.plot(desired_setpoints, 'r--', label='Setpoints')
        #plt.show()
        return desired_setpoints

    def compute_fitness(self, individual, setpoints, start_controller_input, start_sim_input): #Controller fitness
        weakness = 0.0
        individual.fast_net.reset()

        control_input = np.copy(start_controller_input) #Input to the controller
        sim_input = np.copy(start_sim_input) #Input to the simulator

        for example in range(len(setpoints) - 1):  # For all training examples
            # Fill in the setpoint to control input
            control_input[0][-1] = setpoints[example][0]

            # # Add noise to the state input to the controller
            # if self.parameters.sensor_noise != 0:  # Add sensor noise
            #     for i in range(19):
            #         std = self.parameters.sensor_noise * abs(noise_input[0][i]) / 100.0
            #         if std != 0:
            #             noise_input[0][i] += np.random.normal(0, std / 2.0)
            #
            # if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
            #     for i in self.parameters.sensor_failure:
            #         noise_input[0][i] = 0
            #

            #RUN THE CONTROLLER TO GET CONTROL OUTPUT
            control_out = individual.fast_net.predict(control_input)
            #
            # # Add actuator noise (controls)
            # if self.parameters.actuator_noise != 0:
            #     for i in range(len(control_out[0])):
            #         std = self.parameters.actuator_noise * abs(control_out[0][i]) / 100.0
            #         if std != 0:
            #             control_out[0][i] += np.random.normal(0, std / 2.0)
            #
            # if self.parameters.actuator_failure != None:  # Failed actuator outputs 0 regardless
            #     for i in self.parameters.actuator_failure:
            #         control_out[0][i] = 0


            #Fill in the controls
            sim_input[0][19] = control_out[0][0]
            sim_input[0][20] = control_out[0][1]

            # Use the simulator to get the next state
            simulator_out = self.simulator.predict(sim_input)

            # Calculate error (weakness)
            weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

            # Fill in the simulator inputs and control inputs
            for i in range(simulator_out.shape[-1]):
                sim_input[0][i] = simulator_out[0][i]
                control_input[0][i] = simulator_out[0][i]

        return -weakness

    def evolve(self, gen):

        #Fitness evaluation list for the generation
        fitness_evals = [0.0] * (self.parameters.population_size)

        for eval in range(parameters.num_evals): #Take multiple samples
            #Figure initial position and setpoints for the generation
            setpoints = self.get_setpoints()
            if self.parameters.is_random_initial_state: start_sim_input = np.copy(self.train_data[randint(0,len(self.train_data))])
            else: start_sim_input = np.reshape(np.copy(self.train_data[0]), (1, len(self.train_data[0])))
            start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
            for i in range(start_sim_input.shape[-1]-2): start_controller_input[0][i] = start_sim_input[0][i]


            #Test all individuals and assign fitness
            for index, individual in enumerate(self.pop): #Test all genomes/individuals
                fitness = self.compute_fitness(individual, setpoints, start_controller_input, start_sim_input)
                fitness_evals[index] += fitness/(1.0*parameters.num_evals)
        gen_best_fitness = max(fitness_evals)

        #Champion Individual
        champion_index = fitness_evals.index(max(fitness_evals))
        valid_score = 0.0
        for eval in range(parameters.num_evals):  # Take multiple samples
            setpoints = self.get_setpoints()
            if self.parameters.is_random_initial_state: start_sim_input = np.copy(self.valid_data[randint(0,len(self.valid_data))])
            else: start_sim_input = np.reshape(np.copy(self.valid_data[0]), (1, len(self.valid_data[0])))
            start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
            for i in range(start_sim_input.shape[-1]-2): start_controller_input[0][i] = start_sim_input[0][i]
            valid_score += self.compute_fitness(self.pop[champion_index], setpoints, start_controller_input, start_sim_input)/(1.0*parameters.num_evals)


        #Save population and Champion
        if gen % 50 == 0:
            for index, individual in enumerate(self.pop): #Save population
                self.save(individual, self.save_foldername + 'Controller_' + str(index))
            self.save(self.pop[champion_index], self.save_foldername + 'Champion_Controller') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals)

        return gen_best_fitness, valid_score

    def data_preprocess(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
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

        #Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data

    def test_restore(self, individual):
        train_x = self.train_data[0:-1]
        train_y = self.train_data[1:,0:-2]
        print individual.sess.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})
        self.load(individual, 'Controller_' + str(98))
        print individual.sess.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Reconfigurable Controller Training ', parameters.arch_type

    control_task = Task_Controller(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, valid_score = control_task.evolve(gen)
        print 'Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














