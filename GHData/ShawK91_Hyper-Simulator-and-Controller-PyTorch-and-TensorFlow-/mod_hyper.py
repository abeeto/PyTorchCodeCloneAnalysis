from random import randint
import math
from scipy.special import expit
import os, cPickle
import tensorflow as tf
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, torch
from copy import deepcopy
import torch.nn.functional as F


class Fast_GRUMB:
    def __init__(self, num_input, num_hnodes, num_output, output_activation, mean = 0, std = 1):
        self.arch_type = 'Fast_GRUMB'
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes
        if output_activation == 'sigmoid': self.output_activation = self.fast_sigmoid
        elif output_activation == 'tanh': self.output_activation = np.tanh
        else: self.output_activation = None

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, (num_input, num_hnodes)))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_hnodes)))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_input, num_hnodes)))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))

        #Forget gate
        self.w_readgate = np.mat(np.random.normal(mean, std, (num_input, num_hnodes)))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_hnodes)))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std, (num_input, num_hnodes)))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std, (num_hnodes, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))

        #Biases
        self.w_input_gate_bias = np.mat(np.random.normal(mean, std, (1, num_hnodes)))
        self.w_block_input_bias = np.mat(np.random.normal(mean, std, (1, num_hnodes)))
        self.w_readgate_bias = np.mat(np.random.normal(mean, std, (1, num_hnodes)))
        self.w_writegate_bias = np.mat(np.random.normal(mean, std, (1, num_hnodes)))

        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((1,self.num_output)))
        self.memory = np.mat(np.zeros((1,self.num_hnodes)))

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_hid_out': self.w_hid_out,
                            'w_input_gate_bias': self.w_input_gate_bias,
                           'w_block_input_bias': self.w_block_input_bias,
                            'w_readgate_bias': self.w_readgate_bias,
                           'w_writegate_bias': self.w_writegate_bias}


    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        layer_input = expit(layer_input)
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def forward(self, input): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input gate
        ig_1 = self.linear_combination(input, self.w_inpgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_inpgate)
        ig_3 = self.linear_combination(self.memory, self.w_mem_inpgate)
        input_gate_out = ig_1 + ig_2 + ig_3 + self.w_input_gate_bias
        input_gate_out = self.fast_sigmoid(input_gate_out)

        #Input processing
        ig_1 = self.linear_combination(input, self.w_inp)
        ig_2 = self.linear_combination(self.output, self.w_rec_inp)
        block_input_out = ig_1 + ig_2 + self.w_block_input_bias
        block_input_out = self.fast_sigmoid(block_input_out)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        ig_1 = self.linear_combination(input, self.w_readgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_readgate)
        ig_3 = self.linear_combination(self.memory, self.w_mem_readgate)
        read_gate_out = ig_1 + ig_2 + ig_3 + self.w_readgate_bias
        read_gate_out = self.fast_sigmoid(read_gate_out)

        #Memory Output
        memory_output = np.multiply(read_gate_out, self.memory)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = memory_output + input_out

        #Write gate (memory cell)
        ig_1 = self.linear_combination(input, self.w_writegate)
        ig_2 = self.linear_combination(self.output, self.w_rec_writegate)
        ig_3 = self.linear_combination(self.memory, self.w_mem_writegate)
        write_gate_out = ig_1 + ig_2 + ig_3 + self.w_writegate_bias
        write_gate_out = self.fast_sigmoid(write_gate_out)

        #Write to memory Cell - Update memory
        self.memory += np.multiply(write_gate_out, np.tanh(hidden_act))

        #Compute final output
        self.output = self.linear_combination(hidden_act, self.w_hid_out)
        if self.output_activation != None: self.output = self.output_activation(self.output)

        return np.array(self.output).tolist()

    def reset(self):
        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((1,self.num_output)))
        self.memory = np.mat(np.zeros((1,self.num_hnodes)))

    def predict(self, input):
        return self.forward(input)

class Fast_FF:
    def __init__(self, num_input, num_hnodes, num_output, output_activation, mean = 0, std = 1):
        self.arch_type = 'FF'
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes
        if output_activation == 'sigmoid': self.output_activation = self.fast_sigmoid
        elif output_activation == 'tanh': self.output_activation = np.tanh
        else: self.output_activation = None


        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_input, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))

        #Biases
        self.w_inp_bias = np.mat(np.random.normal(mean, std, (1, num_hnodes)))
        self.w_hid_out_bias = np.mat(np.random.normal(mean, std, (1, num_output)))

        self.param_dict = {'w_inp': self.w_inp,
                           'w_hid_out': self.w_hid_out}


    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        layer_input = expit(layer_input)
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def forward(self, input): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input processing
        hidden_act = self.fast_sigmoid(self.linear_combination(input, self.w_inp) + self.w_inp_bias)

        #Compute final output
        self.output = self.linear_combination(hidden_act, self.w_hid_out) + self.w_hid_out_bias
        if self.output_activation != None: self.output = self.output_activation(self.output)

        return np.array(self.output).tolist()

    def reset(self):
        return

    def predict(self, input):
        return self.forward(input)

class Fast_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters;
        self.ssne_param = self.parameters.ssne_param;
        self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input;
        self.num_hidden = self.ssne_param.num_hnodes;
        self.num_output = self.ssne_param.num_output

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def crossover_inplace(self, gene1, gene2):
        keys = list(gene1.param_dict.keys())

        # References to the variable tensors
        W1 = gene1.param_dict
        W2 = gene2.param_dict
        num_variables = len(W1)
        if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

        # Crossover opertation [Indexed by column, not rows]
        num_cross_overs = randint(1, num_variables * 2)  # Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, num_variables - 1)  # Choose which tensor to perturb
            receiver_choice = random.random()  # Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, W1[keys[tensor_choice]].shape[-1] - 1)  #
                W1[keys[tensor_choice]][:, ind_cr] = W2[keys[tensor_choice]][:, ind_cr]
                #W1[keys[tensor_choice]][ind_cr, :] = W2[keys[tensor_choice]][ind_cr, :]
            else:
                ind_cr = randint(0, W2[keys[tensor_choice]].shape[-1] - 1)  #
                W2[keys[tensor_choice]][:, ind_cr] = W1[keys[tensor_choice]][:, ind_cr]
                #W2[keys[tensor_choice]][ind_cr, :] = W1[keys[tensor_choice]][ind_cr, :]

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05


        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        ssne_probabilities = np.random.uniform(0,1,num_structures)*2


        for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
            if random.random()<ssne_prob:

                num_mutations = randint(1, math.ceil(num_mutation_frac * W[key].size))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = randint(0, randint(0, W[key].shape[0] - 1))
                    ind_dim2 = randint(0, randint(0, W[key].shape[-1] - 1))
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                      W[key][
                                                                                          ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                                                                                          ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                        W[key][ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master
        keys = master.param_dict.keys()
        for key in keys:
            replacee.param_dict[key][:] = master.param_dict[key]


    def reset_genome(self, gene):
        keys = gene.param_dict
        for key in keys:
            dim = gene.param_dict[key].shape
            gene.param_dict[key][:] = np.mat(np.random.uniform(-1, 1, (dim[0], dim[1])))

    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals);
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.ssne_param.extinction_prob: #An extinction event
            print
            print "######################Extinction Event Triggered#######################"
            print
            for i in offsprings:
                if random.random() < self.ssne_param.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(pop[i].fast_net)
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i].fast_net, replacee=pop[replacee].fast_net)

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=pop[off_i].fast_net, replacee=pop[i].fast_net)
            self.copy_individual(master=pop[off_j].fast_net, replacee=pop[j].fast_net)
            self.crossover_inplace(pop[i].fast_net, pop[j].fast_net)

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i].fast_net, pop[j].fast_net)

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i].fast_net)

    def save_model(self, model, filename):
        torch.save(model, filename)

class PT_GRUMB(nn.Module):
    def __init__(self, input_size, memory_size, output_size, output_activation):
        super(PT_GRUMB, self).__init__()

        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None
        self.fast_net = Fast_GRUMB(input_size, memory_size, output_size, output_activation)

        #Input gate
        self.w_inpgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(output_size, memory_size), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(input_size, memory_size), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(output_size, memory_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Biases
        self.w_input_gate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
        self.w_block_input_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
        self.w_readgate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)
        self.w_writegate_bias = Parameter(torch.zeros(1, memory_size), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=1)

    def reset(self):
        # Adaptive components
        self.mem = Variable(torch.zeros(1, self.memory_size), requires_grad=1)
        self.out = Variable(torch.zeros(1, self.output_size), requires_grad=1)

    # Some bias
    def graph_compute(self, input, rec_output, mem):
        # Compute hidden activation
        block_inp = F.sigmoid(input.mm(self.w_inp) + rec_output.mm(self.w_rec_inp) + self.w_block_input_bias)
        inp_gate = F.sigmoid(input.mm(self.w_inpgate) + mem.mm(self.w_mem_inpgate) + rec_output.mm(
            self.w_rec_inpgate) + self.w_input_gate_bias)
        inp_out = block_inp * inp_gate

        mem_out = F.sigmoid(input.mm(self.w_readgate) + rec_output.mm(self.w_rec_readgate) + mem.mm(self.w_mem_readgate) + self.w_readgate_bias) * mem

        hidden_act = mem_out + inp_out

        write_gate_out = F.sigmoid(input.mm(self.w_writegate) + mem.mm(self.w_mem_writegate) + rec_output.mm(self.w_rec_writegate) + self.w_writegate_bias)
        mem = mem + write_gate_out * F.tanh(hidden_act)

        output = hidden_act.mm(self.w_hid_out)
        if self.output_activation != None: output = self.output_activation(output)

        return output, mem


    def forward(self, input):
        x = Variable(torch.Tensor(input), requires_grad=True); x = x.unsqueeze(0)
        self.out, self.mem = self.graph_compute(x, self.out, self.mem)
        return self.out

    def predict(self, input):
        out = self.forward(input)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

    def to_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].numpy()

    def from_fast_net(self):
        keys = self.state_dict().keys() #Get all keys
        params = self.state_dict() #Self params
        fast_net_params = self.fast_net.param_dict #Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])

class PT_FF(nn.Module):
    def __init__(self, input_size, memory_size, output_size, output_activation):
        super(PT_FF, self).__init__()
        self.is_static = False #Distinguish between this and static policy
        self.fast_net = Fast_FF(input_size, memory_size, output_size, output_activation)
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        self.input_size = input_size; self.memory_size = memory_size; self.output_size = output_size

        #Block Input
        self.w_inp = Parameter(torch.ones(input_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

        # Turn grad off for evolutionary algorithm
        #self.turn_grad_off()


    def reset(self):
        #Adaptive components
        self.agent_sensor = 0.0; self.last_reward = 0.0

    def graph_compute(self, input):
        return F.sigmoid(input.mm(self.w_inp)).mm(self.w_hid_out)

    def forward(self, input):
        return self.graph_compute(input)

    def predict(self, input, is_static=False):
        out = self.forward(input, is_static)
        output = out.data.numpy()
        return output

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

    def to_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].numpy()

    def from_fast_net(self):
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])







class TF_simulator_nodes():
    def __init__(self, num_input, num_hidden, num_output):
        #Placeholder for input and target
        self.input = tf.placeholder("float", [None, num_input])
        self.target = tf.placeholder("float", [None, num_output])

        # Call trainable variables (neural weights)
        self.w_h1 = tf.Variable(tf.random_uniform([num_input, num_hidden], -1, 1))
        self.w_b1 = tf.Variable(tf.random_uniform([num_hidden], -1, 1))
        self.w_h2 = tf.Variable(tf.random_uniform([num_hidden, num_output], -1, 1))
        self.w_b2 = tf.Variable(tf.random_uniform([num_output], -1, 1))

        # Feedforward operation
        self.h_1 = tf.nn.sigmoid(tf.matmul(self.input, self.w_h1) + self.w_b1)
        self.net_out = tf.matmul(self.h_1, self.w_h2) + self.w_b2
        # self.net_out = tf.nn.sigmoid(self.net_out)

        # Define loss function and backpropagation (optimizer)
        self.cost = tf.losses.absolute_difference(self.target, self.net_out)
        self.bprop = tf.train.AdamOptimizer(0.1).minimize(self.cost)
        #bprop = tf.train.GradientDescentOptimizer(0.09).minimize(self.cost)

class TF_Simulator(): #TF Simulator individual
    def __init__(self):

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

class TF_FAST_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param; self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output


    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):

        #New weights initialize as copy of previous weights
        new_W1 = gene1.W
        new_W2 = gene2.W


        #Crossover opertation (NOTE THE INDICES CROSSOVER BY COLUMN NOT ROWS)
        num_cross_overs = randint(1, len(new_W1) * 2) #Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, len(new_W1)-1) #Choose which tensor to perturb
            receiver_choice = random.random() #Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, new_W1[tensor_choice].shape[-1]-1)  #
                new_W1[tensor_choice][:, ind_cr] = new_W2[tensor_choice][:, ind_cr]
            else:
                ind_cr = randint(0, new_W2[tensor_choice].shape[-1]-1)  #
                new_W2[tensor_choice][:, ind_cr] = new_W1[tensor_choice][:, ind_cr]




    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05



        # New weights initialize as copy of previous weights
        new_W = gene.W
        num_variables = len(new_W)


        num_tensor_mutation = randint(0, num_variables-1) #Number of mutation operation level of tensors
        for _ in range(num_tensor_mutation):
            tensor_choice = randint(0, num_variables-1)#Choose which tensor to perturb
            num_mutations = randint(1, math.ceil(num_mutation_frac * new_W[tensor_choice].size)) #Number of mutation instances
            for _ in range(num_mutations):
                ind_dim1 = randint(0, randint(0, new_W[tensor_choice].shape[0]-1))
                ind_dim2 = randint(0, randint(0, new_W[tensor_choice].shape[-1]-1))
                if random.random() < super_mut_prob:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, super_mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])
                else:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])

                # Regularization hard limit
                    new_W[tensor_choice][ind_dim1][ind_dim2] = self.regularize_weight(new_W[tensor_choice][ind_dim1][ind_dim2])

    def copy_individual(self, master, replacee): #Replace the replacee individual with master
       replacee.W = deepcopy(master.W)



    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])
            #pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            #pop[i] = copy.deepcopy(pop[off_i])
            #pop[j] = copy.deepcopy(pop[off_j])
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i])

    def save_pop(self, filename='Pop'):
        filename = str(self.current_gen) + '_' + filename
        pickle_object(self.pop, filename)

class TF_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param; self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.ssne_param.num_input; self.num_hidden = self.ssne_param.num_hnodes; self.num_output = self.ssne_param.num_output


    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):
        #References to the variable tensors
        variable_tensors= tf.trainable_variables()
        num_variables = len(variable_tensors)

        #New weights initialize as copy of previous weights
        new_W1 = gene1.sess.run(tf.trainable_variables())
        new_W2 = gene2.sess.run(tf.trainable_variables())


        #Crossover opertation (NOTE THE INDICES CROSSOVER BY COLUMN NOT ROWS)
        num_cross_overs = randint(1, num_variables * 2) #Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = randint(0, num_variables-1) #Choose which tensor to perturb
            receiver_choice = random.random() #Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = randint(0, new_W1[tensor_choice].shape[-1]-1)  #
                new_W1[tensor_choice][:, ind_cr] = new_W2[tensor_choice][:, ind_cr]
            else:
                ind_cr = randint(0, new_W2[tensor_choice].shape[-1]-1)  #
                new_W2[tensor_choice][:, ind_cr] = new_W1[tensor_choice][:, ind_cr]

        #Assign the new weights to individuals
        for i in range(num_variables):
            #Create operations for assigning new weights
            op_1 = variable_tensors[i].assign(new_W1[i])
            op_2 = variable_tensors[i].assign(new_W2[i])

            #Run them in session
            gene1.sess.run(op_1)
            gene2.sess.run(op_2)

    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05

        # References to the variable tensors
        variable_tensors = tf.trainable_variables()
        num_variables = len(variable_tensors)

        # New weights initialize as copy of previous weights
        new_W = gene.sess.run(tf.trainable_variables())


        num_tensor_mutation = randint(0, num_variables-1) #Number of mutation operation level of tensors
        for _ in range(num_tensor_mutation):
            tensor_choice = randint(0, num_variables-1)#Choose which tensor to perturb
            num_mutations = randint(1, math.ceil(num_mutation_frac * new_W[tensor_choice].size)) #Number of mutation instances
            for _ in range(num_mutations):
                ind_dim1 = randint(0, randint(0, new_W[tensor_choice].shape[0]-1))
                ind_dim2 = randint(0, randint(0, new_W[tensor_choice].shape[-1]-1))
                if random.random() < super_mut_prob:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, super_mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])
                else:
                    new_W[tensor_choice][ind_dim1][ind_dim2] += random.gauss(0, mut_strength * new_W[tensor_choice][ind_dim1][ind_dim2])

                # Regularization hard limit
                    new_W[tensor_choice][ind_dim1][ind_dim2] = self.regularize_weight(new_W[tensor_choice][ind_dim1][ind_dim2])

        # Assign the new weights to individuals
        for i in range(num_variables):
            # Create operations for assigning new weights
            op_1 = variable_tensors[i].assign(new_W[i])

            # Run them in session
            gene.sess.run(op_1)

    def copy_individual(self, master, replacee): #Replace the replacee individual with master
        #References to the variable tensors
        variable_tensors= tf.trainable_variables()
        num_variables = len(variable_tensors)

        #New weights initialize as copy of previous weights
        master_W = master.sess.run(tf.trainable_variables())

        #Assign the new weights to individuals
        for i in range(num_variables):
            #Create operations for assigning new weights
            op = variable_tensors[i].assign(master_W[i])

            #Run them in session
            replacee.sess.run(op)



    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)
        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])
            #pop[replacee] = copy.deepcopy(pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            #pop[i] = copy.deepcopy(pop[off_i])
            #pop[j] = copy.deepcopy(pop[off_j])
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(pop[i])

    def save_pop(self, filename='Pop'):
        filename = str(self.current_gen) + '_' + filename
        pickle_object(self.pop, filename)

class TF_SSNE_BCK:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param; self.arch_type = self.parameters.arch_type
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.fitness_evals = [[] for x in xrange(parameters.population_size)]  # Fitness eval list

        # Simulator save
        self.marker = 'TF_ANN'
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)


        num_input = self.ssne_param.num_input
        num_hidden = self.ssne_param.num_hnodes
        num_output = self.ssne_param.num_output
        # Placeholder for input and target
        self.input = tf.placeholder("float", [None, num_input])
        self.target = tf.placeholder("float", [None, num_output])

        # Call trainable variables (neural weights)
        self.w_h1 = tf.Variable(tf.random_uniform([num_input, num_hidden], -1, 1))
        self.w_b1 = tf.Variable(tf.random_uniform([num_hidden], -1, 1))
        self.w_h2 = tf.Variable(tf.random_uniform([num_hidden, num_output], -1, 1))
        self.w_b2 = tf.Variable(tf.random_uniform([num_output], -1, 1))

        # Feedforward operation
        self.h_1 = tf.nn.sigmoid(tf.matmul(self.input, self.w_h1) + self.w_b1)
        self.net_out = tf.matmul(self.h_1, self.w_h2) + self.w_b2
        # self.net_out = tf.nn.sigmoid(self.net_out)

        # Define loss function and backpropagation (optimizer)
        self.cost = tf.losses.absolute_difference(self.target, self.net_out)
        self.bprop = tf.train.AdamOptimizer(0.1).minimize(self.cost)
        self.saver = tf.train.Saver()

        # Create population
        self.pop = []
        for i in range(self.population_size):
            self.pop.append(tf.Session())
            self.pop[i].run(tf.global_variables_initializer())

    def run_bprop_test(self, num_gen, train_x, train_y, agent):
        for gen in range(num_gen):
            agent.run(self.bprop, feed_dict={self.input: train_x, self.target: train_y})
            #print agent.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})



    def save(self, individual, filename = 'tf_ann.ckpt'):
        return self.saver.save(individual, self.save_foldername + filename)

    def load(self, filename = './tf_ann.ckpt'):
        self.saver.restore(self.sess, self.save_foldername + filename)


    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def crossover_inplace(self, gene1, gene2):

        if self.ssne_param.type_id == 'memoried':  # Memory net
            # INPUT GATES
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_inpgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                    gene1.w_inpgate[ind_cr, :] = gene2.w_inpgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                    gene2.w_inpgate[ind_cr, :] = gene1.w_inpgate[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_rec_inpgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                    gene1.w_rec_inpgate[ind_cr, :] = gene2.w_rec_inpgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                    gene2.w_rec_inpgate[ind_cr, :] = gene1.w_rec_inpgate[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_mem_inpgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                    gene1.w_mem_inpgate[ind_cr, :] = gene2.w_mem_inpgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                    gene2.w_mem_inpgate[ind_cr, :] = gene1.w_mem_inpgate[ind_cr, :]
                else:
                    continue

            # BLOCK INPUTS
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_inp))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_inp) - 1)
                    gene1.w_inp[ind_cr, :] = gene2.w_inp[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_inp) - 1)
                    gene2.w_inp[ind_cr, :] = gene1.w_inp[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_rec_inp))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                    gene1.w_rec_inp[ind_cr, :] = gene2.w_rec_inp[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                    gene2.w_rec_inp[ind_cr, :] = gene1.w_rec_inp[ind_cr, :]
                else:
                    continue

            # FORGET GATES
            # Layer 1
            num_cross_overs = randint(1, len(gene1.w_forgetgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                    gene1.w_forgetgate[ind_cr, :] = gene2.w_forgetgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                    gene2.w_forgetgate[ind_cr, :] = gene1.w_forgetgate[ind_cr, :]
                else:
                    continue

            # Layer 2
            num_cross_overs = randint(1, len(gene1.w_rec_forgetgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                    gene1.w_rec_forgetgate[ind_cr, :] = gene2.w_rec_forgetgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                    gene2.w_rec_forgetgate[ind_cr, :] = gene1.w_rec_forgetgate[ind_cr, :]
                else:
                    continue

            # Layer 3
            num_cross_overs = randint(1, len(gene1.w_mem_forgetgate))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                    gene1.w_mem_forgetgate[ind_cr, :] = gene2.w_mem_forgetgate[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                    gene2.w_mem_forgetgate[ind_cr, :] = gene1.w_mem_forgetgate[ind_cr, :]
                else:
                    continue

            # OUTPUT WEIGHTS
            num_cross_overs = randint(1, len(gene1.w_output))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_output) - 1)
                    gene1.w_output[ind_cr, :] = gene2.w_output[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_output) - 1)
                    gene2.w_output[ind_cr, :] = gene1.w_output[ind_cr, :]
                else:
                    continue

            # MEMORY CELL (PRIOR)
            # 1-dimensional so point crossovers
            num_cross_overs = randint(1, int(gene1.w_rec_forgetgate.shape[1] / 2))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                    gene1.w_rec_forgetgate[0, ind_cr:] = gene2.w_rec_forgetgate[0, ind_cr:]
                elif rand < 0.66:
                    ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                    gene2.w_rec_forgetgate[0, :ind_cr] = gene1.w_rec_forgetgate[0, :ind_cr]
                else:
                    continue

            if self.num_substructures == 13:  # Only for NTM
                # WRITE GATES
                # Layer 1
                num_cross_overs = randint(1, len(gene1.w_writegate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_writegate) - 1)
                        gene1.w_writegate[ind_cr, :] = gene2.w_writegate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_writegate) - 1)
                        gene2.w_writegate[ind_cr, :] = gene1.w_writegate[ind_cr, :]
                    else:
                        continue

                # Layer 2
                num_cross_overs = randint(1, len(gene1.w_rec_writegate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                        gene1.w_rec_writegate[ind_cr, :] = gene2.w_rec_writegate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                        gene2.w_rec_writegate[ind_cr, :] = gene1.w_rec_writegate[ind_cr, :]
                    else:
                        continue

                # Layer 3
                num_cross_overs = randint(1, len(gene1.w_mem_writegate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                        gene1.w_mem_writegate[ind_cr, :] = gene2.w_mem_writegate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                        gene2.w_mem_writegate[ind_cr, :] = gene1.w_mem_writegate[ind_cr, :]
                    else:
                        continue

        else:  # Normal net
            # First layer
            num_cross_overs = randint(1, len(gene1.w_01))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_01) - 1)
                    gene1.w_01[ind_cr, :] = gene2.w_01[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_01) - 1)
                    gene2.w_01[ind_cr, :] = gene1.w_01[ind_cr, :]
                else:
                    continue

            # Second layer
            num_cross_overs = randint(1, len(gene1.w_12))
            for i in range(num_cross_overs):
                rand = random.random()
                if rand < 0.33:
                    ind_cr = randint(0, len(gene1.w_12) - 1)
                    gene1.w_12[ind_cr, :] = gene2.w_12[ind_cr, :]
                elif rand < 0.66:
                    ind_cr = randint(0, len(gene1.w_12) - 1)
                    gene2.w_12[ind_cr, :] = gene1.w_12[ind_cr, :]
                else:
                    continue

    def regularize_weight(self, weight):
        if weight > self.ssne_param.weight_magnitude_limit:
            weight = self.ssne_param.weight_magnitude_limit
        if weight < -self.ssne_param.weight_magnitude_limit:
            weight = -self.ssne_param.weight_magnitude_limit
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05

        # Initiate distribution
        if self.ssne_param.mut_distribution == 1:  # Gaussian
            ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
        elif self.ssne_param.mut_distribution == 2:  # Laplace
            ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
        elif self.ssne_param.mut_distribution == 3:  # Uniform
            ss_mut_dist = np.random.uniform(0, 1, self.num_substructures)
        else:
            ss_mut_dist = [1] * self.num_substructures

        if self.ssne_param.type_id == 'memoried':  # Memory net
            # INPUT GATES
            # Layer 1
            if random.random() < ss_mut_dist[0]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * gene.w_inpgate[
                            ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inpgate[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[1]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                               gene.w_rec_inpgate[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_rec_inpgate[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[2]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_inpgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_mem_inpgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_mem_inpgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                               super_mut_strength *
                                                                               gene.w_mem_inpgate[
                                                                                   ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_mem_inpgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_mem_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_mem_inpgate[ind_dim1, ind_dim2])

            # BLOCK INPUTS
            # Layer 1
            if random.random() < ss_mut_dist[3]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inp.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_inp.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_inp.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                       super_mut_strength * gene.w_inp[
                                                                           ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inp[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inp[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[4]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inp.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_inp.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_inp.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                           super_mut_strength * gene.w_rec_inp[
                                                                               ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inp[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_rec_inp[ind_dim1, ind_dim2])

            # FORGET GATES
            # Layer 1
            if random.random() < ss_mut_dist[5]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                              super_mut_strength *
                                                                              gene.w_forgetgate[
                                                                                  ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_forgetgate[ind_dim1, ind_dim2])

            # Layer 2
            if random.random() < ss_mut_dist[6]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_rec_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_rec_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_rec_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                  gene.w_rec_forgetgate[
                                                                                      ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_rec_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2])

            # Layer 3
            if random.random() < ss_mut_dist[7]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_mem_forgetgate.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_mem_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_mem_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                  gene.w_mem_forgetgate[
                                                                                      ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_mem_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2])

            # OUTPUT WEIGHTS
            if random.random() < ss_mut_dist[8]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_output.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_output.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_output.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_output[ind_dim1, ind_dim2] += random.gauss(0,
                                                                          super_mut_strength * gene.w_output[
                                                                              ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_output[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_output[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_output[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_output[ind_dim1, ind_dim2])

            # MEMORY CELL (PRIOR)
            if random.random() < ss_mut_dist[9]:
                num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                for i in range(num_mutations):
                    ind_dim1 = 0
                    ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                              super_mut_strength *
                                                                              gene.w_forgetgate[
                                                                                  ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                        gene.w_forgetgate[ind_dim1, ind_dim2])

            if self.num_substructures == 13:  # ONLY FOR NTM
                # WRITE GATES
                # Layer 1
                if random.random() < ss_mut_dist[10]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_writegate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_writegate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_writegate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 super_mut_strength *
                                                                                 gene.w_writegate[
                                                                                     ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                 mut_strength *
                                                                                 gene.w_writegate[
                                                                                     ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_writegate[ind_dim1, ind_dim2])

                # Layer 2
                if random.random() < ss_mut_dist[11]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_writegate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_rec_writegate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_rec_writegate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                     super_mut_strength *
                                                                                     gene.w_rec_writegate[
                                                                                         ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                     gene.w_rec_writegate[
                                                                                         ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_rec_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_rec_writegate[ind_dim1, ind_dim2])

                # Layer 3
                if random.random() < ss_mut_dist[12]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_writegate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_mem_writegate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_mem_writegate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                     super_mut_strength *
                                                                                     gene.w_mem_writegate[
                                                                                         ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                     gene.w_mem_writegate[
                                                                                         ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_mem_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_mem_writegate[ind_dim1, ind_dim2])

        else:  # Normal net
            # Layer 1
            num_mutations = randint(1, int(num_mutation_frac * gene.w_01.size))
            for i in range(num_mutations):
                ind_dim1 = randint(0, gene.w_01.shape[0] - 1)
                ind_dim2 = randint(0, gene.w_01.shape[1] - 1)
                if random.random() < super_mut_prob:  # Super mutation
                    gene.w_01[ind_dim1, ind_dim2] += random.gauss(0,
                                                                  super_mut_strength * gene.w_01[
                                                                      ind_dim1, ind_dim2])
                else:  # Normal mutation
                    gene.w_01[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_01[
                        ind_dim1, ind_dim2])

                # Regularization hard limit
                gene.w_01[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_01[ind_dim1, ind_dim2])

            # Layer 2
            num_mutations = randint(1, int(num_mutation_frac * gene.w_12.size))
            for i in range(num_mutations):
                ind_dim1 = randint(0, gene.w_12.shape[0] - 1)
                ind_dim2 = randint(0, gene.w_12.shape[1] - 1)
                if random.random() < super_mut_prob:  # Super mutation
                    gene.w_12[ind_dim1, ind_dim2] += random.gauss(0,
                                                                  super_mut_strength * gene.w_12[
                                                                      ind_dim1, ind_dim2])
                else:  # Normal mutation
                    gene.w_12[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_12[
                        ind_dim1, ind_dim2])

                # Regularization hard limit
                gene.w_12[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_12[ind_dim1, ind_dim2])

    def epoch(self):
        # Reset the memory Bank the adaptive/plastic structures for all genomes
        for gene in self.pop:
            gene.reset_bank()

        self.current_gen += 1
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(self.fitness_evals);
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = [];
        new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning eleitst candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.pop[replacee] = copy.deepcopy(self.pop[i])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[randint(0, len(unselects) - 1)])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.pop[i] = copy.deepcopy(self.pop[off_i])
            self.pop[j] = copy.deepcopy(self.pop[off_j])
            self.crossover_inplace(self.pop[i], self.pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(self.pop[i], self.pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.ssne_param.mutation_prob:
                    self.mutate_inplace(self.pop[i])

        # Bank the adaptive/plastic structures for all genomes with new changes
        for gene in self.pop:
            gene.set_bank()

    def save_pop(self, filename='Pop'):
        filename = str(self.current_gen) + '_' + filename
        pickle_object(self.pop, filename)

class SSNE:
        def __init__(self, parameters, ssne_param, arch_type):
            self.current_gen = 0
            self.parameters = parameters;
            self.ssne_param = ssne_param
            self.num_weights = self.ssne_param.total_num_weights;
            self.population_size = self.parameters.population_size;
            self.num_elitists = int(self.ssne_param.elite_fraction * parameters.population_size)
            if self.num_elitists < 1: self.num_elitists = 1

            self.fitness_evals = [[] for x in xrange(parameters.population_size)]  # Fitness eval list
            # Create population
            self.pop = []
            if self.ssne_param.type_id == 'memoried':
                if arch_type == 'quasi_gru':
                    for i in range(self.population_size):
                        self.pop.append(
                            Quasi_GRU(self.ssne_param.num_input, self.ssne_param.num_hnodes, self.ssne_param.num_output))
                    self.hof_net = Quasi_GRU(self.ssne_param.num_input, self.ssne_param.num_hnodes,
                                        self.ssne_param.num_output)
                    self.num_substructures = 10
                elif arch_type == 'quasi_ntm':
                    for i in range(self.population_size):
                        self.pop.append(
                            Quasi_NTM(self.ssne_param.num_input, self.ssne_param.num_hnodes, self.ssne_param.num_output))
                    self.hof_net = Quasi_NTM(self.ssne_param.num_input, self.ssne_param.num_hnodes,
                                              self.ssne_param.num_output)
                    self.num_substructures = 13
            else:
                for i in range(self.population_size):
                    self.pop.append(
                        normal_net(self.ssne_param.num_input, self.ssne_param.num_hnodes, self.ssne_param.num_output))
                self.hof_net = normal_net(self.ssne_param.num_input, self.ssne_param.num_hnodes,
                                          self.ssne_param.num_output)
                self.num_substructures = 4

        def selection_tournament(self, index_rank, num_offsprings, tournament_size):
            total_choices = len(index_rank)
            offsprings = []
            for i in range(num_offsprings):
                winner = np.min(np.random.randint(total_choices, size=tournament_size))
                offsprings.append(index_rank[winner])

            offsprings = list(set(offsprings))  # Find unique offsprings
            if len(offsprings) % 2 != 0:  # Number of offsprings should be even
                offsprings.append(offsprings[randint(0, len(offsprings) - 1)])
            return offsprings

        def list_argsort(self, seq):
            return sorted(range(len(seq)), key=seq.__getitem__)

        def crossover_inplace(self, gene1, gene2):

            if self.ssne_param.type_id == 'memoried':  # Memory net
                # INPUT GATES
                # Layer 1
                num_cross_overs = randint(1, len(gene1.w_inpgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                        gene1.w_inpgate[ind_cr, :] = gene2.w_inpgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_inpgate) - 1)
                        gene2.w_inpgate[ind_cr, :] = gene1.w_inpgate[ind_cr, :]
                    else:
                        continue

                # Layer 2
                num_cross_overs = randint(1, len(gene1.w_rec_inpgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                        gene1.w_rec_inpgate[ind_cr, :] = gene2.w_rec_inpgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_rec_inpgate) - 1)
                        gene2.w_rec_inpgate[ind_cr, :] = gene1.w_rec_inpgate[ind_cr, :]
                    else:
                        continue

                # Layer 3
                num_cross_overs = randint(1, len(gene1.w_mem_inpgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                        gene1.w_mem_inpgate[ind_cr, :] = gene2.w_mem_inpgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_mem_inpgate) - 1)
                        gene2.w_mem_inpgate[ind_cr, :] = gene1.w_mem_inpgate[ind_cr, :]
                    else:
                        continue

                # BLOCK INPUTS
                # Layer 1
                num_cross_overs = randint(1, len(gene1.w_inp))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_inp) - 1)
                        gene1.w_inp[ind_cr, :] = gene2.w_inp[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_inp) - 1)
                        gene2.w_inp[ind_cr, :] = gene1.w_inp[ind_cr, :]
                    else:
                        continue

                # Layer 2
                num_cross_overs = randint(1, len(gene1.w_rec_inp))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                        gene1.w_rec_inp[ind_cr, :] = gene2.w_rec_inp[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_rec_inp) - 1)
                        gene2.w_rec_inp[ind_cr, :] = gene1.w_rec_inp[ind_cr, :]
                    else:
                        continue

                # FORGET GATES
                # Layer 1
                num_cross_overs = randint(1, len(gene1.w_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                        gene1.w_forgetgate[ind_cr, :] = gene2.w_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_forgetgate) - 1)
                        gene2.w_forgetgate[ind_cr, :] = gene1.w_forgetgate[ind_cr, :]
                    else:
                        continue

                # Layer 2
                num_cross_overs = randint(1, len(gene1.w_rec_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                        gene1.w_rec_forgetgate[ind_cr, :] = gene2.w_rec_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_rec_forgetgate) - 1)
                        gene2.w_rec_forgetgate[ind_cr, :] = gene1.w_rec_forgetgate[ind_cr, :]
                    else:
                        continue

                # Layer 3
                num_cross_overs = randint(1, len(gene1.w_mem_forgetgate))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                        gene1.w_mem_forgetgate[ind_cr, :] = gene2.w_mem_forgetgate[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_mem_forgetgate) - 1)
                        gene2.w_mem_forgetgate[ind_cr, :] = gene1.w_mem_forgetgate[ind_cr, :]
                    else:
                        continue

                # OUTPUT WEIGHTS
                num_cross_overs = randint(1, len(gene1.w_output))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_output) - 1)
                        gene1.w_output[ind_cr, :] = gene2.w_output[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_output) - 1)
                        gene2.w_output[ind_cr, :] = gene1.w_output[ind_cr, :]
                    else:
                        continue

                # MEMORY CELL (PRIOR)
                # 1-dimensional so point crossovers
                num_cross_overs = randint(1, int(gene1.w_rec_forgetgate.shape[1] / 2))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                        gene1.w_rec_forgetgate[0, ind_cr:] = gene2.w_rec_forgetgate[0, ind_cr:]
                    elif rand < 0.66:
                        ind_cr = randint(0, gene1.w_rec_forgetgate.shape[1] - 1)
                        gene2.w_rec_forgetgate[0, :ind_cr] = gene1.w_rec_forgetgate[0, :ind_cr]
                    else:
                        continue

                if self.num_substructures == 13:  # Only for NTM
                    # WRITE GATES
                    # Layer 1
                    num_cross_overs = randint(1, len(gene1.w_writegate))
                    for i in range(num_cross_overs):
                        rand = random.random()
                        if rand < 0.33:
                            ind_cr = randint(0, len(gene1.w_writegate) - 1)
                            gene1.w_writegate[ind_cr, :] = gene2.w_writegate[ind_cr, :]
                        elif rand < 0.66:
                            ind_cr = randint(0, len(gene1.w_writegate) - 1)
                            gene2.w_writegate[ind_cr, :] = gene1.w_writegate[ind_cr, :]
                        else:
                            continue

                    # Layer 2
                    num_cross_overs = randint(1, len(gene1.w_rec_writegate))
                    for i in range(num_cross_overs):
                        rand = random.random()
                        if rand < 0.33:
                            ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                            gene1.w_rec_writegate[ind_cr, :] = gene2.w_rec_writegate[ind_cr, :]
                        elif rand < 0.66:
                            ind_cr = randint(0, len(gene1.w_rec_writegate) - 1)
                            gene2.w_rec_writegate[ind_cr, :] = gene1.w_rec_writegate[ind_cr, :]
                        else:
                            continue

                    # Layer 3
                    num_cross_overs = randint(1, len(gene1.w_mem_writegate))
                    for i in range(num_cross_overs):
                        rand = random.random()
                        if rand < 0.33:
                            ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                            gene1.w_mem_writegate[ind_cr, :] = gene2.w_mem_writegate[ind_cr, :]
                        elif rand < 0.66:
                            ind_cr = randint(0, len(gene1.w_mem_writegate) - 1)
                            gene2.w_mem_writegate[ind_cr, :] = gene1.w_mem_writegate[ind_cr, :]
                        else:
                            continue

            else:  # Normal net
                # First layer
                num_cross_overs = randint(1, len(gene1.w_01))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_01) - 1)
                        gene1.w_01[ind_cr, :] = gene2.w_01[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_01) - 1)
                        gene2.w_01[ind_cr, :] = gene1.w_01[ind_cr, :]
                    else:
                        continue

                # Second layer
                num_cross_overs = randint(1, len(gene1.w_12))
                for i in range(num_cross_overs):
                    rand = random.random()
                    if rand < 0.33:
                        ind_cr = randint(0, len(gene1.w_12) - 1)
                        gene1.w_12[ind_cr, :] = gene2.w_12[ind_cr, :]
                    elif rand < 0.66:
                        ind_cr = randint(0, len(gene1.w_12) - 1)
                        gene2.w_12[ind_cr, :] = gene1.w_12[ind_cr, :]
                    else:
                        continue

        def regularize_weight(self, weight):
            if weight > self.ssne_param.weight_magnitude_limit:
                weight = self.ssne_param.weight_magnitude_limit
            if weight < -self.ssne_param.weight_magnitude_limit:
                weight = -self.ssne_param.weight_magnitude_limit
            return weight

        def mutate_inplace(self, gene):
            mut_strength = 0.2
            num_mutation_frac = 0.2
            super_mut_strength = 10
            super_mut_prob = 0.05

            # Initiate distribution
            if self.ssne_param.mut_distribution == 1:  # Gaussian
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.ssne_param.mut_distribution == 2:  # Laplace
                ss_mut_dist = np.random.normal(random.random(), random.random() / 2, self.num_substructures)
            elif self.ssne_param.mut_distribution == 3:  # Uniform
                ss_mut_dist = np.random.uniform(0, 1, self.num_substructures)
            else:
                ss_mut_dist = [1] * self.num_substructures

            if self.ssne_param.type_id == 'memoried':  # Memory net
                # INPUT GATES
                # Layer 1
                if random.random() < ss_mut_dist[0]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inpgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_inpgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_inpgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * gene.w_inpgate[
                                ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inpgate[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inpgate[ind_dim1, ind_dim2])

                # Layer 2
                if random.random() < ss_mut_dist[1]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inpgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_rec_inpgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_rec_inpgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                   gene.w_rec_inpgate[
                                                                                       ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_rec_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inpgate[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_rec_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_rec_inpgate[ind_dim1, ind_dim2])

                # Layer 3
                if random.random() < ss_mut_dist[2]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_inpgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_mem_inpgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_mem_inpgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                   super_mut_strength *
                                                                                   gene.w_mem_inpgate[
                                                                                       ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_mem_inpgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_mem_inpgate[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_mem_inpgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_mem_inpgate[ind_dim1, ind_dim2])

                # BLOCK INPUTS
                # Layer 1
                if random.random() < ss_mut_dist[3]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_inp.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_inp.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_inp.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                           super_mut_strength * gene.w_inp[
                                                                               ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_inp[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_inp[ind_dim1, ind_dim2])

                # Layer 2
                if random.random() < ss_mut_dist[4]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_inp.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_rec_inp.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_rec_inp.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0,
                                                                               super_mut_strength * gene.w_rec_inp[
                                                                                   ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_rec_inp[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_rec_inp[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_rec_inp[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_rec_inp[ind_dim1, ind_dim2])

                # FORGET GATES
                # Layer 1
                if random.random() < ss_mut_dist[5]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_forgetgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_forgetgate[ind_dim1, ind_dim2])

                # Layer 2
                if random.random() < ss_mut_dist[6]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_forgetgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_rec_forgetgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_rec_forgetgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                      super_mut_strength *
                                                                                      gene.w_rec_forgetgate[
                                                                                          ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_rec_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                      gene.w_rec_forgetgate[
                                                                                          ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_rec_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_rec_forgetgate[ind_dim1, ind_dim2])

                # Layer 3
                if random.random() < ss_mut_dist[7]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_forgetgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_mem_forgetgate.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_mem_forgetgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                      super_mut_strength *
                                                                                      gene.w_mem_forgetgate[
                                                                                          ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_mem_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                      gene.w_mem_forgetgate[
                                                                                          ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_mem_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_mem_forgetgate[ind_dim1, ind_dim2])


                # OUTPUT WEIGHTS
                if random.random() < ss_mut_dist[8]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_output.size))
                    for i in range(num_mutations):
                        ind_dim1 = randint(0, gene.w_output.shape[0] - 1)
                        ind_dim2 = randint(0, gene.w_output.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_output[ind_dim1, ind_dim2] += random.gauss(0,
                                                                              super_mut_strength * gene.w_output[
                                                                                  ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_output[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_output[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_output[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_output[ind_dim1, ind_dim2])

                # MEMORY CELL (PRIOR)
                if random.random() < ss_mut_dist[9]:
                    num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_forgetgate.size))
                    for i in range(num_mutations):
                        ind_dim1 = 0
                        ind_dim2 = randint(0, gene.w_forgetgate.shape[1] - 1)
                        if random.random() < super_mut_prob:  # Super mutation
                            gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                  super_mut_strength *
                                                                                  gene.w_forgetgate[
                                                                                      ind_dim1, ind_dim2])
                        else:  # Normal mutation
                            gene.w_forgetgate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_forgetgate[
                                ind_dim1, ind_dim2])

                        # Regularization hard limit
                        gene.w_forgetgate[ind_dim1, ind_dim2] = self.regularize_weight(
                            gene.w_forgetgate[ind_dim1, ind_dim2])

                if self.num_substructures == 13: #ONLY FOR NTM
                    # WRITE GATES
                    # Layer 1
                    if random.random() < ss_mut_dist[10]:
                        num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_writegate.size))
                        for i in range(num_mutations):
                            ind_dim1 = randint(0, gene.w_writegate.shape[0] - 1)
                            ind_dim2 = randint(0, gene.w_writegate.shape[1] - 1)
                            if random.random() < super_mut_prob:  # Super mutation
                                gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                     super_mut_strength *
                                                                                     gene.w_writegate[
                                                                                         ind_dim1, ind_dim2])
                            else:  # Normal mutation
                                gene.w_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                     mut_strength *
                                                                                     gene.w_writegate[
                                                                                         ind_dim1, ind_dim2])

                            # Regularization hard limit
                            gene.w_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                                gene.w_writegate[ind_dim1, ind_dim2])

                    # Layer 2
                    if random.random() < ss_mut_dist[11]:
                        num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_rec_writegate.size))
                        for i in range(num_mutations):
                            ind_dim1 = randint(0, gene.w_rec_writegate.shape[0] - 1)
                            ind_dim2 = randint(0, gene.w_rec_writegate.shape[1] - 1)
                            if random.random() < super_mut_prob:  # Super mutation
                                gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                         super_mut_strength *
                                                                                         gene.w_rec_writegate[
                                                                                             ind_dim1, ind_dim2])
                            else:  # Normal mutation
                                gene.w_rec_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                         gene.w_rec_writegate[
                                                                                             ind_dim1, ind_dim2])

                            # Regularization hard limit
                            gene.w_rec_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                                gene.w_rec_writegate[ind_dim1, ind_dim2])

                    # Layer 3
                    if random.random() < ss_mut_dist[12]:
                        num_mutations = randint(1, math.ceil(num_mutation_frac * gene.w_mem_writegate.size))
                        for i in range(num_mutations):
                            ind_dim1 = randint(0, gene.w_mem_writegate.shape[0] - 1)
                            ind_dim2 = randint(0, gene.w_mem_writegate.shape[1] - 1)
                            if random.random() < super_mut_prob:  # Super mutation
                                gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0,
                                                                                         super_mut_strength *
                                                                                         gene.w_mem_writegate[
                                                                                             ind_dim1, ind_dim2])
                            else:  # Normal mutation
                                gene.w_mem_writegate[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *
                                                                                         gene.w_mem_writegate[
                                                                                             ind_dim1, ind_dim2])

                            # Regularization hard limit
                            gene.w_mem_writegate[ind_dim1, ind_dim2] = self.regularize_weight(
                                gene.w_mem_writegate[ind_dim1, ind_dim2])

            else:  # Normal net
                # Layer 1
                num_mutations = randint(1, int(num_mutation_frac * gene.w_01.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_01.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_01.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_01[ind_dim1, ind_dim2] += random.gauss(0,
                                                                      super_mut_strength * gene.w_01[
                                                                          ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_01[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_01[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_01[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_01[ind_dim1, ind_dim2])

                # Layer 2
                num_mutations = randint(1, int(num_mutation_frac * gene.w_12.size))
                for i in range(num_mutations):
                    ind_dim1 = randint(0, gene.w_12.shape[0] - 1)
                    ind_dim2 = randint(0, gene.w_12.shape[1] - 1)
                    if random.random() < super_mut_prob:  # Super mutation
                        gene.w_12[ind_dim1, ind_dim2] += random.gauss(0,
                                                                      super_mut_strength * gene.w_12[
                                                                          ind_dim1, ind_dim2])
                    else:  # Normal mutation
                        gene.w_12[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * gene.w_12[
                            ind_dim1, ind_dim2])

                    # Regularization hard limit
                    gene.w_12[ind_dim1, ind_dim2] = self.regularize_weight(gene.w_12[ind_dim1, ind_dim2])

        def epoch(self):
            # Reset the memory Bank the adaptive/plastic structures for all genomes
            for gene in self.pop:
                gene.reset_bank()

            self.current_gen += 1
            # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
            index_rank = self.list_argsort(self.fitness_evals);
            index_rank.reverse()
            elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

            # Selection step
            offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                                   tournament_size=3)

            # Figure out unselected candidates
            unselects = [];
            new_elitists = []
            for i in range(self.population_size):
                if i in offsprings or i in elitist_index:
                    continue
                else:
                    unselects.append(i)
            random.shuffle(unselects)

            # Elitism step, assigning eleitst candidates to some unselects
            for i in elitist_index:
                replacee = unselects.pop(0)
                new_elitists.append(replacee)
                self.pop[replacee] = copy.deepcopy(self.pop[i])

            # Crossover for unselected genes with 100 percent probability
            if len(unselects) % 2 != 0:  # Number of unselects left should be even
                unselects.append(unselects[randint(0, len(unselects) - 1)])
            for i, j in zip(unselects[0::2], unselects[1::2]):
                off_i = random.choice(new_elitists);
                off_j = random.choice(offsprings)
                self.pop[i] = copy.deepcopy(self.pop[off_i])
                self.pop[j] = copy.deepcopy(self.pop[off_j])
                self.crossover_inplace(self.pop[i], self.pop[j])

            # Crossover for selected offsprings
            for i, j in zip(offsprings[0::2], offsprings[1::2]):
                if random.random() < self.ssne_param.crossover_prob: self.crossover_inplace(self.pop[i], self.pop[j])

            # Mutate all genes in the population except the new elitists
            for i in range(self.population_size):
                if i not in new_elitists:  # Spare the new elitists
                    if random.random() < self.ssne_param.mutation_prob:
                        self.mutate_inplace(self.pop[i])

            # Bank the adaptive/plastic structures for all genomes with new changes
            for gene in self.pop:
                gene.set_bank()

        def save_pop(self, filename='Pop'):
            filename = str(self.current_gen) + '_' + filename
            pickle_object(self.pop, filename)

class statistics(): #Tracker
    def __init__(self, parameters):
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if parameters.is_memoried_predator:
            if parameters.is_memoried_prey:
                self.file_save = 'mem_mem.csv'
            else:
                self.file_save = 'mem_norm.csv'
        else:
            if parameters.is_memoried_prey:
                self.file_save = 'norm_mem.csv'
            else:
                self.file_save = 'norm_norm.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = 'avg_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = 'hof_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')


    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')


def import_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    from keras.models import model_from_json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def simulator_test_perfect(model, filename = 'ColdAir.csv', downsample_rate=25):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

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

    print ('TESTING NOW')
    input = np.reshape(data[0], (1, 21))  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data)-1):  # For all training examples
        model_out = model.predict(input)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]# * normalizer[index] + min[index]
            track_target[index][example] = data[example+1][index]# * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out)):
            input[k] = input = np.reshape(data[example+1], (1, 21))
        # Fill in two control variables
        input[0][19] = data[example + 1][19]
        input[0][20] = data[example + 1][20]


    for index in range(19):
        plt.plot(track_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-',label='TF_Simulator: ' + str(index))
        #np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        #np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        #print track_output[index]
        plt.show()

def simulator_results(model, filename = 'ColdAir.csv', downsample_rate=25):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

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

    print ('TESTING NOW')
    input = np.reshape(data[0], (1, 21))  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data)-1):  # For all training examples
        model_out = model.predict(input)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]# * normalizer[index] + min[index]
            track_target[index][example] = data[example+1][index]# * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out[0])):
            input[0][k] = model_out[0][k]
        # Fill in two control variables
        input[0][19] = data[example + 1][19]
        input[0][20] = data[example + 1][20]


    for index in range(19):
        plt.plot(track_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-',label='TF_Simulator: ' + str(index))
        #np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        #np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        #print track_output[index]
        plt.show()

def controller_results(individual, setpoints, start_controller_input, start_sim_input, simulator):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    control_input = np.copy(start_controller_input)  # Input to the controller
    sim_input = np.copy(start_sim_input)  # Input to the simulator
    track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))

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

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.predict(control_input)
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


        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11]
        #weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = simulator_out[0][i]
            control_input[0][i] = simulator_out[0][i]

        #decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
        #setpoints = np.array(setpoints[0:-1])
        #setpoints = np.concatenate((decorator, setpoints))
        #track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints, 'r--', label='Desired Turbine Speed' )
    plt.plot(track_output, 'b-', label='Achieved Turbine Speed')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='upper right', prop={'size': 15})
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

def pt_controller_results(individual, setpoints, start_controller_input, start_sim_input,
                           simulator):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    control_input = np.copy(start_controller_input)  # Input to the controller
    sim_input = np.copy(start_sim_input)  # Input to the simulator
    track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))
    individual.fast_net.reset()

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

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
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


        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11]
        # weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = simulator_out[0][i]
            control_input[0][i] = simulator_out[0][i]

            # decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
            # setpoints = np.array(setpoints[0:-1])
            # setpoints = np.concatenate((decorator, setpoints))
            # track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints, 'r--', label='Desired Turbine Speed')
    plt.plot(track_output, 'b-', label='Achieved Turbine Speed')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='upper right', prop={'size': 15})
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

def controller_results_bprop(individual, setpoints, start_controller_input, start_sim_input, simulator, train_x):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    #Normalizer #TODO DONT DO THIS
    if True:
        # Import training data and clear away the two top lines
        downsample_rate = 25
        data = np.loadtxt('ColdAir.csv', delimiter=',', skiprows=2)

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

        control_input = np.copy(start_controller_input)  # Input to the controller
        sim_input = np.copy(start_sim_input)  # Input to the simulator
        track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))

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

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.predict(control_input)

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


        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11] * normalizer[11] + min[11]
        setpoints[example][0] = setpoints[example][0] * normalizer[11] + min[11]
        #weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = train_x[example+1][i]
            control_input[0][i] = train_x[example+1][i]

            #sim_input[0][i] = simulator_out[0][i]
            #control_input[0][i] = simulator_out[0][i]



        #decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
        #setpoints = np.array(setpoints[0:-1])
        #setpoints = np.concatenate((decorator, setpoints))
        #track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints[0:-10,0:], 'r--', label='Target Turbine Speed' )
    plt.plot(track_output[0:-10,0:], 'b-', label='Controller Turbine Speed ')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='lower right', prop={'size': 15})
    plt.xlabel("Time (min)", fontsize = 15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

def pickle_object(obj, filename):
    import cPickle
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

def roulette_wheel(scores):
    scores = scores / np.sum(scores)  # Normalize
    rand = random.random()
    counter = 0
    for i in range(len(scores)):
        counter += scores[i]
        if rand < counter:
            return i

def team_selection(gridworld, parameters):
    # MAKE SELECTION POOLS
    selection_pool = [];
    max_pool_size = 0  # Selection pool listing the individuals with multiples for to match number of evaluations
    for i in range(parameters.num_predator):  # Filling the selection pool
        if parameters.use_neat:
            ig_num_individuals = len(
                gridworld.predator_list[i].evo_net.genome_list)  # NEAT's number of individuals can change
        else:
            ig_num_individuals = parameters.population_size  # For keras_evo-net the number of individuals stays constant at population size
        selection_pool.append(np.arange(ig_num_individuals))
        for j in range(parameters.num_evals_ccea - 1): selection_pool[i] = np.append(selection_pool[i],
                                                                     np.arange(ig_num_individuals))
        if len(selection_pool[i]) > max_pool_size: max_pool_size = len(selection_pool[i])
    for i in range(parameters.num_prey):  # Filling the selection pool
        if parameters.use_neat:
            ig_num_individuals = len(
                gridworld.prey_list[i].evo_net.genome_list)  # NEAT's number of individuals can change
        else:
            ig_num_individuals = parameters.population_size  # For keras_evo-net the number of individuals stays constant at population size
        selection_pool.append(np.arange(ig_num_individuals))
        for j in range(parameters.num_evals_ccea - 1): selection_pool[i + parameters.num_predator] = np.append(
            selection_pool[i + parameters.num_predator], np.arange(ig_num_individuals))
        if len(selection_pool[i + parameters.num_predator]) > max_pool_size: max_pool_size = len(
            selection_pool[i + parameters.num_predator])

    if parameters.use_neat:
        for i, pool in enumerate(selection_pool):  # Equalize the selection pool
            diff = max_pool_size - len(pool)
            if diff != 0:
                ig_cap = len(pool) / parameters.num_evals_ccea
                while diff > ig_cap:
                    selection_pool[i] = np.append(selection_pool[i], np.arange(ig_cap))
                    diff -= ig_cap
                selection_pool[i] = np.append(selection_pool[i], np.arange(diff))

    return selection_pool



#BACKUPS
def pstats():
    import pstats
    p = pstats.Stats('profile.profile')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('cumulative').print_stats(50)
    p.sort_stats('cumulative').print_stats(50)

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

def return_mem_address(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]