from __future__ import absolute_import
import os
import sys
import time
import argparse
import shutil
import time
import libs.q_learner as ql
from libs.state_string_utils import StateStringUtils  
from libs.state_enumerator import State, StateEnumerator
import models.mnist.hyper_parameters
import models.mnist.state_space_parameters as state_space_parameters

parser = argparse.ArgumentParser(description='MetaQNN MNIST Training')
parser.add_argument('--data', metavar='DIR', default='./MNIST' , #give path to dataset
                    help='path to dataset')

def main():
	global args
	args = parser.parse_args()
	gen = ql.QLearner(state_space_parameters, 1, args.data)
	enum = StateEnumerator(state_space_parameters)

	for episode in state_space_parameters.epsilon_schedule:

		epsilon = episode[0]
		M = episode[1]

		for ite in range(1,M+1):
			previous_action_values = gen.qstore.q.copy()
			gen.generate_net(epsilon)
			gen.sample_replay_for_update()

	gen.qstore.save_to_csv('./qVal1.csv')
	gen.replay_dictionary.to_csv('./replayDict1.csv')

if __name__ == '__main__':
    main()





