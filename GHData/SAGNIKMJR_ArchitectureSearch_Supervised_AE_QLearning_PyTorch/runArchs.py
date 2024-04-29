import argparse
from pandas import read_csv
import pandas as pd
from libs.state_string_utils import StateStringUtils
import libs.train_ as train_
import libs.state_enumerator as se
import libs.cnn as cnn
import models.mnist.hyper_parameters
import models.mnist.state_space_parameters as state_space_parameters

parser = argparse.ArgumentParser(description='Reconstruction architecure training')
parser.add_argument('--data', metavar='DIR', default='/home/shared/sagnik/datasets/MNIST' , #give path to dataset
                    help='path to dataset')
parser.add_argument('--csv_path', metavar = 'CSV_PATH', default = './replayDict1.csv', \
					help = 'path to csv sorted in terms of inverse loss')
parser.add_argument('--no', metavar='NO', default = 100, help ='no. of top architectures to be trained')

def main():
	global args
	args = parser.parse_args()
	replay_dictionary_inverse_loss = read_csv(args.csv_path)
	replay_dictionary_acc = pd.DataFrame(columns=['net',
                                              'accuracy_best_val',
                                              'accuracy_last_val',
                                              'epsilon',
                                              'train_flag'])
	for i in range(int(args.no)):
		row = replay_dictionary_inverse_loss.iloc[[i]]
		net = row['net'].values[0]
		epsilon = row['epsilon'].values[0]
		# net = row['net']
		# print(net)
		stringutils = StateStringUtils(state_space_parameters)
		state_list = stringutils.convert_model_string_to_states(cnn.parse('net', net))
		state_list = stringutils.add_drop_out_states(state_list)
        # net_string = self.stringutils.state_list_to_string(bucketed_state_list)
		net_string = stringutils.state_list_to_string(state_list)
		# print(state_list[3].layer_type, state_list[3].filter_size)
		acc_best_val, acc_last_val, train_flag = train_.train_val_net2(state_list,\
																	  state_space_parameters, \
																	  args.data)
		replay_dictionary_acc = replay_dictionary_acc.append(pd.DataFrame([[net_string, acc_best_val, acc_last_val, \
		                    epsilon, train_flag]], columns=['net', 'accuracy_best_val', \
		                    'accuracy_last_val', 'epsilon', 'train_flag']), ignore_index = True)
	replay_dictionary_acc.to_csv('./replayDict1_acc.csv')




if __name__ == '__main__':
	main()
