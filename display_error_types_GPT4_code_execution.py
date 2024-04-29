import numpy as np
import argparse

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpt4_engine', type=str, default='gpt-4-0125-preview', help='GPT-4 engine')
parser.add_argument('--alt_alphabet', action='store_true', help="Alternative synthetic alphabet.")
args = parser.parse_args()

# Data directory
data_dir = './' + args.gpt4_engine + '_code_execution/'

# Load responses for error trials and categorize as wrong vs. based on valid alternative rule
prob_types = ['add_letter', 'succ', 'pred', 'remove_redundant', 'fix_alphabet', 'sort']
N_error = 0
N_valid_altrule = 0
for intsize in range(2):
	print('Interval size = ' + str(intsize+1) + '\n')
	for p in range(len(prob_types)):
		print('Problem type: ' + str(prob_types[p]))
		# Load responses
		data_fname = data_dir + 'int' + str(intsize+1) + '_' + prob_types[p]
		if args.alt_alphabet:
			data_fname += '_altalphabet'
		data_fname += '_results.npz'
		valid_altrule = np.load(data_fname)['valid_altrule']
		# Report results
		print(str(int(valid_altrule.sum())) + ' / ' + str(len(valid_altrule)) + ' errors based on valid alternate rule\n')
		N_error += len(valid_altrule)
		N_valid_altrule += int(valid_altrule.sum())

# Report final results
print(str(N_valid_altrule) + ' / ' + str(N_error) + ' errors based on valid alternate rule')