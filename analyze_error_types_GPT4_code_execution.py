import numpy as np
import argparse
import builtins

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpt4_engine', type=str, default='gpt-4-0125-preview', help='GPT-4 engine')
parser.add_argument('--alt_alphabet', action='store_true', help="Alternative synthetic alphabet.")
args = parser.parse_args()

# Data directory
data_dir = './' + args.gpt4_engine + '_code_execution/'

# Load responses for error trials and categorize as wrong vs. based on valid alternative rule
N_error = 0
N_valid_altrule = 0
for intsize in range(2):
	print('Interval size = ' + str(intsize+1) + '\n')
	# Load problems
	all_prob_fname = './all_prob_synthetic_int' + str(intsize+1) + '.npz'
	all_prob = np.load(all_prob_fname, allow_pickle=True)['all_prob']
	prob_types = builtins.list(all_prob.item().keys())
	prob_types = prob_types[:6]
	for p in range(len(prob_types)):
		print('Problem type: ' + str(prob_types[p]) + '\n')
		prob_ind = np.where(np.array(prob_types) == prob_types[p])[0][0]
		# Load responses
		data_fname = data_dir + 'int' + str(intsize+1) + '_' + prob_types[p]
		if args.alt_alphabet:
			data_fname += '_altalphabet'
		data_fname += '_results.npz'
		prob_type_responses = np.load(data_fname)['prob_type_responses']
		prob_type_correct = np.load(data_fname)['prob_type_correct']
		valid_altrule = []
		for i in range(len(prob_type_correct)):
			if prob_type_correct[i] == 0:
				print('trial ' + str(i+1) + ' of ' + str(len(prob_type_correct)) + '...\n')
				# Response
				response = prob_type_responses[i]
				# Load solution
				solution = ''.join(all_prob.item()[prob_types[prob_ind]]['prob'][i][1][1])
				# Print response and solution
				print(response + 'Correct answer: ' + solution + '\n\n')
				# Categorize error
				error_type = int(input('[0] Wrong\n[1] Valid alternate rule\nResponse: '))
				valid_altrule.append(error_type)
				# Update counts
				N_error += 1
				if error_type == 1:
					N_valid_altrule += 1
		# Save data
		np.savez(data_fname, prob_type_responses=prob_type_responses, prob_type_correct=prob_type_correct, valid_altrule=valid_altrule)

# Report final results
print(str(N_valid_altrule) + ' / ' + str(N_error) + ' errors based on valid alternate rule')