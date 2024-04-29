import numpy as np
import csv

# Load human data
human_data_int1 = np.load('./behavioral_data_int1/ind_subj_results.npz')
human_data_int2 = np.load('./behavioral_data_int2/ind_subj_results.npz')
# Load GPT-4 data
gpt4_data_int1 = np.load('./gpt-4-0125-preview_int1/acc.npz')
gpt4_data_int2 = np.load('./gpt-4-0125-preview_int2/acc.npz')

# Create CSV comparing GPT-4 w/ human behavior
header = ['subjID', 'correct_pred', 'human_vs_gpt', 'prob_type', 'intsize']
with open('human_vs_GPT4.csv', 'w') as fid:
	writer = csv.writer(fid)
	writer.writerow(header)
	# Write human data
	# Interval size = 1
	for s in range(human_data_int1['all_subj_correct_pred'].shape[0]):
		for p in range(human_data_int1['all_subj_correct_pred'].shape[1]):
			row = [s, int(human_data_int1['all_subj_correct_pred'][s,p]), 0, human_data_int1['all_subj_prob_subtype'][s,p], 0]
			writer.writerow(row)
	# Interval size = 2
	for s in range(human_data_int2['all_subj_correct_pred'].shape[0]):
		for p in range(human_data_int2['all_subj_correct_pred'].shape[1]):
			row = [s + human_data_int1['all_subj_correct_pred'].shape[0], int(human_data_int2['all_subj_correct_pred'][s,p]), 0, human_data_int2['all_subj_prob_subtype'][s,p], 1]
			writer.writerow(row)
	# Write GPT-4 data
	s = human_data_int1['all_subj_correct_pred'].shape[0] + human_data_int2['all_subj_correct_pred'].shape[0]
	# Interval size = 1
	for t in range(gpt4_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_data_int1['ind_trial_results'].shape[0]):
			row = [s, int(gpt4_data_int1['ind_trial_results'][p,t]), 1, p, 0]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_data_int2['ind_trial_results'].shape[0]):
			row = [s, int(gpt4_data_int2['ind_trial_results'][p,t]), 1, p, 1]
			writer.writerow(row)

# Load data for GPT4 + code execution
gpt4_CE_data_int1 = np.load('./gpt-4-0125-preview_code_execution_int1/acc.npz')
gpt4_CE_data_int2 = np.load('./gpt-4-0125-preview_code_execution_int2/acc.npz')

# Create CSV comparing GPT-4 + code execution w/ human behavior
with open('./human_vs_GPT4_code_execution.csv', 'w') as fid:
	writer = csv.writer(fid)
	writer.writerow(header)
	# Write human data
	# Interval size = 1
	for s in range(human_data_int1['all_subj_correct_pred'].shape[0]):
		for p in range(human_data_int1['all_subj_correct_pred'].shape[1]):
			row = [s, int(human_data_int1['all_subj_correct_pred'][s,p]), 0, human_data_int1['all_subj_prob_subtype'][s,p], 0]
			writer.writerow(row)
	# Interval size = 2
	for s in range(human_data_int2['all_subj_correct_pred'].shape[0]):
		for p in range(human_data_int2['all_subj_correct_pred'].shape[1]):
			row = [s + human_data_int1['all_subj_correct_pred'].shape[0], int(human_data_int2['all_subj_correct_pred'][s,p]), 0, human_data_int2['all_subj_prob_subtype'][s,p], 1]
			writer.writerow(row)
	# Write GPT-4 data
	s = human_data_int1['all_subj_correct_pred'].shape[0] + human_data_int2['all_subj_correct_pred'].shape[0]
	# Interval size = 1
	for t in range(gpt4_CE_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_data_int1['ind_trial_results'].shape[0]):
			row = [s, int(gpt4_CE_data_int1['ind_trial_results'][p,t]), 1, p, 0]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_CE_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_data_int2['ind_trial_results'].shape[0]):
			row = [s, int(gpt4_CE_data_int2['ind_trial_results'][p,t]), 1, p, 1]
			writer.writerow(row)

# Load data for GPT-4 + code execution w/ alternative synthetic alphabet
gpt4_CE_altalphabet_data_int1 = np.load('./gpt-4-0125-preview_code_execution_int1_altalphabet/acc.npz')
gpt4_CE_altalphabet_data_int2 = np.load('./gpt-4-0125-preview_code_execution_int2_altalphabet/acc.npz')

# Create CSV comparing GPT-4 + code execution on original vs. alternative synthetic alphabets
header = ['correct_pred', 'prob_type', 'intsize', 'alphabet_type']
with open('./GPT4_code_execution_comparing_alphabets.csv', 'w') as fid:
	writer = csv.writer(fid)
	writer.writerow(header)
	# Write GPT-4 data for original alphabet
	# Interval size = 1
	for t in range(gpt4_CE_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_data_int1['ind_trial_results'].shape[0]):
			row = [int(gpt4_CE_data_int1['ind_trial_results'][p,t]), p, 0, 0]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_CE_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_data_int2['ind_trial_results'].shape[0]):
			row = [int(gpt4_CE_data_int2['ind_trial_results'][p,t]), p, 1, 0]
			writer.writerow(row)
	# Write GPT-4 data for alternative alphabet
	# Interval size = 1
	for t in range(gpt4_CE_altalphabet_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_altalphabet_data_int1['ind_trial_results'].shape[0]):
			row = [int(gpt4_CE_altalphabet_data_int1['ind_trial_results'][p,t]), p, 0, 1]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_CE_altalphabet_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_CE_altalphabet_data_int2['ind_trial_results'].shape[0]):
			row = [int(gpt4_CE_altalphabet_data_int2['ind_trial_results'][p,t]), p, 1, 1]
			writer.writerow(row)

# Load data for older GPT-4 engine
gpt4_older_data_int1 = np.load('./gpt-4-1106-preview_int1/acc.npz')
gpt4_older_data_int2 = np.load('./gpt-4-1106-preview_int2/acc.npz')

# Create CSV comparing GPT-4 w/ old (1106) vs. new (0125) engine
header = ['correct_pred', 'prob_type', 'intsize', 'engine_type']
with open('./GPT4_comparing_engines.csv', 'w') as fid:
	writer = csv.writer(fid)
	writer.writerow(header)
	# Write GPT-4 data
	# Interval size = 1
	for t in range(gpt4_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_data_int1['ind_trial_results'].shape[0]):
			row = [int(gpt4_data_int1['ind_trial_results'][p,t]), p, 0, 0]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_data_int2['ind_trial_results'].shape[0]):
			row = [int(gpt4_data_int2['ind_trial_results'][p,t]), p, 1, 0]
			writer.writerow(row)
	# Write data for older GPT-4 engine
	# Interval size = 1
	for t in range(gpt4_older_data_int1['ind_trial_results'].shape[1]):
		for p in range(gpt4_older_data_int1['ind_trial_results'].shape[0]):
			row = [int(gpt4_older_data_int1['ind_trial_results'][p,t]), p, 0, 1]
			writer.writerow(row)
	# Interval size = 2
	for t in range(gpt4_older_data_int2['ind_trial_results'].shape[1]):
		for p in range(gpt4_older_data_int2['ind_trial_results'].shape[0]):
			row = [int(gpt4_older_data_int2['ind_trial_results'][p,t]), p, 1, 1]
			writer.writerow(row)

