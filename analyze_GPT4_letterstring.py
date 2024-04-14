import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import builtins
import argparse
import pdb
import os

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt4_engine', type=str, default='gpt-4-0125-preview', help='GPT-4 engine')
args = parser.parse_args()

# Load data
data_fname = './' + args.gpt4_engine + '_int' + str(args.interval_size)
data_fname += '_results.npz'
all_responses = np.load(data_fname)['all_prob_type_responses']
N_prob_types = all_responses.shape[0]
N_trials_per_prob_type = all_responses.shape[1]
# Load problems
if args.interval_size == 1:
	all_prob = np.load('./all_prob_synthetic_int1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
	all_prob = np.load('./all_prob_synthetic_int2.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())

# Calculate performance
all_prob_type_correct_pred = []
for p in range(N_prob_types):
	all_correct_pred = []
	for t in range(N_trials_per_prob_type):
		response = all_responses[p][t]
		response_parsed = response.split('Answer:')[-1].split('[')[-1].split(']')[0].split(' ')
		print(response_parsed)
		correct_answer = all_prob.item()[prob_types[p]]['prob'][t][1][1]
		if np.array(correct_answer).astype(str).shape[0] == np.array(response_parsed).astype(str).shape[0]:
			correct_pred = np.all(np.array(correct_answer).astype(str) == np.array(response_parsed))
		else:
			correct_pred = False
		all_correct_pred.append(correct_pred)
	all_prob_type_correct_pred.append(all_correct_pred)
# Convert to arrays
all_prob_type_correct_pred = np.array(all_prob_type_correct_pred)

# Create directory for results
results_dir = './' + args.gpt4_engine + '_int' + str(args.interval_size)
results_dir += '/'
check_path(results_dir)

# Plot settings
gpt4_color = 'darkmagenta'
plot_fontsize = 10
title_fontsize = 12
axis_label_fontsize = 12
bar_width = 0.8

# Calculate accuracy for all problem types
all_prob_types = ['add_letter', 'succ', 'pred', 'remove_redundant', 'fix_alphabet', 'sort']
all_acc = []
all_ci_lower = []
all_ci_upper = []
ind_trial_results = []
for p in range(len(all_prob_types)):
	correct_pred = np.array(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_prob_types[p])[0][0]]).astype(float)
	all_acc.append(correct_pred.mean())
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_ci_lower.append(ci_lower)
	all_ci_upper.append(ci_upper)
	ind_trial_results.append(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_prob_types[p])[0][0]])
all_acc = np.array(all_acc)
all_ci_lower = np.array(all_ci_lower)
all_ci_upper = np.array(all_ci_upper)
all_lower_err = all_acc - all_ci_lower
all_upper_err =  all_ci_upper - all_acc
all_err = np.array([all_lower_err, all_upper_err])
ind_trial_results = np.array(ind_trial_results)

# Plot
all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
x_points = np.arange(len(all_prob_types))
ax = plt.subplot(111)
plt.bar(x_points, all_acc, yerr=all_err, color=gpt4_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_prob_type_names), fontsize=plot_fontsize)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.legend(['GPT-4'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig(results_dir + 'acc.png', dpi=300, bbox_inches="tight")
plt.close()
# Calculate overall accuracy
overall_acc = all_prob_type_correct_pred.mean()
ci_lower, ci_upper = proportion_confint(all_prob_type_correct_pred.sum(), all_prob_type_correct_pred.flatten().shape[0])
lower_err = overall_acc - ci_lower
upper_err =  ci_upper - overall_acc 
overall_err = np.array([lower_err, upper_err])
print('Overall accuracy = ' + str(overall_acc))
# Save results
np.savez(results_dir + 'acc.npz', all_acc=all_acc, all_err=all_err, overall_acc=overall_acc, overall_err=overall_err, ind_trial_results=ind_trial_results)

