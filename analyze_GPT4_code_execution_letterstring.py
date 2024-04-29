import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import argparse
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
parser.add_argument('--alt_alphabet', action='store_true', help="Alternative synthetic alphabet.")
args = parser.parse_args()

# Data directory
data_dir = './' + args.gpt4_engine + '_code_execution/'

# Create directory for results
results_dir = './' + args.gpt4_engine + '_code_execution_int' + str(args.interval_size)
if args.alt_alphabet:
	results_dir += '_altalphabet'
results_dir += '/'
check_path(results_dir)

# Calculate accuracy for all problem types
all_prob_types = ['add_letter', 'succ', 'pred', 'remove_redundant', 'fix_alphabet', 'sort']
all_acc = []
all_ci_lower = []
all_ci_upper = []
ind_trial_results = []
for p in range(len(all_prob_types)):
	data_fname = data_dir + 'int' + str(args.interval_size) + '_' + all_prob_types[p]
	if args.alt_alphabet:
		data_fname += '_altalphabet'
	data_fname += '_results.npz'
	correct_pred = np.load(data_fname)['prob_type_correct']
	all_acc.append(correct_pred.mean())
	ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
	all_ci_lower.append(ci_lower)
	all_ci_upper.append(ci_upper)
	ind_trial_results.append(correct_pred)
all_acc = np.array(all_acc)
all_ci_lower = np.array(all_ci_lower)
all_ci_upper = np.array(all_ci_upper)
all_lower_err = all_acc - all_ci_lower
all_upper_err =  all_ci_upper - all_acc
all_err = np.array([all_lower_err, all_upper_err])
ind_trial_results = np.array(ind_trial_results)

# Plot settings
gpt4_color = 'darkmagenta'
plot_fontsize = 10
title_fontsize = 12
axis_label_fontsize = 12
bar_width = 0.8

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
save_fname = results_dir
save_fname += 'acc.png'
plt.savefig(save_fname, dpi=300, bbox_inches="tight")
plt.close()
# Calculate overall accuracy
overall_acc = ind_trial_results.mean()
ci_lower, ci_upper = proportion_confint(ind_trial_results.sum(), ind_trial_results.flatten().shape[0])
lower_err = overall_acc - ci_lower
upper_err =  ci_upper - overall_acc 
overall_err = np.array([lower_err, upper_err])
print('Overall accuracy = ' + str(overall_acc))
# Save results
save_fname = results_dir
save_fname += 'acc.npz'
np.savez(save_fname, all_acc=all_acc, all_err=all_err, overall_acc=overall_acc, overall_err=overall_err, ind_trial_results=ind_trial_results)

