import numpy as np
import matplotlib.pyplot as plt
import builtins
from scipy.stats import sem
from statsmodels.stats.proportion import proportion_confint
from itertools import combinations
import argparse
import glob
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

# Load data
# GPT-4
gpt4_int1_results = np.load('./gpt-4-0125-preview_int1/acc.npz')
gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
# GPT-4 + analysis
gpt4_analysis_int1_results = np.load('./GPT4_analysis_int1_results.npz')
gpt4_analysis_int2_results = np.load('./GPT4_analysis_int2_results.npz')
# Human
human_int1_results = np.load('./behavioral_data_int1/acc.npz')
human_int2_results = np.load('./behavioral_data_int2/acc.npz')

# Get accuracy for each condition
## GPT-4
# Interval size = 1
gpt4_int1_acc = gpt4_int1_results['overall_acc'].item()
gpt4_int1_err = gpt4_int1_results['overall_err'][0]
# Interval size = 2
gpt4_int2_acc = gpt4_int2_results['overall_acc'].item()
gpt4_int2_err = gpt4_int1_results['overall_err'][0]
# Combined
gpt4_acc = np.array([gpt4_int1_acc, gpt4_int2_acc])
gpt4_err = np.array([gpt4_int1_err, gpt4_int2_err])
## GPT-4 + analysis
# Interval size = 1
gpt4_analysis_int1_N_correct = gpt4_analysis_int1_results['correct'].sum()
gpt4_analysis_int1_N = gpt4_analysis_int1_results['N_per_probtype'] * gpt4_analysis_int1_results['correct'].shape[0]
gpt4_analysis_int1_acc = gpt4_analysis_int1_N_correct / gpt4_analysis_int1_N
ci_lower, _ = proportion_confint(gpt4_analysis_int1_N_correct, gpt4_analysis_int1_N)
gpt4_analysis_int1_err = gpt4_analysis_int1_acc - ci_lower
# Interval size = 2
gpt4_analysis_int2_N_correct = gpt4_analysis_int2_results['correct'].sum()
gpt4_analysis_int2_N = gpt4_analysis_int2_results['N_per_probtype'] * gpt4_analysis_int2_results['correct'].shape[0]
gpt4_analysis_int2_acc = gpt4_analysis_int2_N_correct / gpt4_analysis_int2_N
ci_lower, _ = proportion_confint(gpt4_analysis_int2_N_correct, gpt4_analysis_int2_N)
gpt4_analysis_int2_err = gpt4_analysis_int2_acc - ci_lower
# Combined
gpt4_analysis_acc = np.array([gpt4_analysis_int1_acc, gpt4_analysis_int2_acc])
gpt4_analysis_err = np.array([gpt4_analysis_int1_err, gpt4_analysis_int2_err])
## Human
# Interval size = 1
human_int1_acc = human_int1_results['overall_acc'].item()
human_int1_err = human_int1_results['overall_err'].item()
# Interval size = 2
human_int2_acc = human_int2_results['overall_acc'].item()
human_int2_err = human_int1_results['overall_err'].item()
# Combined
human_acc = np.array([human_int1_acc, human_int2_acc])
human_err = np.array([human_int1_err, human_int2_err])

# Plot parameters
total_bar_width = 0.8
ind_bar_width = total_bar_width / 3
N_cond = 2
x_points = np.arange(N_cond)
human_color = 'powderblue'
gpt4_color = 'darkmagenta'
gpt4_analysis_color = 'mediumseagreen'
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14

# Combined results (combining across problem types)
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width), gpt4_acc, yerr=gpt4_err, color=gpt4_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points, gpt4_analysis_acc, yerr=gpt4_analysis_err, color=gpt4_analysis_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width), human_acc, yerr=human_err, color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, ['1', '2'], fontsize=plot_fontsize)
plt.xlabel('Interval size', fontsize=axis_label_fontsize)
plt.legend(['GPT-4', 'GPT-4 + code execution', 'Human'],fontsize=plot_fontsize,frameon=False)
hide_top_right(ax)
plt.tight_layout()
plt.savefig('./combined_results.pdf', dpi=300, bbox_inches="tight")
plt.close()

# Analyze by separate problem types
## GPT-4 + analysis
# Interval size = 1
gpt4_analysis_int1_probtype_acc = gpt4_analysis_int1_results['correct'] / gpt4_analysis_int1_results['N_per_probtype']
ci_lower, ci_upper = proportion_confint(gpt4_analysis_int1_results['correct'], gpt4_analysis_int1_results['N_per_probtype'])
gpt4_analysis_int1_probtype_err = np.stack([gpt4_analysis_int1_probtype_acc - ci_lower, ci_upper - gpt4_analysis_int1_probtype_acc])
# Interval size = 2
gpt4_analysis_int2_probtype_acc = gpt4_analysis_int2_results['correct'] / gpt4_analysis_int2_results['N_per_probtype']
ci_lower, ci_upper = proportion_confint(gpt4_analysis_int2_results['correct'], gpt4_analysis_int2_results['N_per_probtype'])
gpt4_analysis_int2_probtype_err = np.stack([gpt4_analysis_int2_probtype_acc - ci_lower, ci_upper - gpt4_analysis_int2_probtype_acc])

## Plot separately for different interval-size conditions
all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
N_cond = 6
x_points = np.arange(N_cond)
# Interval size = 1
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width), gpt4_int1_results['all_acc'], yerr=gpt4_int1_results['all_err'], color=gpt4_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points, gpt4_analysis_int1_probtype_acc, yerr=gpt4_analysis_int1_probtype_err, color=gpt4_analysis_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width), human_int1_results['all_acc'], yerr=human_int1_results['all_err'], color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.legend(['GPT-4', 'GPT-4 + code execution', 'Human'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.65,1))
plt.title('Interval size = 1', fontsize=title_fontsize)
hide_top_right(ax)
plt.savefig('./int1_results.pdf', dpi=300, bbox_inches="tight")
plt.close()
# Interval size = 2
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width), gpt4_int2_results['all_acc'], yerr=gpt4_int2_results['all_err'], color=gpt4_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points, gpt4_analysis_int2_probtype_acc, yerr=gpt4_analysis_int2_probtype_err, color=gpt4_analysis_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width), human_int2_results['all_acc'], yerr=human_int2_results['all_err'], color=human_color, edgecolor='black', width=ind_bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.legend(['GPT-4', 'GPT-4 + code execution', 'Human'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.65,1))
plt.title('Interval size = 2', fontsize=title_fontsize)
hide_top_right(ax)
plt.savefig('./int2_results.pdf', dpi=300, bbox_inches="tight")
plt.close()






