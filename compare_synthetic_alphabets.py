import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def hide_top_right(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# Load data
# GPT-4 + code execution w/ old synthetic alphabet
gpt4_code_execution_int1_results = np.load('./gpt-4-0125-preview_code_execution_int1/acc.npz')
gpt4_code_execution_int2_results = np.load('./gpt-4-0125-preview_code_execution_int2/acc.npz')
# GPT-4 + code execution w/ new synthetic alphabet
gpt4_code_execution_altalphabet_int1_results = np.load('./gpt-4-0125-preview_code_execution_int1_altalphabet/acc.npz')
gpt4_code_execution_altalphabet_int2_results = np.load('./gpt-4-0125-preview_code_execution_int2_altalphabet/acc.npz')

# Get accuracy for each condition
## GPT-4 + code execution w/ old synthetic alphabet
# Interval size = 1
gpt4_code_execution_int1_acc = gpt4_code_execution_int1_results['overall_acc'].item()
gpt4_code_execution_int1_err = gpt4_code_execution_int1_results['overall_err'][0]
# Interval size = 2
gpt4_code_execution_int2_acc = gpt4_code_execution_int2_results['overall_acc'].item()
gpt4_code_execution_int2_err = gpt4_code_execution_int1_results['overall_err'][0]
# Combined
gpt4_code_execution_acc = np.array([gpt4_code_execution_int1_acc, gpt4_code_execution_int2_acc])
gpt4_code_execution_err = np.array([gpt4_code_execution_int1_err, gpt4_code_execution_int2_err])
## GPT-4 + code execution w/ new synthetic alphabet
# Interval size = 1
gpt4_code_execution_altalphabet_int1_acc = gpt4_code_execution_altalphabet_int1_results['overall_acc'].item()
gpt4_code_execution_altalphabet_int1_err = gpt4_code_execution_altalphabet_int1_results['overall_err'][0]
# Interval size = 2
gpt4_code_execution_altalphabet_int2_acc = gpt4_code_execution_altalphabet_int2_results['overall_acc'].item()
gpt4_code_execution_altalphabet_int2_err = gpt4_code_execution_altalphabet_int1_results['overall_err'][0]
# Combined
gpt4_code_execution_altalphabet_acc = np.array([gpt4_code_execution_altalphabet_int1_acc, gpt4_code_execution_altalphabet_int2_acc])
gpt4_code_execution_altalphabet_err = np.array([gpt4_code_execution_altalphabet_int1_err, gpt4_code_execution_altalphabet_int2_err])

# Plot parameters
total_bar_width = 0.8
ind_bar_width = total_bar_width / 2
N_cond = 6
x_points = np.arange(N_cond)
gpt4_code_execution_color = 'mediumseagreen'
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14

## Plot separately for different interval-size conditions
all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
# Interval size = 1
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), gpt4_code_execution_int1_results['all_acc'], yerr=gpt4_code_execution_int1_results['all_err'], color=gpt4_code_execution_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), gpt4_code_execution_altalphabet_int1_results['all_acc'], yerr=gpt4_code_execution_altalphabet_int1_results['all_err'], color=gpt4_code_execution_color, edgecolor='black', width=ind_bar_width, ecolor='gray', hatch='//')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.legend(['GPT-4 + code execution', 'GPT-4 + code execution\nw/ new synthetic alphabet'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.65,1))
plt.title('Interval size = 1', fontsize=title_fontsize)
hide_top_right(ax)
plt.savefig('./int1_results_old_vs_new_alphabet.pdf', dpi=300, bbox_inches="tight")
plt.close()
# Interval size = 2
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width/2), gpt4_code_execution_int2_results['all_acc'], yerr=gpt4_code_execution_int2_results['all_err'], color=gpt4_code_execution_color, edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width/2), gpt4_code_execution_altalphabet_int2_results['all_acc'], yerr=gpt4_code_execution_altalphabet_int2_results['all_err'], color=gpt4_code_execution_color, edgecolor='black', width=ind_bar_width, ecolor='gray', hatch='//')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.legend(['GPT-4 + code execution', 'GPT-4 + code execution\nw/ new synthetic alphabet'],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.65,1))
plt.title('Interval size = 2', fontsize=title_fontsize)
hide_top_right(ax)
plt.savefig('./int2_results_old_vs_new_alphabet.pdf', dpi=300, bbox_inches="tight")
plt.close()






