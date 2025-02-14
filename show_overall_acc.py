import numpy as np
from statsmodels.stats.proportion import proportion_confint

# GPT-4
gpt4_int1_results = np.load('./gpt-4-0125-preview_int1/acc.npz')
gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
gpt4_overall_acc = np.mean([gpt4_int1_results['overall_acc'],gpt4_int2_results['overall_acc']])
N_correct = np.round(gpt4_overall_acc * 600)
ci_lower, _ = proportion_confint(N_correct, 600)
gpt4_ci = gpt4_overall_acc - ci_lower
print('GPT-4, accuracy = ' + str(gpt4_overall_acc) + ', CI = ' + str(gpt4_ci))

# GPT-4 + CoT
gpt4_cot_int1_results = np.load('./gpt-4-0125-preview_COT_int1/acc.npz')
gpt4_cot_int2_results = np.load('./gpt-4-0125-preview_COT_int2/acc.npz')
gpt4_cot_overall_acc = np.mean([gpt4_cot_int1_results['overall_acc'],gpt4_cot_int2_results['overall_acc']])
N_correct = np.round(gpt4_cot_overall_acc * 120)
ci_lower, _ = proportion_confint(N_correct, 120)
gpt4_cot_ci = gpt4_cot_overall_acc - ci_lower
print('GPT-4 + chain-of-thought, accuracy = ' + str(gpt4_cot_overall_acc) + ', CI = ' + str(gpt4_cot_ci))

# GPT-4 + code execution
gpt4_CE_int1_results = np.load('./gpt-4-0125-preview_code_execution_int1/acc.npz')
gpt4_CE_int2_results = np.load('./gpt-4-0125-preview_code_execution_int2/acc.npz')
gpt4_CE_overall_acc = np.mean([gpt4_CE_int1_results['overall_acc'],gpt4_CE_int2_results['overall_acc']])
N_correct = np.round(gpt4_CE_overall_acc * 120)
ci_lower, _ = proportion_confint(N_correct, 120)
gpt4_CE_ci = gpt4_CE_overall_acc - ci_lower
print('GPT-4 + code execution, accuracy = ' + str(gpt4_CE_overall_acc) + ', CI = ' + str(gpt4_CE_ci))

# Code execution control model
gpt4_CEcontrol_int1_results = np.load('./gpt-4-0125-preview_code_execution_control_int1/acc.npz')
gpt4_CEcontrol_int2_results = np.load('./gpt-4-0125-preview_code_execution_control_int2/acc.npz')
gpt4_CEcontrol_overall_acc = np.mean([gpt4_CEcontrol_int1_results['overall_acc'],gpt4_CEcontrol_int2_results['overall_acc']])
N_correct = np.round(gpt4_CEcontrol_overall_acc * 600)
ci_lower, _ = proportion_confint(N_correct, 600)
gpt4_CEcontrol_ci = gpt4_CEcontrol_overall_acc - ci_lower
print('code execution control model, accuracy = ' + str(gpt4_CEcontrol_overall_acc) + ', CI = ' + str(gpt4_CEcontrol_ci))
