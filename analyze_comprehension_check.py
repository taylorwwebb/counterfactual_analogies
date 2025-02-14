import numpy as np

# Load data
results = np.load('./counterfactual_comprehension_check_results.npz')
all_responses = results['all_responses']
all_int = results['all_int']

# Convert to integers
all_responses = np.array(all_responses).astype(int)
all_int = np.array(all_int).astype(int)

# Accuracy
acc = (all_responses == all_int).astype(float).mean()
neg1_acc = (all_responses[all_int==-1] == all_int[all_int==-1]).astype(float).mean()
neg2_acc = (all_responses[all_int==-2] == all_int[all_int==-2]).astype(float).mean()
pos1_acc = (all_responses[all_int==1] == all_int[all_int==1]).astype(float).mean()
pos2_acc = (all_responses[all_int==2] == all_int[all_int==2]).astype(float).mean()

# Present
print('Overall accuracy = ' + str(acc))
print('Accuracy for -1 = ' + str(neg1_acc))
print('Accuracy for -2 = ' + str(neg2_acc))
print('Accuracy for +1 = ' + str(pos1_acc))
print('Accuracy for +2 = ' + str(pos2_acc))