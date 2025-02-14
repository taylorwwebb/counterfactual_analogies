from openai import OpenAI
import numpy as np
import builtins
import argparse
import os
import time

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

# Set up OpenAI client
client = OpenAI()

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--prob_type', type=str, default='succ', help="Problem types: ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']")
parser.add_argument('--gpt4_engine', type=str, default='gpt-4-0125-preview', help='GPT-4 engine')
parser.add_argument('--alt_alphabet', action='store_true', help="Alternative synthetic alphabet.")
args = parser.parse_args()

# Load all problems
if args.interval_size == 1:
	all_prob = np.load('./all_prob_synthetic_int1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
	all_prob = np.load('./all_prob_synthetic_int2.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())
prob_types = prob_types[:6]
prob_ind = np.where(np.array(prob_types) == args.prob_type)[0][0]

# Synthetic alphabet and prompt
if args.alt_alphabet:
	orig_alphabet = ['x', 'y', 'l', 'k', 'w', 'b', 'f', 'z', 't', 'n', 'j', 'r', 'q', 'a', 'h', 'v', 'g', 'm', 'u', 'o', 'p', 'd', 'i', 'c', 's', 'e']
	alt_alphabet = ['n', 'h', 'v', 'b', 'o', 'p', 'y', 'z', 't', 'm', 'r', 'w', 'x', 'f', 'i', 'q', 'd', 'j', 'l', 'c', 'a', 's', 'k', 'g', 'e', 'u']
	alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[n h v b o p y z t m r w x f i q d j l c a s k g e u]\n\nHere is the problem:\n\n"
else:
	alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[x y l k w b f z t n j r q a h v g m u o p d i c s e]\n\nHere is the problem:\n\n"

# Evaluate
N_trials_per_prob_type = 10
N_max_attempts = 5
prob_type_responses = []
prob_type_correct = []
prob_type_N_attempts = []
for t in range(N_trials_per_prob_type):
	print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...\n')
	# Problem
	prob = all_prob.item()[prob_types[prob_ind]]['prob'][t]
	# Convert to alternative alphabet
	if args.alt_alphabet:
		new_prob = []
		for i in range(len(prob)):
			new_prob_row = []
			for j in range(len(prob[i])):
				new_prob_cell = []
				for k in range(len(prob[i][j])):
					new_prob_cell.append(alt_alphabet[orig_alphabet.index(prob[i][j][k])])
				new_prob_row.append(new_prob_cell)
			new_prob.append(new_prob_row)
		prob = new_prob
	# Prompt
	prompt = ''
	prompt += alphabet_prompt
	prompt += '['
	for i in range(len(prob[0][0])):
		prompt += str(prob[0][0][i])
		if i < len(prob[0][0]) - 1:
			prompt += ' '
	prompt += '] ['
	for i in range(len(prob[0][1])):
		prompt += str(prob[0][1][i])
		if i < len(prob[0][1]) - 1:
			prompt += ' '
	prompt += ']\n['
	for i in range(len(prob[1][0])):
		prompt += str(prob[1][0][i])
		if i < len(prob[1][0]) - 1:
			prompt += ' '
	prompt += '] [ ? ]'
	print(prompt)
	# Get response
	response_successful = False
	N_attempts = 0
	while not response_successful and N_attempts < N_max_attempts:
		response = []
		while len(response) == 0:
			try:
				completion = client.chat.completions.create(
								  model=args.gpt4_engine,
								  temperature=0,
								  top_p=0,
								  messages=[
								    {"role": "user", "content": prompt}
								  ]
								)
				response = completion.choices[0].message.content
			except:
				print('trying again...')
				time.sleep(5)
		N_attempts += 1
		# Present response and correct answer
		solution = ''.join(prob[1][1])
		print(response + '\n\nCorrect answer: ' + solution + '\n\n')
		# Determine accuracy
		correct = int(input('[-1] No answer given\n[0] Incorrect\n[1] Correct\nResponse: '))
		if correct >= 0:
			response_successful = True
		if not response_successful:
			if N_attempts >= N_max_attempts:
				correct = 0
				print('No answer given. Maximum number of attempts reached.\n\n')
			else:
				print('No answer given. Running again.\n\n')
	prob_type_responses.append(response)
	prob_type_correct.append(correct)
	prob_type_N_attempts.append(N_attempts)

# Report overall performance
print('Final results: ' + str(np.array(prob_type_correct).sum()) + '/' + str(len(prob_type_correct)) + ' correct\n')

# Save
save_dir = './' + args.gpt4_engine + '_COT/'
check_path(save_dir)
save_fname = save_dir + 'int' + str(args.interval_size) + '_' + args.prob_type
if args.alt_alphabet:
	save_fname += '_altalphabet'
save_fname += '_results.npz'
np.savez(save_fname, prob_type_responses=prob_type_responses, prob_type_correct=prob_type_correct, prob_type_N_attempts=prob_type_N_attempts)
