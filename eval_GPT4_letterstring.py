from openai import OpenAI
import numpy as np
import builtins
import argparse
import time

# Set up OpenAI client
client = OpenAI()

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt4_engine', type=str, default='gpt-4-0125-preview', help='GPT-4 engine')
args = parser.parse_args()

# Load all problems
if args.interval_size == 1:
	all_prob = np.load('./all_prob_synthetic_int1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
	all_prob = np.load('./all_prob_synthetic_int2.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())
prob_types = prob_types[:6]
N_prob_types = len(prob_types)

# Synthetic alphabet and prompt
alphabet = ['x', 'y', 'l', 'k', 'w', 'b', 'f', 'z', 't', 'n', 'j', 'r', 'q', 'a', 'h', 'v', 'g', 'm', 'u', 'o', 'p', 'd', 'i', 'c', 's', 'e']
alphabet = ' '.join(alphabet)
alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[x y l k w b f z t n j r q a h v g m u o p d i c s e]\n\nHere is the problem:\n\n"

# Evaluate
N_trials_per_prob_type = 50
all_prob_type_responses = []
all_prob_type_completions = []
for p in range(N_prob_types):
	print('problem type' + str(p+1) + ' of ' + str(N_prob_types) + '...')
	prob_type_responses = []
	prob_type_completions = []
	for t in range(N_trials_per_prob_type):
		print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
		# Generate prompt
		prob = all_prob.item()[prob_types[p]]['prob'][t]
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
		prompt += '] [ ? ]\n\n'
		prompt += 'Please only provide the answer. Do not provide any additional explanation.\n\n'
		prompt += 'Answer:'
		print(prompt)
		# Get response
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
		print(response)
		prob_type_responses.append(response)
		prob_type_completions.append(completion)
	all_prob_type_responses.append(prob_type_responses)
	all_prob_type_completions.append(prob_type_completions)
	# Save
	save_fname = './' + args.gpt4_engine + '_int' + str(args.interval_size)
	save_fname += '_results.npz'
	np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, all_prob_type_completions=all_prob_type_completions)
