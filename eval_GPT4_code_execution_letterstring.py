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

# Setup GPT-4
assistant = client.beta.assistants.create(
	model=args.gpt4_engine,
	tools=[{"type": "code_interpreter"}], 
)

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
prob_type_responses = []
prob_type_correct = []
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
	while not response_successful:
		thread = client.beta.threads.create()
		client.beta.threads.messages.create(
			thread_id=thread.id,
			role="user",
			content=prompt
		)
		run = client.beta.threads.runs.create_and_poll(
			thread_id=thread.id,
			assistant_id=assistant.id,
		)
		if run.status == 'completed':
			messages = client.beta.threads.messages.list(
				thread_id=thread.id
			)
			run_steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
			response = '\n'
			for t in range(len(run_steps.data)):
				if run_steps.data[(-1 - t)].type == 'message_creation':
					message_id = run_steps.data[(-1 - t)].step_details.message_creation.message_id
					for m in range(len(messages.data)):
						if messages.data[m].id == message_id:
							response += messages.data[m].content[0].text.value
							response += '\n\n'
				if run_steps.data[(-1 - t)].type == 'tool_calls':
					response += 'Code interpreter:\n'
					response += run_steps.data[(-1 - t)].step_details.tool_calls[0].code_interpreter.input
					response += '\n\nOutput:\n'
					response += run_steps.data[(-1 - t)].step_details.tool_calls[0].code_interpreter.outputs[0].logs
					response += '\n\n'
			# Present response and correct answer
			solution = ''.join(prob[1][1])
			print(response + 'Correct answer: ' + solution + '\n\n')
			# Determine accuracy
			correct = int(input('[-1] No answer given\n[0] Incorrect\n[1] Correct\nResponse: '))
			if correct >= 0:
				response_successful = True
		else:
			print('trying again...')
			time.sleep(5)
		if not response_successful:
			print('No answer given. Running again.\n\n')
	prob_type_responses.append(response)
	prob_type_correct.append(correct)

# Report overall performance
print('Final results: ' + str(np.array(prob_type_correct).sum()) + '/' + str(len(prob_type_correct)) + ' correct\n')

# Save
save_dir = './' + args.gpt4_engine + '_code_execution/'
check_path(save_dir)
save_fname = save_dir + 'int' + str(args.interval_size) + '_' + args.prob_type
if args.alt_alphabet:
	save_fname += '_altalphabet'
save_fname += '_results.npz'
np.savez(save_fname, prob_type_responses=prob_type_responses, prob_type_correct=prob_type_correct)
