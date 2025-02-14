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
gpt4_engine = 'gpt-4-0125-preview'

# Alphabet
alphabet = ['x', 'y', 'l', 'k', 'w', 'b', 'f', 'z', 't', 'n', 'j', 'r', 'q', 'a', 'h', 'v', 'g', 'm', 'u', 'o', 'p', 'd', 'i', 'c', 's', 'e']
alphabet_prompt = "Consider the following fictional alphabet:\n\n[x y l k w b f z t n j r q a h v g m u o p d i c s e]\n\n"

# Positive intervals
N_tests = 100
all_responses = []
all_int = []
for i in range(2):
	for s in range(N_tests):
		# Select letters
		letter_ind = np.arange(len(alphabet)-(i+1))
		np.random.shuffle(letter_ind)
		ind1 = letter_ind[0]
		ind2 = ind1 + (i+1)
		letter1 = alphabet[ind1]
		letter2 = alphabet[ind2]
		# Generate prompt
		prompt = ''
		prompt += alphabet_prompt
		prompt += 'In this alphabet, what is the interval (distance and direction) from ' + letter1 + ' to ' + letter2 + '?\n\n'
		prompt += 'Please respond only with +/- followed by an integer (e.g. +1, -1, +2, -2, etc.).\n\nAnswer:'
		print(prompt)
		# Get response
		response = []
		while len(response) == 0:
			try:
				completion = client.chat.completions.create(
								  model=gpt4_engine,
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
		# Correct answer
		correct_answer = '+' + str(i+1)
		print('Correct answer: ' + correct_answer)
		# Collect responses
		all_responses.append(response)
		all_int.append(correct_answer)

# Negative intervals
for i in range(2):
	for s in range(N_tests):
		# Select letters
		letter_ind = np.arange((i+1),len(alphabet))
		np.random.shuffle(letter_ind)
		ind1 = letter_ind[0]
		ind2 = ind1 - (i+1)
		letter1 = alphabet[ind1]
		letter2 = alphabet[ind2]
		# Generate prompt
		prompt = ''
		prompt += alphabet_prompt
		prompt += 'In this alphabet, what is the interval (distance and direction) from ' + letter1 + ' to ' + letter2 + '?\n\n'
		prompt += 'Please respond only with +/- followed by an integer (e.g. +1, -1, +2, -2, etc.).\n\nAnswer:'
		print(prompt)
		# Get response
		response = []
		while len(response) == 0:
			try:
				completion = client.chat.completions.create(
								  model=gpt4_engine,
								  messages=[
								    {"role": "user", "content": prompt}
								  ]
								)
				response = completion.choices[0].message.content
			except:
				print('trying again...')
				time.sleep(5)
		print(response)
		# Correct answer
		correct_answer = '-' + str(i+1)
		print('Correct answer: ' + correct_answer)
		# Collect responses
		all_responses.append(response)
		all_int.append(correct_answer)

# Save data
np.savez('./counterfactual_comprehension_check_results.npz', all_responses=all_responses, all_int=all_int)

