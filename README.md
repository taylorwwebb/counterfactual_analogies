# Evidence from counterfactual tasks supports emergent analogical reasoning in large language models

Code for the paper [Evidence from counterfactual tasks supports emergent analogical reasoning in large language models](https://arxiv.org/abs/2404.13070v2).

Counterfactual letter string analogy problem sets are included in ```all_prob_synthetic_int1.npz``` (for interval-size-1) and ```all_prob_synthetic_int2.npz``` (for interval-size-2).

To evaluate GPT-4 on these problems (without code execution), run the following command for interval-size-1:
```
python3 ./eval_GPT4_letterstring.py --interval_size 1
```
and the following command for interval-size-2:
```
python3 ./eval_GPT4_letterstring.py --interval_size 2
```
To run experiments using the older GPT-4 engine, include the argument ```--gpt4_engine gpt-4-1106-preview```.

To analyze GPT-4's performance, run the following command for interval-size-1 (again specifying the older GPT-4 engine if desired):
```
python3 ./analyze_GPT4_letterstring.py --interval_size 1
```
and the following command for interval-size-2:
```
python3 ./analyze_GPT4_letterstring.py --interval_size 2
```

To evaluate GPT-4 + code execution on these problems, run the following command, specifying the interval size and problem type:
```
python3 ./eval_GPT4_code_execution_letterstring.py --interval_size 1 --prob_type succ
```
The full set of problem types is ```['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort']```. To run experiments using the alternative synthetic alphabet, include the argument ```--alt_alphabet```. This evaluation script is interactive. Each response from GPT-4 is presented along with the correct answer, and the user is prompted to specify whether the answer is correct (by entering 1), incorrect (by entering 0), or did not provide an answer (by entering -1).

To analyze the performance of GPT-4 + code execution, run the following command (specifying the interval size, and whether the alternative synthetic alphabet was used):
```
python3 ./analyze_GPT4_code_execution_letterstring.py
```

To analyze the errors made by GPT-4 + code execution, run the following command:
```
python3 ./analyze_error_types_GPT4_code_execution.py
```
The full response and correct answer for each error problem will be presented, and the user is prompted to indicate whether the answer is based on a valid alternative rule (by entering 1) or simply wrong (by entering 0). To summarize the results of this analysis, run the following command:
```
python3 ./display_error_types_GPT4_code_execution.py
```

To run the analysis comparing the performance of GPT-4 + code execution on the original vs. alternative synthetic alphabets, run the following command:
```
python3 ./compare_synthetic_alphabets.py
```

To run the analysis comparing the performance of GPT-4 with the old (1106) vs. new (0125) engines, run the following command:
```
python3 ./compare_GPT4_engines.py
```

To create CSV files for statistical analyses, run the following command:
```
python3 ./create_csv.py
```

To perform statistical analyses, run the following R script:
```
./analysis.R
```

All data for the results presented in the paper (including both human behavioral data and results for the evaluations of GPT-4 and GPT-4 + code execution) are included in this repository.

## Prerequisites

- Python 3
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [R](https://www.r-project.org/)


## Authorship

All code was written by [Taylor Webb](https://github.com/taylorwwebb). 
