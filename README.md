# Evidence from counterfactual tasks supports emergent analogical reasoning in large language models

Code for the paper 'Evidence from counterfactual tasks supports emergent analogical reasoning in large language models'.

Counterfactual letter string analogy problem sets are included in ```all_prob_synthetic_int1.npz``` (for interval-size-1) and ```all_prob_synthetic_int2.npz``` (for interval-size-2).

To evaluate GPT-4 on these problems (without code execution), run the following command for interval-size-1:
```
python3 ./eval_GPT4_letterstring.py --interval_size 1
```
and the following command for interval-size-2:
```
python3 ./eval_GPT4_letterstring.py --interval_size 2
```

To analyze GPT-4's performance, run the following command for interval-size-1:
```
python3 ./analyze_GPT4_letterstring.py --interval_size 1
```
and the following command for interval-size-2:
```
python3 ./analyze_GPT4_letterstring.py --interval_size 2
```

Complete responses for experiments with GPT-4 + code execution can be found in ```./GPT4_analysis_responses/int1/``` and ```./GPT4_analysis_responses/int2/```.

Human behavioral data can be found in ```./behavioral_data_int1/``` and ```./behavioral_data_int2/```.

Data for the comparison between human behavior and GPT-4 can be found in ```./human_vs_GPT4.csv```. Data for the comparison between human behavior and GPT-4 + code execution can be found in ```./human_vs_GPT4_analysis.csv```.

To perform the analyses comparing human behavior, GPT-4, and GPT-4 + code execution, run the following command:
```
python3 ./combined_analysis.py
```
To perform statistical analyses, run the following R script:
```
./analysis.R
```

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
