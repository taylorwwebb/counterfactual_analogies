setwd("./")

# Human vs. GPT-4
data <-read.csv("./human_vs_GPT4.csv")
model <- glm(correct_pred ~ human_vs_gpt, data=data, family="binomial")
summary(model)

# Human vs. GPT-4 + chain-of-thought
data <-read.csv("./human_vs_GPT4_COT.csv")
model <- glm(correct_pred ~ human_vs_gpt, data=data, family="binomial")
summary(model)

# GPT-4 vs. GPT-4 + chain-of-thought
data <-read.csv("./GPT4_vs_GPT4_COT.csv")
model <- glm(correct_pred ~ gpt_vs_cot, data=data, family="binomial")
summary(model)

# GPT-4 + chain-of-thought vs. GPT-4 + code execution
data <-read.csv("./GPT4_code_execution_vs_GPT4_COT.csv")
model <- glm(correct_pred ~ code_vs_cot, data=data, family="binomial")
summary(model)

# Human vs. GPT-4 + code execution
data <-read.csv("./human_vs_GPT4_code_execution.csv")
model <- glm(correct_pred ~ human_vs_gpt, data=data, family="binomial")
summary(model)

# GPT-4 vs. code execution control model
data <-read.csv("./GPT4_vs_code_execution_control.csv")
model <- glm(correct_pred ~ gpt_vs_control, data=data, family="binomial")
summary(model)



