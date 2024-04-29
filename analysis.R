setwd("./")

# Human vs. GPT-4
data <-read.csv("./human_vs_GPT4.csv")
model <- glm(correct_pred ~ human_vs_gpt + intsize + human_vs_gpt:intsize, data=data, family="binomial")
summary(model)

# Human-only analysis
human_data <- subset(data, human_vs_gpt==0)
model <- glm(correct_pred ~ intsize, data=human_data, family="binomial")
summary(model)

# GPT-4-only analysis
gpt4_data <- subset(data, human_vs_gpt==1)
model <- glm(correct_pred ~ intsize, data=gpt4_data, family="binomial")
summary(model)

# Human vs. GPT-4 + code execution
data <-read.csv("./human_vs_GPT4_code_execution.csv")
model <- glm(correct_pred ~ human_vs_gpt + intsize + human_vs_gpt:intsize, data=data, family="binomial")
summary(model)

# GPT-4-only analysis
gpt4_data <- subset(data, human_vs_gpt==1)
model <- glm(correct_pred ~ intsize, data=gpt4_data, family="binomial")
summary(model)

# GPT-4 + code execution, original vs. alternative synthetic alphabets
data <-read.csv("./GPT4_code_execution_comparing_alphabets.csv")
model <- glm(correct_pred ~ alphabet_type, data=data, family="binomial")
summary(model)

# GPT-4, old vs. new engines
data <-read.csv("./GPT4_comparing_engines.csv")
model <- glm(correct_pred ~ engine_type, data=data, family="binomial")
summary(model)