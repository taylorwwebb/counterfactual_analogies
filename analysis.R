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

# Human vs. GPT-4 + analysis
data <-read.csv("./human_vs_GPT4_analysis.csv")
model <- glm(correct_pred ~ human_vs_gpt + intsize + human_vs_gpt:intsize, data=data, family="binomial")
summary(model)

# GPT-4-only analysis
gpt4_data <- subset(data, human_vs_gpt==1)
model <- glm(correct_pred ~ intsize, data=gpt4_data, family="binomial")
summary(model)

