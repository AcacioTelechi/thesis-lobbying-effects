library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(gridExtra)

# Set figure directory
fig_dir <- "./Tese/figures/descriptive_treatment"
tables_dir <- "./Tese/tables/descriptive_treatment"

# Create directory if it doesn't exist
if (!dir.exists(fig_dir)) {
    dir.create(fig_dir, recursive = TRUE)
}
if (!dir.exists(tables_dir)) {
    dir.create(tables_dir, recursive = TRUE)
}

# Load data
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# Variables
domain_cols <- c(
  "l_agriculture",
  "l_economics_and_trade",
  "l_education",
  "l_environment_and_climate",
  "l_foreign_and_security_affairs",
  "l_health",
  "l_human_rights",
  "l_infrastructure_and_industry",
  "l_technology"
)

rename_domains <- c(
  "agriculture" = "Agricultura",
  "economics_and_trade" = "Economia e Comércio",
  "education" = "Educação",
  "environment_and_climate" = "Meio Ambiente e Clima",
  "foreign_and_security_affairs" = "Assuntos Externos e  Segurança",
  "health" = "Saúde",
  "human_rights" = "Direitos Humanos",
  "infrastructure_and_industry" = "Infraestrutura e  Indústria",
  "technology" = "Tecnologia"
)

df$domain <- rename_domains[df$domain]
df$domain <- factor(df$domain, levels = rename_domains)

# ============================================
# 1) Taxa de tratamento por domínio
# ============================================

# Group by lobbyist_id, domain, sum questions and meetings
df_domain <- df %>%
    group_by(member_id, domain) %>%
    summarise(
        questions = sum(questions, na.rm = TRUE),
        meetings = sum(meetings, na.rm = TRUE)
    ) %>%
    ungroup()

# Table with number of members by domain, number of members
# treated by domain, and number of members not treated by domain
df_domain_summary <- df_domain %>%
    group_by(domain) %>%
    summarise(
        members = n_distinct(member_id),
        members_treated = n_distinct(member_id[meetings > 0]),
        total_meetings = sum(meetings, na.rm = TRUE),
        total_questions = sum(questions, na.rm = TRUE)
    ) %>%
    ungroup()

df_domain_summary$treatment_rate <- df_domain_summary$members_treated / df_domain_summary$members
df_domain_summary$intensity <- df_domain_summary$total_meetings / df_domain_summary$members_treated

# Order by treatment rate desc
df_domain_summary <- df_domain_summary[order(-df_domain_summary$treatment_rate),]


# Create a copy for latex formatting
df_domain_summary_tex <- df_domain_summary
df_domain_summary_tex$treatment_rate <- sprintf("%.2f", df_domain_summary_tex$treatment_rate * 100)
df_domain_summary_tex$intensity <- sprintf("%.2f", df_domain_summary_tex$intensity)


# Rename columns
colnames(df_domain_summary) <- c("Domínio", "MEPs", "MEPs Tratados", "Total de Reuniões", "Total de Perguntas", "Taxa de Tratamento (%)", "Intensidade Média de Tratamento")

# Save the table to the thesis tables directory
write.csv(df_domain_summary, file.path(tables_dir, "tab_domain_summary.csv"), row.names = FALSE)
write.table(df_domain_summary_tex,
  file.path(tables_dir, "tab_domain_summary.tex"),
  row.names = FALSE,
  col.names = FALSE,
  sep = " & ",
  quote = FALSE,
  eol = " \\\\\n",
  dec = ","
)

# ============================================
# 2) Zero inflation
# ============================================

df_id <- df %>%
    group_by(member_id, domain) %>%
    summarise(
        questions = sum(questions, na.rm = TRUE),
        meetings = sum(meetings, na.rm = TRUE)
    ) %>%
    ungroup()

df_dt <- df %>%
    group_by(domain, Y.m) %>%
    summarise(
        questions = sum(questions, na.rm = TRUE),
        meetings = sum(meetings, na.rm = TRUE)
    ) %>%
    ungroup()

zeros_idt <- count(df, questions == 0, meetings == 0)
zeros_id <- count(df_id, questions == 0, meetings == 0)
zeros_dt <- count(df_dt, questions == 0, meetings == 0)

# Pint results of the proportions of zeros in questiosn and meetings by idt, id, and dt
# idt
zero_questions_idt <- (zeros_idt$n[3] + zeros_idt$n[4]) / nrow(df)
zero_meetings_idt <- (zeros_idt$n[2] + zeros_idt$n[4]) / nrow(df)
# id
zero_questions_id <- (zeros_id$n[3] + zeros_id$n[4]) / nrow(df_id)
zero_meetings_id <- (zeros_id$n[2] + zeros_id$n[4]) / nrow(df_id)
# dt
zero_questions_dt <- (zeros_dt$n[2])/nrow(df_dt)
zero_meetings_dt <- 0 /nrow(df_dt)

# create a dataframe with the results
df_zeros <- data.frame(
    level = c("idt", "id", "dt"),
    questions = c(zero_questions_idt, zero_questions_id, zero_questions_dt),
    meetings = c(zero_meetings_idt, zero_meetings_id, zero_meetings_dt)
)

# print the dataframe
print(df_zeros)

# save the dataframe to a csv file