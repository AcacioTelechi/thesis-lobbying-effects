# install.packages("fixest") # if needed
# install.# install.packages("fixest") # if needed
# install.packages("MASS") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)
library(MASS) # For glm.nb

# Additional packages for marginal means and data export
# install.packages("emmeans") # if needed
# install.packages("data.table") # if needed
library(emmeans)
library(data.table)


# Ensure output directories
figures_dir <- file.path("Tese", "figures", "descriptives_lobbyists")
tables_dir <- file.path("Tese", "tables", "descriptives_lobbyists")
outputs_dir <- file.path("outputs", "descriptives_lobbyists")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)

df <- read.csv("./data/silver/df_meetings_lobbyists.csv", stringsAsFactors = TRUE)

# ===========================================
# 0) Descriptive table (total lobbyists, total meetings, total meetings/lobbyist)
# ===========================================

df_descriptive <- data.frame(
  category = levels(df$l_category)
)
df_descriptive$Lobistas <- tapply(df$lobbyist_id, df$l_category, function(x) length(unique(x)))
df_descriptive$Reuniões <- tapply(df$meeting_date, df$l_category, length)
df_descriptive <- na.omit(df_descriptive)

df_descriptive$Reuniões_por_lobista <- df_descriptive$Reuniões / df_descriptive$Lobistas

# Export df_descriptive to csv
write.csv(df_descriptive, file.path(tables_dir, "tab_descriptive_lobbyists.csv"), row.names = FALSE)


# ===========================================
# 1) Budget
# ===========================================


# Group by lobbyist_id, count meetings, max "l_ln_max_budget"
# Remove -Inf values from l_ln_max_budget before aggregation
df$clean_ln_max_budget <- df$l_ln_max_budget
df$clean_ln_max_budget[!is.finite(df$clean_ln_max_budget)] <- NA

# Group by lobbyist_id and count meetings, get max budget ignoring -Inf/NA
df_grouped <- aggregate(
  x = list(max_budget = df$clean_ln_max_budget, category = df$l_category),
  by = list(lobbyist_id = df$lobbyist_id),
  FUN = function(x) {
    if (is.numeric(x)) {
      max(x, na.rm = TRUE)
    } else if (is.factor(x)) {
      as.character(x[1])
    } else {
      length(x)
    }
  }
)

names(df_grouped)[names(df_grouped) == "x.max_budget"] <- "max_budget"

df_grouped <- df_grouped[df_grouped$max_budget > 0, ]

# Check how many lobbyists have a max budget greater than 17
print(paste("Number of lobbyists with a max budget greater than 17:", nrow(df_grouped[df_grouped$max_budget > 17, ])))


## Plot the histogram of max_budget
p1_max_budget <- ggplot(df_grouped, aes(x = category, y = max_budget)) +
  geom_boxplot(fill = "steelblue", color = "black") +
  labs(x = "Categoria", y = "ln(Orçamento máximo)") +
  theme_minimal()

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "boxplot_max_budget_by_category.png"),
  plot = p1_max_budget,
  width = 7, height = 5, dpi = 300
)

# ===========================================
# 2) Country
# ===========================================

# Group by lobbyist_id, count meetings, max "l_head_office_country"
df_grouped_country <- aggregate(
  x = list(country = df$l_head_office_country),
  by = list(lobbyist_id = df$lobbyist_id),
  FUN = function(x) {
    if (is.factor(x)) {
      as.character(x[1])
    }
  }
)

eu_countries <- c(
  "AUSTRIA",
  "BELGIUM",
  "BULGARIA",
  "CROATIA",
  "CYPRUS",
  "CZECH REPUBLIC",
  "DENMARK",
  "ESTONIA",
  "FRANCE",
  "GERMANY",
  "GREECE",
  "HUNGARY",
  "IRELAND",
  "ITALY",
  "LATVIA",
  "LITHUANIA",
  "LUXEMBOURG",
  "MALTA",
  "NETHERLANDS",
  "POLAND",
  "PORTUGAL",
  "ROMANIA",
  "SLOVENIA",
  "SLOVAKIA",
  "SPAIN",
  "SWEDEN"
)

df_grouped_country$is_eu <- df_grouped_country$country %in% eu_countries

# Plot the country distribution
# Order countries by descending frequency
country_counts <- as.data.frame(table(df_grouped_country$country))
colnames(country_counts) <- c("country", "freq")
df_country_counts <- merge(country_counts, unique(df_grouped_country[, c("country", "is_eu")]), by = "country", how = "left")
df_country_counts$percentage <- df_country_counts$freq / sum(df_country_counts$freq)

# Order countries by descending frequency
order_countries <- country_counts$country[order(-country_counts$freq)]


# Filter to top 20 countries
df_country_counts_top20 <- df_country_counts[df_country_counts$country %in% order_countries[1:20], ]

# order countries by descending frequency
df_country_counts_top20 <- df_country_counts_top20[order(-df_country_counts_top20$freq), ]
# Ensure the plot uses the correct order by setting domain as a factor
df_country_counts_top20$country <- factor(
  df_country_counts_top20$country,
  levels = df_country_counts_top20$country
)



total_organizations <- sum(df_country_counts$freq)
total_organizations_top20 <- sum(df_country_counts_top20$freq)
total_organizations_in_eu <- sum(df_country_counts$freq[df_country_counts$is_eu])
percentage_organizations_top20 <- total_organizations_top20 / total_organizations
percentage_organizations_in_eu <- total_organizations_in_eu / total_organizations


print(paste("Total organizations:", total_organizations))
print(paste("Total organizations in top 20:", total_organizations_top20))
print(paste("Percentage of organizations in top 20:", percentage_organizations_top20))

print(paste("Total organizations in EU:", total_organizations_in_eu))
print(paste("Percentage of organizations in EU:", percentage_organizations_in_eu))



p2_country_distribution <- ggplot(df_country_counts_top20, aes(x = country, y = freq, fill = is_eu)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_manual(
    values = c("TRUE" = "#B45C1F", "FALSE" = "#1f77b4"),
    labels = c("TRUE" = "Sim", "FALSE" = "Não"),
    name = "Membro da UE",
  ) +
  labs(x = "País", y = "Frequência") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "barplot_country_distribution.png"),
  plot = p2_country_distribution,
  width = 7, height = 5, dpi = 300
)


# ===========================================
# 3) Domain
# ===========================================

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

# Count how many lobbyists are in each domain
domain_lobbyist_counts <- sapply(domain_cols, function(col) {
  length(unique(df$lobbyist_id[df[[col]] == 1]))
})


rename_domains <- c(
  "l_agriculture" = "Agricultura",
  "l_economics_and_trade" = "Economia e Comércio",
  "l_education" = "Educação",
  "l_environment_and_climate" = "Meio Ambiente e Clima",
  "l_foreign_and_security_affairs" = "Política Externa e Segurança",
  "l_health" = "Saúde",
  "l_human_rights" = "Direitos Humanos",
  "l_infrastructure_and_industry" = "Infraestrutura e Indústria",
  "l_technology" = "Tecnologia"
)

domain_lobbyist_counts <- data.frame(
  domain = names(rename_domains)[domain_cols],
  lobbyist_count = domain_lobbyist_counts[domain_cols]
)

domain_lobbyist_counts$domain <- rename_domains[rownames(domain_lobbyist_counts)]

domain_lobbyist_counts$percentage <- domain_lobbyist_counts$lobbyist_count / sum(domain_lobbyist_counts$lobbyist_count)

# Order descending by lobbyist_count
domain_lobbyist_counts <- domain_lobbyist_counts[order(-domain_lobbyist_counts$lobbyist_count), ]

# Ensure the plot uses the correct order by setting domain as a factor
domain_lobbyist_counts$domain <- factor(
  domain_lobbyist_counts$domain,
  levels = domain_lobbyist_counts$domain
)

# Plot the domain distribution
p3_domain_distribution <- ggplot(domain_lobbyist_counts, aes(x = domain, y = lobbyist_count)) +
  geom_bar(stat = "identity", fill = "#1f77b4", color = "black") +
  labs(x = "Domínio", y = "Frequência") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = paste0(round(percentage * 100, 1), "%")), vjust = -0.5)

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "barplot_domain_distribution.png"),
  plot = p3_domain_distribution,
  width = 7, height = 5, dpi = 300
)


# ===========================================
# 4) Number of themes per lobbyist
# ===========================================

df_themes_per_lobbyist <- df

df_themes_per_lobbyist$number_of_themes <- rowSums(df_themes_per_lobbyist[, domain_cols])

df_themes_per_lobbyist <- aggregate(number_of_themes ~ lobbyist_id, data = df_themes_per_lobbyist, max)

mean_themes <- mean(df_themes_per_lobbyist$number_of_themes)

# Plot the number of themes per lobbyist
p4_themes_per_lobbyist <- ggplot(df_themes_per_lobbyist, aes(x = number_of_themes)) +
  geom_histogram(binwidth = 1, fill = "#1f77b4", color = "black") +
  geom_vline(xintercept = mean_themes, color = "#B45C1F", linetype = "dashed") +
  annotate(
    "text",
    x = mean_themes,
    y = 800,
    label = paste0("Média: ", round(mean_themes, 1)),
    vjust = 1, hjust = -0.1,
    color = "#B45C1F",
    fontface = "bold"
  ) +
  labs(x = "Número de temas", y = "Frequência") +
  theme_minimal()

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "histogram_themes_per_lobbyist.png"),
  plot = p4_themes_per_lobbyist,
  width = 7, height = 5, dpi = 300
)
