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
