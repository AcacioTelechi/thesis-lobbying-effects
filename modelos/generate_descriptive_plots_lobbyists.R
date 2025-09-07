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

#===========================================
# 1) Estimate using the PPML from H1
#===========================================

df <- read.csv("./data/silver/df_meetings_lobbyists.csv", stringsAsFactors = TRUE)

# Group by lobbyist_id, count meetings, max "l_ln_max_budget"
# Remove -Inf values from l_ln_max_budget before aggregation
df$clean_ln_max_budget <- df$l_ln_max_budget
df$clean_ln_max_budget[!is.finite(df$clean_ln_max_budget)] <- NA

# Group by lobbyist_id and count meetings, get max budget ignoring -Inf/NA
df_grouped <- aggregate(
  x = list(meetings = df$lobbyist_id, max_budget = df$clean_ln_max_budget),
  by = list(lobbyist_id = df$lobbyist_id),
  FUN = function(x) {
    if (is.numeric(x)) {
      max(x, na.rm = TRUE)
    } else {
      length(x)
    }
  }
)
names(df_grouped)[names(df_grouped) == "x.meetings"] <- "meetings"
names(df_grouped)[names(df_grouped) == "x.max_budget"] <- "max_budget"

df_grouped <- df_grouped[df_grouped$meetings > 0, ]

df_grouped <- df_grouped[df_grouped$max_budget > 0, ]

# Check how many lobbyists have a max budget greater than 17
print(paste("Number of lobbyists with a max budget greater than 17:", nrow(df_grouped[df_grouped$max_budget > 17, ])))

# ---- plots

## Plot the histogram of max_budget
p1_max_budget <- ggplot(df_grouped, aes(x = max_budget)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "black") +
  labs(title = "Histogram of Max Budget", x = "Max Budget", y = "Frequency") +
  theme_minimal()

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "histogram_max_budget.png"),
  plot = p1_max_budget,
  width = 7, height = 5, dpi = 300
)


## Plot the histogram of meetings
p1_meetings <- ggplot(df_grouped, aes(x = meetings)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "black") +
  labs(title = "Histogram of Meetings", x = "Meetings", y = "Frequency") +
  theme_minimal()

# Save the plot to the thesis figures directory
ggsave(
  filename = file.path(figures_dir, "histogram_meetings.png"),
  plot = p1_meetings,
  width = 7, height = 5, dpi = 300
)

