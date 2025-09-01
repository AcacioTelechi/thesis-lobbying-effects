# install.packages("fixest") # if needed
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

df <- read.csv("./data/silver/df_meetings_lobbyists.csv", stringsAsFactors = TRUE)

# Group by lobbyist_id and member_id to count meetings
df_grouped <- aggregate(
  x = list(meetings = df$lobbyist_id),
  by = list(lobbyist_id = df$lobbyist_id),
  FUN = length
)

# Df for lobbyist data (get unique lobbyist characteristics)
df_lobbyists <- df[!duplicated(df$lobbyist_id), ]

# Merge meeting counts with lobbyist characteristics
df_merged <- merge(df_grouped, df_lobbyists, by = "lobbyist_id")

# Set 'NGOs' as the reference category to compare against
df_merged$l_category <- relevel(df_merged$l_category, ref = "NGOs")

# --- Clean predictors to avoid NA/NaN/Inf in model matrix ---
# Ensure numeric controls are numeric and finite
num_controls <- c(
  "l_ln_max_budget",
  "l_agriculture", "l_economics_and_trade", "l_education",
  "l_environment_and_climate", "l_foreign_and_security_affairs",
  "l_health", "l_human_rights", "l_infrastructure_and_industry", "l_technology"
)
for (nm in num_controls) {
  if (!nm %in% names(df_merged)) next
  df_merged[[nm]] <- suppressWarnings(as.numeric(df_merged[[nm]]))
  df_merged[[nm]][!is.finite(df_merged[[nm]])] <- NA
}
# Meetings should be integer and finite
df_merged$meetings <- as.integer(df_merged$meetings)

# Handle factor with missing values
df_merged$l_head_office_country <- as.character(df_merged$l_head_office_country)
df_merged$l_head_office_country[is.na(df_merged$l_head_office_country) | df_merged$l_head_office_country == ""] <- "Unknown"
df_merged$l_head_office_country <- as.factor(df_merged$l_head_office_country)

# Drop unused levels in l_category and ensure factor
df_merged$l_category <- droplevels(as.factor(df_merged$l_category))

# Build model frame and drop rows with any missing values in model vars
model_vars <- c("meetings", "l_category", num_controls, "l_head_office_country")
model_vars <- model_vars[model_vars %in% names(df_merged)]
rows_before <- nrow(df_merged)
model_df <- df_merged[complete.cases(df_merged[, model_vars, drop = FALSE]), ]
rows_after <- nrow(model_df)
cat(sprintf("Rows before cleaning: %d; after cleaning: %d (removed %d)\n", rows_before, rows_after, rows_before - rows_after))


# --- Model 1: Negative Binomial Regression to model meeting frequency ---
# This is preferred over lm for count data.
# The coefficient for 'l_categoryBusiness' will tell us the log-count difference
# in meetings compared to NGOs, holding other variables constant.
model_nb <- glm.nb(
  meetings ~ l_category + l_ln_max_budget + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

# Print model summary
summary(model_nb)
msummary(model_nb, stars = TRUE, gof_omit = "AIC|BIC|Log.Lik.")

# Export model summary to LaTeX
msummary(
  list("NB meetings model" = model_nb),
  stars = TRUE,
  gof_omit = "AIC|BIC|Log.Lik.",
  output = file.path("Tese", "tables", "h2_test", "tab_meetings_nb.tex")
)

# Quick descriptive check: average meetings by category
avg_meetings_by_cat <- aggregate(meetings ~ l_category, data = model_df, FUN = mean)
print(avg_meetings_by_cat)

# -------------------------------------------
# Effect plots
# -------------------------------------------

# Ensure output directories
figures_dir <- file.path("Tese", "figures", "h2_test")
tables_dir <- file.path("Tese", "tables", "h2_test")
outputs_dir <- file.path("outputs", "h2_test")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)

# --- Marginal predicted meetings by category (emmeans) ---
# Build 'at' as a named list of numeric scalars (global means of numeric covariates)
numeric_vars <- intersect(num_controls, names(model_df))
at_vals <- as.list(sapply(numeric_vars, function(nm) mean(model_df[[nm]], na.rm = TRUE)))
# Increase grid limit safely
emm_options(rg.limit = 10000)

rg <- ref_grid(
  model_nb,
  at = at_vals,
  nuisance = "l_head_office_country",
  cov.reduce = mean,
  weights = "proportional"
)
emm_meet <- emmeans(rg, ~ l_category, type = "response")
emm_df <- as.data.frame(emm_meet)
# Standardize column names
setDT(emm_df)
setnames(emm_df, old = c("response", "SE", "asymp.LCL", "asymp.UCL"), new = c("mean_meetings", "se", "ci_lo", "ci_hi"), skip_absent = TRUE)

# Save emmeans CSV
fwrite(emm_df, file = file.path(outputs_dir, "emmeans_meetings_by_category.csv"))

# Plot: Predicted meetings by category with 95% CI
p_pred_meet <- ggplot(emm_df, aes(x = mean_meetings, y = l_category)) +
  geom_vline(xintercept = 0, color = "gray85") +
  geom_point(color = "#1f77b4", size = 2.8) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, color = "#1f77b4") +
  labs(x = "Reuniões previstas (média marginal)", y = "Categoria") +
  theme_minimal()

ggsave(file.path(figures_dir, "fig_pred_meetings_by_category.png"), p_pred_meet, width = 8, height = 5.5, dpi = 300)

ggsave(file.path(figures_dir, "fig_pred_meetings_by_category.pdf"), p_pred_meet, width = 8, height = 5.5)

# --- Total effect calculation using PPML per-meeting coefficients ---
ppml_coeffs_df <- data.frame(
  l_category = factor(c("NGOs", "Business"), levels = levels(model_df$l_category)),
  per_meeting_coef = c(0.09, 0.025)
)

# Merge with emmeans
emm_with_coef <- merge(emm_df, ppml_coeffs_df, by = "l_category", all.x = TRUE)

# Compute total effect and CI by scaling emmean and CI by the per-meeting coefficient
emm_with_coef$total_effect <- emm_with_coef$mean_meetings * emm_with_coef$per_meeting_coef
emm_with_coef$total_effect_lo <- emm_with_coef$ci_lo * emm_with_coef$per_meeting_coef
emm_with_coef$total_effect_hi <- emm_with_coef$ci_hi * emm_with_coef$per_meeting_coef

# # Save total effect CSV
# fwrite(emm_with_coef[, c("l_category", "mean_meetings", "ci_lo", "ci_hi", "per_meeting_coef", "total_effect", "total_effect_lo", "total_effect_hi")],
#    file = file.path(outputs_dir, "total_effect_by_category.csv")
# )

# Plot: Total effect by category with 95% CI
p_total_effect <- ggplot(emm_with_coef, aes(x = total_effect, y = l_category)) +
  geom_vline(xintercept = 0, color = "gray85") +
  geom_point(color = "#d62728", size = 2.8) +
  geom_errorbarh(aes(xmin = total_effect_lo, xmax = total_effect_hi), height = 0.2, color = "#d62728") +
  labs(x = "Efeito total estimado (reuniões previstas × coeficiente PPML)", y = "Categoria") +
  theme_minimal()

ggsave(file.path(figures_dir, "fig_total_effect_by_category.png"), p_total_effect, width = 8, height = 5.5, dpi = 300)

ggsave(file.path(figures_dir, "fig_total_effect_by_category.pdf"), p_total_effect, width = 8, height = 5.5)

# # --- Ratios and printed summary ---
# # Ratio of predicted meetings
# if (all(c("NGOs", "Business") %in% emm_with_coef$l_category)) {
#   mean_NGO <- emm_with_coef$mean_meetings[emm_with_coef$l_category == "NGOs"]
#   mean_Biz <- emm_with_coef$mean_meetings[emm_with_coef$l_category == "Business"]
#   ratio_meet <- as.numeric(mean_Biz / mean_NGO)
#   # Ratio of total effects
#   te_NGO <- emm_with_coef$total_effect[emm_with_coef$l_category == "NGOs"]
#   te_Biz <- emm_with_coef$total_effect[emm_with_coef$l_category == "Business"]
#   ratio_te <- as.numeric(te_Biz / te_NGO)
#   cat(sprintf("\nRatio (Business vs NGOs) - Predicted meetings: %.2f\n", ratio_meet))
#   cat(sprintf("Ratio (Business vs NGOs) - Total effect: %.2f\n", ratio_te))
# }

