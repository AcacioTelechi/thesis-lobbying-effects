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

model_nb_interaction <- glm.nb(
  meetings ~ l_category + l_ln_max_budget + l_category:l_ln_max_budget + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

# Print model summary
msummary(
  list("NB meetings model" = model_nb, "NB meetings model interaction" = model_nb_interaction), 
  stars = TRUE, 
  gof_omit = "AIC|BIC|Log.Lik.",
  coef_omit = "l_head_office"
)

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

# --- Center ln(max budget) for clean interpretation at average budget ---
mean_ln_budget <- mean(model_df$l_ln_max_budget, na.rm = TRUE)
model_df$l_ln_max_budget_c <- model_df$l_ln_max_budget - mean_ln_budget

model_nb_interaction_c <- glm.nb(
  meetings ~ l_category + l_ln_max_budget_c + l_category:l_ln_max_budget_c + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

# Export centered model table
msummary(
  list(
    "NB meetings model" = model_nb, 
    "NB meetings model interaction" = model_nb_interaction,
    "NB meetings model (centered)" = model_nb_interaction_c),
  stars = TRUE,
  gof_omit = "AIC|BIC|Log.Lik.",
  coef_omit = "l_head_office",
  # output = file.path(tables_dir, "tab_meetings_nb_centered.tex")
)

# Build grid for centered model predictions
budget_seq <- seq(
  quantile(model_df$l_ln_max_budget, 0.05, na.rm = TRUE),
  quantile(model_df$l_ln_max_budget, 0.95, na.rm = TRUE),
  length.out = 60
)
newdata_center <- expand.grid(
  l_category = droplevels(unique(model_df$l_category)),
  l_ln_max_budget_c = budget_seq - mean_ln_budget
)
# Set controls at means / modal for clean predictions
newdata_center$l_agriculture <- mean(model_df$l_agriculture, na.rm = TRUE)
newdata_center$l_economics_and_trade <- mean(model_df$l_economics_and_trade, na.rm = TRUE)
newdata_center$l_education <- mean(model_df$l_education, na.rm = TRUE)
newdata_center$l_environment_and_climate <- mean(model_df$l_environment_and_climate, na.rm = TRUE)
newdata_center$l_foreign_and_security_affairs <- mean(model_df$l_foreign_and_security_affairs, na.rm = TRUE)
newdata_center$l_health <- mean(model_df$l_health, na.rm = TRUE)
newdata_center$l_human_rights <- mean(model_df$l_human_rights, na.rm = TRUE)
newdata_center$l_infrastructure_and_industry <- mean(model_df$l_infrastructure_and_industry, na.rm = TRUE)
newdata_center$l_technology <- mean(model_df$l_technology, na.rm = TRUE)
mode_country <- names(which.max(table(model_df$l_head_office_country)))
newdata_center$l_head_office_country <- factor(mode_country, levels = levels(model_df$l_head_office_country))

# Predictions with standard errors on response scale
pred_center <- predict(model_nb_interaction_c, newdata = newdata_center, type = "link", se.fit = TRUE)
newdata_center$mean_meetings <- exp(pred_center$fit)
newdata_center$ci_lo <- exp(pred_center$fit - 1.96 * pred_center$se.fit)
newdata_center$ci_hi <- exp(pred_center$fit + 1.96 * pred_center$se.fit)
newdata_center$l_ln_max_budget <- newdata_center$l_ln_max_budget_c + mean_ln_budget

# Save CSV
fwrite(newdata_center, file.path(outputs_dir, "pred_meetings_centered_vs_budget_by_category.csv"))

# Plot expected meetings vs ln(budget) with CIs
p_meet_vs_budget_c <- ggplot(newdata_center, aes(x = l_ln_max_budget, y = mean_meetings, color = l_category, fill = l_category)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, color = NA) +
  geom_vline(xintercept = mean_ln_budget, linetype = "dashed", color = "gray60") +
  labs(x = "ln(orçamento máximo)", y = "Reuniões previstas", color = "Categoria", fill = "Categoria",
       subtitle = "Linha pontilhada: orçamento médio (escala log)") +
  theme_minimal()

ggsave(file.path(figures_dir, "fig_pred_meetings_vs_budget_centered_by_category.png"), p_meet_vs_budget_c, width = 9, height = 5.5, dpi = 300)

ggsave(file.path(figures_dir, "fig_pred_meetings_vs_budget_centered_by_category.pdf"), p_meet_vs_budget_c, width = 9, height = 5.5)

# Ratio curve: Business / NGOs expected meetings vs ln(budget)
lvls <- levels(model_df$l_category)
if (all(c("NGOs", "Business") %in% lvls)) {
  wide <- reshape(newdata_center[, c("l_category", "l_ln_max_budget", "mean_meetings", "ci_lo", "ci_hi")],
                  idvar = "l_ln_max_budget", timevar = "l_category", direction = "wide")
  wide$ratio_mean <- wide$mean_meetings.Business / wide$mean_meetings.NGOs
  # Delta method for CI on ratio (approx via log scale)
  # Here, we approximate using bounds: conservative envelope
  wide$ratio_lo <- (wide$ci_lo.Business / wide$ci_hi.NGOs)
  wide$ratio_hi <- (wide$ci_hi.Business / wide$ci_lo.NGOs)

  # Compute crossover lnB* where ratio = 1 using coefficients
  coefs <- coef(model_nb_interaction_c)
  b_biz <- unname(coefs["l_categoryBusiness"])
  b_int <- unname(coefs["l_categoryBusiness:l_ln_max_budget_c"])
  lnB_star_c <- if (!is.na(b_biz) && !is.na(b_int) && b_int != 0) -b_biz / b_int else NA_real_
  lnB_star <- lnB_star_c + mean_ln_budget

  ratio_df <- wide
  fwrite(ratio_df, file.path(outputs_dir, "ratio_business_over_ngo_vs_budget_centered.csv"))

  p_ratio <- ggplot(ratio_df, aes(x = l_ln_max_budget, y = ratio_mean)) +
    geom_hline(yintercept = 1, color = "gray70") +
    geom_line(color = "#9467bd", size = 1) +
    geom_ribbon(aes(ymin = ratio_lo, ymax = ratio_hi), fill = "#9467bd", alpha = 0.15) +
    { if (is.finite(lnB_star)) geom_vline(xintercept = lnB_star, linetype = "dotted", color = "#B45C1F") else NULL } +
    labs(x = "ln(orçamento máximo)", y = "Razão Business/NGOs (reuniões previstas)",
         subtitle = "Linha horizontal: paridade (1). Linha pontilhada: ponto de cruzamento estimado") +
    theme_minimal()

  ggsave(file.path(figures_dir, "fig_ratio_business_over_ngo_vs_budget_centered.png"), p_ratio, width = 9, height = 5.5, dpi = 300)
  ggsave(file.path(figures_dir, "fig_ratio_business_over_ngo_vs_budget_centered.pdf"), p_ratio, width = 9, height = 5.5)
}

