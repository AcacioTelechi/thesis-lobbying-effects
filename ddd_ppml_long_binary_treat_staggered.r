# ============================================
# DDD FE + PPML (wui)
# ============================================


# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)

# Prefer fast, column-selective IO if available
if (requireNamespace("data.table", quietly = TRUE)) {
  fread <- data.table::fread
} else {
  fread <- NULL
}

# --- Load only the columns we need to save memory
# Required columns for current models: outcome, treatment-time indicator, FE and clustering fields, dates
needed_cols <- c(
  "questions", "post", "member_id", "domain_time",
  "time_fe", "first_treatment_period", "treated"
)

if (!is.null(fread)) {
  df <- fread("df_long.csv", select = needed_cols, showProgress = FALSE)
  df <- as.data.frame(df)
} else {
  # Fallback if data.table not installed
  df <- read.csv("df_long.csv", stringsAsFactors = FALSE)
  df <- df[, intersect(colnames(df), needed_cols)]
}

# Convert types compactly
df$member_id <- as.factor(df$member_id)
df$domain_time <- as.factor(df$domain_time)
df$questions <- as.numeric(df$questions)
df$post <- as.integer(df$post)
if ("treated" %in% names(df)) df$treated <- as.integer(df$treated)


# Fixed effects will use existing columns directly (avoid duplicating columns)

# 3) Date variables (overwrite in place to avoid extra copies)
df$time_fe <- as.Date(as.character(df$time_fe))
df$first_treatment_period <- as.Date(as.character(df$first_treatment_period))

# Helper logical for restricted cohort (no persistent copy)
keep_2020_2023 <- is.na(df$first_treatment_period) |
  (df$first_treatment_period >= as.Date("2020-01-01") &
     df$first_treatment_period < as.Date("2024-01-01"))

# 4) Build formulas (no-controls variants only to reduce memory)
# TWFE PPML with time-varying treatment indicator (post)
twfe_formula_no_controls <- as.formula("questions ~ post | member_id + domain_time")

# # Sun & Abraham event-study using cohort (first_treatment_period) and period (time_fe)
# sa_formula_no_controls <- as.formula("questions ~ sunab(first_treatment_period, time_fe) | member_id + domain_time")

# Event study with restricted leads and lags [-3,3]
sa_formula_restricted <- as.formula("questions ~ sunab(first_treatment_period, time_fe, ref.p = -1, ref.c = 0, bin.rel = -3:3) | member_id + domain_time")

# =============================
# Staggered DiD with PPML (fepois)
# =============================

# TWFE with post only
m_twfe_ppml_no_controls <- fepois(
  twfe_formula_no_controls,
  data    = df,
  cluster = ~domain_time
)

# # Sun & Abraham event-study
# m_sa_ppml_no_controls <- fepois(
#   sa_formula_no_controls,
#   data    = df,
#   cluster = ~domain_time
# )

# Sun & Abraham event-study with restricted leads/lags [-3,3]
m_sa_ppml_restricted <- fepois(
  sa_formula_restricted,
  data    = df,
  cluster = ~domain_time
)

m_sa_ppml_treated_after_2020_before_2024 <- fepois(
  sa_formula_no_controls,
  data    = df[keep_2020_2023, ],
  cluster = ~domain_time
)

# Restricted event study for the filtered sample
m_sa_ppml_restricted_filtered <- fepois(
  sa_formula_restricted,
  data    = df[keep_2020_2023, ],
  cluster = ~domain_time
)

# Per domain

results <- list(
  "TWFE PPML (no controls)" = m_twfe_ppml_no_controls,
  # "Sun-Abraham PPML (no controls)" = m_sa_ppml_no_controls,
  "Sun-Abraham PPML restricted [-3,3]" = m_sa_ppml_restricted,
  # "Sun-Abraham PPML (treated after 2020 and before 2024)" = m_sa_ppml_treated_after_2020_before_2024,
  "Sun-Abraham PPML restricted [-3,3] filtered" = m_sa_ppml_restricted_filtered
)

# domains <- unique(df$domain)
# for (domain in domains) {
#   df_domain <- df[df$domain == domain, ]
#   m_ddd_ppml_domain <- fepois(
#     full_formula_no_controls,
#     data    = df_domain,
#     cluster = ~cl_dt
#   )
#   results[[domain]] <- m_ddd_ppml_domain
# }

# Nice side-by-side table
modelsummary::msummary(results, stars = TRUE)

# Event study plot for the restricted model
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  
  # Extract event study coefficients
  event_coefs <- coef(m_sa_ppml_restricted)
  event_se <- sqrt(diag(vcov(m_sa_ppml_restricted)))
  
  # Create event study data frame
  event_data <- data.frame(
    period = c(-3, -2, -1, 0, 1, 2, 3),
    coefficient = event_coefs[grep("^first_treatment_period", names(event_coefs))],
    se = event_se[grep("^first_treatment_period", names(event_se))]
  )
  
  # Remove reference period (-1) and normalize to 0
  event_data <- event_data[event_data$period != -1, ]
  event_data$coefficient <- event_data$coefficient - event_data$coefficient[event_data$period == 0]
  
  # Create confidence intervals
  event_data$ci_lower <- event_data$coefficient - 1.96 * event_data$se
  event_data$ci_upper <- event_data$coefficient + 1.96 * event_data$se
  
  # Event study plot
  p <- ggplot(event_data, aes(x = period, y = coefficient)) +
    geom_point(size = 3, color = "blue") +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2, color = "blue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_vline(xintercept = -0.5, linetype = "dashed", color = "gray") +
    labs(
      title = "Event Study: Effect of Lobbying on Questions",
      subtitle = "Restricted to [-3,3] periods around treatment",
      x = "Periods relative to treatment",
      y = "Coefficient (relative to period -1)",
      caption = "Reference period: -1, 95% confidence intervals"
    ) +
    theme_minimal() +
    scale_x_continuous(breaks = c(-3, -2, -1, 0, 1, 2, 3))
  
  print(p)
  
  # Save the plot
  ggsave("Tese/figures/event_study_restricted.png", p, width = 10, height = 6, dpi = 300)
  ggsave("Tese/figures/event_study_restricted.pdf", p, width = 10, height = 6)
}

# Free memory
rm(keep_2020_2023)
gc()
