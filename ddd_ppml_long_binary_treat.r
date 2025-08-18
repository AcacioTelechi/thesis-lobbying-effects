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

# Sun & Abraham event-study using cohort (first_treatment_period) and period (time_fe)
sa_formula_no_controls <- as.formula("questions ~ sunab(first_treatment_period, time_fe) | member_id + domain_time")

# =============================
# Staggered DiD with PPML (fepois)
# =============================

# TWFE with post only
m_twfe_ppml_no_controls <- fepois(
  twfe_formula_no_controls,
  data    = df,
  cluster = ~domain_time
)

# Sun & Abraham event-study
m_sa_ppml_no_controls <- fepois(
  sa_formula_no_controls,
  data    = df,
  cluster = ~domain_time
)

m_sa_ppml_treated_after_2020_before_2024 <- fepois(
  sa_formula_no_controls,
  data    = df[keep_2020_2023, ],
  cluster = ~domain_time
)



# Per domain

results <- list(
  "TWFE PPML (no controls)" = m_twfe_ppml_no_controls,
  "Sun-Abraham PPML (no controls)" = m_sa_ppml_no_controls,
  "Sun-Abraham PPML (treated after 2020 and before 2024)" = m_sa_ppml_treated_after_2020_before_2024
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

# Free memory
rm(keep_2020_2023)
gc()
