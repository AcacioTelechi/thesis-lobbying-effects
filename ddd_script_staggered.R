# ============================================
# DDD FE + PPML (wui)
# ============================================


# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)

# Prefer fast, column-selective IO if available
if (requireNamespace("data.table", quietly = TRUE)) {
  fread <- data.table::fread
} else {
  fread <- NULL
}

# --- Load only the columns we need to save memory
# Controls
controls <- c(
  "meps_POLITICAL_GROUP_5148.0",
  "meps_POLITICAL_GROUP_5151.0",
  "meps_POLITICAL_GROUP_5152.0",
  "meps_POLITICAL_GROUP_5153.0",
  "meps_POLITICAL_GROUP_5154.0",
  "meps_POLITICAL_GROUP_5155.0",
  "meps_POLITICAL_GROUP_5588.0",
  "meps_POLITICAL_GROUP_5704.0",
  "meps_POLITICAL_GROUP_6259.0",
  "meps_POLITICAL_GROUP_6561.0",
  "meps_POLITICAL_GROUP_7018.0",
  "meps_POLITICAL_GROUP_7028.0",
  "meps_POLITICAL_GROUP_7035.0",
  "meps_POLITICAL_GROUP_7036.0",
  "meps_POLITICAL_GROUP_7037.0",
  "meps_POLITICAL_GROUP_7038.0",
  "meps_POLITICAL_GROUP_7150.0",
  "meps_POLITICAL_GROUP_7151.0",
  "meps_COUNTRY_AUT",
  "meps_COUNTRY_BEL",
  "meps_COUNTRY_BGR",
  "meps_COUNTRY_CYP",
  "meps_COUNTRY_CZE",
  "meps_COUNTRY_DEU",
  "meps_COUNTRY_DNK",
  "meps_COUNTRY_ESP",
  "meps_COUNTRY_EST",
  "meps_COUNTRY_FIN",
  "meps_COUNTRY_FRA",
  "meps_COUNTRY_GBR",
  "meps_COUNTRY_GRC",
  "meps_COUNTRY_HRV",
  "meps_COUNTRY_HUN",
  "meps_COUNTRY_IRL",
  "meps_COUNTRY_ITA",
  "meps_COUNTRY_LTU",
  "meps_COUNTRY_LUX",
  "meps_COUNTRY_LVA",
  "meps_COUNTRY_MLT",
  "meps_COUNTRY_NLD",
  "meps_COUNTRY_POL",
  "meps_COUNTRY_PRT",
  "meps_COUNTRY_ROU",
  "meps_COUNTRY_SVK",
  "meps_COUNTRY_SVN",
  "meps_COUNTRY_SWE",
  "meps_COMMITTEE_PARLIAMENTARY_SPECIAL___CHAIR",
  "meps_COMMITTEE_PARLIAMENTARY_SPECIAL___MEMBER",
  "meps_COMMITTEE_PARLIAMENTARY_STANDING___CHAIR",
  "meps_COMMITTEE_PARLIAMENTARY_STANDING___MEMBER",
  "meps_COMMITTEE_PARLIAMENTARY_SUB___CHAIR",
  "meps_COMMITTEE_PARLIAMENTARY_SUB___MEMBER",
  "meps_DELEGATION_PARLIAMENTARY___CHAIR",
  "meps_DELEGATION_PARLIAMENTARY___MEMBER",
  "meps_EU_INSTITUTION___PRESIDENT",
  "meps_EU_INSTITUTION___QUAESTOR",
  "meps_EU_POLITICAL_GROUP___CHAIR",
  "meps_EU_POLITICAL_GROUP___MEMBER_BUREAU",
  "meps_EU_POLITICAL_GROUP___TREASURER",
  "meps_EU_POLITICAL_GROUP___TREASURER_CO",
  "meps_NATIONAL_CHAMBER___PRESIDENT_VICE",
  "meps_WORKING_GROUP___CHAIR",
  "meps_WORKING_GROUP___MEMBER",
  "meps_WORKING_GROUP___MEMBER_BUREAU",
  "log_meetings_l_category_Business",
  "log_meetings_l_category_NGOs",
  "log_meetings_l_category_Other",
  "log_meetings_l_budget_cat_lower",
  "log_meetings_l_budget_cat_middle",
  "log_meetings_l_budget_cat_upper",
  "log_meetings_l_days_since_registration_lower",
  "log_meetings_l_days_since_registration_middle",
  "log_meetings_l_days_since_registration_upper",
  "log_meetings_member_capacity_Committee_chair",
  "log_meetings_member_capacity_Delegation_chair",
  "log_meetings_member_capacity_Member",
  "log_meetings_member_capacity_Rapporteur",
  "log_meetings_member_capacity_Rapporteur_for_opinion",
  "log_meetings_member_capacity_Shadow_rapporteur",
  "log_meetings_member_capacity_Shadow_rapporteur_for_opinion"
)

# Required columns for current models: outcome, treatment-time indicator, FE and clustering fields, dates
needed_cols <- c(
  "questions", "meetings", "post", "member_id", "domain_time", "party_time", "country_time",
  "time_fe", "first_treatment_period", "treated", controls
)

# Build the controls part of the formula as a string
controls_str <- paste(controls, collapse = " + ")

if (!is.null(fread)) {
  df <- fread("df_long.csv", select = needed_cols, showProgress = FALSE)
  df <- as.data.frame(df)
} else {
  # Fallback if data.table not installed
  df <- read.csv("df_long.csv", stringsAsFactors = FALSE)
  df <- df[, intersect(colnames(df), needed_cols)]
}

# Convert types compactly
df$fe_i <- as.factor(df$member_id)
df$domain_time <- as.factor(df$domain_time)
df$fe_pt <- as.factor(df$party_time)
df$fe_ct <- as.factor(df$country_time)
df$questions <- as.numeric(df$questions)
df$post <- as.integer(df$post)
if ("treated" %in% names(df)) df$treated <- as.integer(df$treated)

# Never treat as zero
df$first_treatment_period <- as.Date(as.character(df$first_treatment_period))
df$first_treatment_period[is.na(df$first_treatment_period)] <- 0

# Create event study variables (as months)
df$event_study_period <- as.integer(as.numeric(as.Date(df$time_fe) - as.Date(df$first_treatment_period)) / 30)
df$event_study_period_m_3 <- df$event_study_period + 3 == 0
df$event_study_period_m_2 <- df$event_study_period + 2 == 0
df$event_study_period_m_1 <- df$event_study_period + 1 == 0
df$event_study_period_0 <- df$event_study_period == 0
df$event_study_period_p_1 <- df$event_study_period - 1 == 0
df$event_study_period_p_2 <- df$event_study_period - 2 == 0
df$event_study_period_p_3 <- df$event_study_period - 3 == 0


# Different post variables
df$post_2020 <- df$time_fe >= as.Date("2020-01-01")


# Fixed effects will use existing columns directly (avoid duplicating columns)

# 3) Date variables (overwrite in place to avoid extra copies)
df$time_fe <- as.Date(as.character(df$time_fe))
df$first_treatment_period <- as.Date(as.character(df$first_treatment_period))

# Helper logical for restricted cohort (no persistent copy)
keep_2020_2023 <- df$treated == 0 |
  (df$first_treatment_period >= as.Date("2020-01-01") &
     df$first_treatment_period < as.Date("2024-01-01"))

# 4) Build formulas (no-controls variants only to reduce memory)
# TWFE PPML with time-varying treatment indicator (post)
twfe_formula_no_controls <- as.formula("questions ~ meetings | fe_ct + fe_pt + domain_time")

# # Sun & Abraham event-study using cohort (first_treatment_period) and period (time_fe)
# sa_formula_no_controls <- as.formula("questions ~ sunab(first_treatment_period, time_fe) | member_id + domain_time")

# Event study with restricted leads and lags [-3,3]
# sa_formula_restricted <- as.formula("questions ~ sunab(first_treatment_period, time_fe, ref.p = -1, ref.c = 0, bin.rel = -3:3) | member_id + domain_time")

# Event study with restricted leads and lags [-3,3]
es_formula_restricted <- as.formula("questions ~ meetings + event_study_period_m_3 + event_study_period_m_2  + event_study_period_0 + event_study_period_p_1 + event_study_period_p_2 + event_study_period_p_3 | fe_ct + fe_pt + domain_time")

es_formula_restricted_with_controls <- as.formula(paste0("questions ~ meetings + event_study_period_m_3 + event_study_period_m_2  + event_study_period_m_1 + event_study_period_0 + event_study_period_p_1 + event_study_period_p_2 + event_study_period_p_3 + ", controls_str, " | fe_ct + fe_pt + domain_time"))

es_formula_restricted_with_controls_2 <- as.formula(paste0("questions ~ event_study_period_m_3 + event_study_period_m_2  + event_study_period_m_1 + event_study_period_0 + event_study_period_p_1 + event_study_period_p_2 + event_study_period_p_3 + ", controls_str, " | fe_ct + fe_pt + domain_time"))


ddd_binary_formula <- as.formula("questions ~ post_2020 * treated  | fe_ct + fe_pt + domain_time")

# =============================
# Staggered DiD with PPML (fepois)
# =============================

# TWFE with post only
m_twfe_ppml_no_controls <- fepois(
  twfe_formula_no_controls,
  data    = df,
  cluster = ~member_id + domain_time
)

# # Sun & Abraham event-study
# m_sa_ppml_no_controls <- fepois(
#   sa_formula_no_controls,
#   data    = df,
#   cluster = ~domain_time
# )

# # Sun & Abraham event-study with restricted leads/lags [-3,3]
# m_sa_ppml_restricted <- fepois(
#   sa_formula_restricted,
#   data    = df,
#   cluster = ~domain_time
# )

# m_sa_ppml_treated_after_2020_before_2024 <- fepois(
#   sa_formula_no_controls,
#   data    = df[keep_2020_2023, ],
#   cluster = ~domain_time
# )

# # Restricted event study for the filtered sample
# m_sa_ppml_restricted_filtered <- fepois(
#   sa_formula_restricted,
#   data    = df[keep_2020_2023, ],
#   cluster = ~domain_time
# )

# Event study with restricted leads and lags [-3,3]
m_es_ppml_restricted <- fepois(
  es_formula_restricted,
  data    = df,
  cluster = ~member_id + domain_time
)

m_es_ppml_restricted_with_controls <- fepois(
  es_formula_restricted_with_controls,
  data    = df,
  cluster = ~member_id + domain_time
)

m_es_ppml_restricted_with_controls_2 <- fepois(
  es_formula_restricted_with_controls_2,
  data    = df,
  cluster = ~member_id + domain_time
)

m_ddd_binary_ppml_restricted <- fepois(
  ddd_binary_formula,
  data    = df[keep_2020_2023,],
  cluster = ~member_id + domain_time
)

# Per domain

results <- list(
  "TWFE PPML (no controls)" = m_twfe_ppml_no_controls,
  # "Sun-Abraham PPML (no controls)" = m_sa_ppml_no_controls,
  # "Sun-Abraham PPML restricted [-3,3]" = m_sa_ppml_restricted,
  # "Sun-Abraham PPML (treated after 2020 and before 2024)" = m_sa_ppml_treated_after_2020_before_2024,
  # "Sun-Abraham PPML restricted [-3,3] filtered" = m_sa_ppml_restricted_filtered
  "Event study PPML restricted [-3,3]" = m_es_ppml_restricted,
  "Event study PPML restricted [-3,3] with controls" = m_es_ppml_restricted_with_controls,
  "Event study PPML restricted [-3,3] with controls 2" = m_es_ppml_restricted_with_controls_2
)

# Nice side-by-side table
modelsummary::msummary(results, stars = TRUE)

# Event study plot for the restricted model
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  
  # Extract event study coefficients
  event_coefs <- coef(m_es_ppml_restricted_with_controls_2)
  event_se <- sqrt(diag(vcov(m_es_ppml_restricted_with_controls_2)))
  
  # Create event study data frame
  event_data <- data.frame(
    period = c(-3, -2, -1, 0, 1, 2, 3),
    coefficient = event_coefs[grep("^event_study_period", names(event_coefs))],
    se = event_se[grep("^event_study_period", names(event_se))]
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
  ggsave("./event_study_restricted.png", p, width = 10, height = 6, dpi = 300)
  ggsave("./event_study_restricted.pdf", p, width = 10, height = 6)
}

# Free memory
rm(keep_2020_2023)
gc()