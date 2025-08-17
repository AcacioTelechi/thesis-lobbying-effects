# ============================================
# DDD FE + PPML (wui)
# ============================================


# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)
library(dplyr)

# --- Assume your data.frame is `df` with columns:
# questions (y), meetings (T), member_id, domain, time, plus controls (e.g., x1, x2, ...)
# Make sure types are appropriate (factors/integers for IDs, numeric for y/T/controls).
df <- read.csv("df_long.csv", stringsAsFactors = TRUE)

# # filter by domain
# df <- df_raw[df_raw$domain == "agriculture", ]

# 1) Build the three FE identifiers (member×domain, member×time, domain×time)
df$fe_i <- df$member_id # μ_id
df$fe_ct <- df$country_time # μ_ct
df$fe_pt <- df$party_time # μ_pt
df$fe_dt <- df$domain_time # μ_dt

# 2) (Recommended) build a cluster for domain×time for two-way clustering
df$cl_dt <- df$domain_time


# 3) Build variables
# treated
df <- df %>%
  group_by(member_id, domain) %>%
  mutate(treated = any(meetings > 0)) %>%
  ungroup()

# post
# For each member_id (i) and domain (d), set post == TRUE for the min(time_fe) where treated == TRUE
df$post <- FALSE
df <- df %>%
  group_by(member_id, domain) %>%
  mutate(
    # Convert time_fe to Date for comparison if it's not already
    time_fe_date = as.Date(as.character(time_fe)),
    min_treat_time = if (any(meetings > 0)) min(time_fe_date[meetings > 0], na.rm = TRUE) else as.Date(NA)
  ) %>%
  ungroup() %>%
  mutate(
    post = !is.na(min_treat_time) & as.Date(as.character(time_fe)) >= min_treat_time
  ) %>%
  select(-min_treat_time, -time_fe_date)
  
# cj - cohort indicator (independent of treatment status)
df$cj <- (as.Date(as.character(df$time_fe)) < as.Date("2022-01-01"))  # early period indicator

df$month <- format(as.Date(df$time_fe), "%m")
df$year <- format(as.Date(df$time_fe), "%Y")


# Create a df for all the units that received first treatment after 2020 and befor 2024
# but keep the untreated

## find the first time a unit was treated after 2020 and before 2024
df_first_treated <- df %>%
  group_by(member_id, domain) %>%
  mutate(
    # Convert time_fe to Date if not already
    time_fe_date = as.Date(as.character(time_fe)),
    first_treatment_date = if (any(meetings > 0)) min(time_fe_date[meetings > 0], na.rm = TRUE) else as.Date(NA)
  ) %>%
  ungroup() %>%
  select(-time_fe_date)

# merge the treated df with the original df
df <- merge(
  df,
  df_first_treated[c("member_id", "domain", "first_treatment_date")],
  by = c("member_id", "domain"),
  all.x = TRUE
)

# create a new variable that is 1 if the unit was treated after 2020 and before 2024, 0 otherwise
df_treated_after_2020_before_2024 <- df %>%
  filter((first_treatment_date >= "2020-01-01" & first_treatment_date < "2024-01-01") | treated == FALSE) 


# 4) Build the formula
# Build the controls part of the formula as a string
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
  # "log_meetings_l_category_Business",
  # "log_meetings_l_category_NGOs",
  # "log_meetings_l_category_Other",
  # "log_meetings_l_budget_cat_lower",
  # "log_meetings_l_budget_cat_middle",
  # "log_meetings_l_budget_cat_upper",
  # "log_meetings_l_days_since_registration_lower",
  # "log_meetings_l_days_since_registration_middle",
  # "log_meetings_l_days_since_registration_upper",
  "log_meetings_member_capacity_Committee_chair",
  "log_meetings_member_capacity_Delegation_chair",
  "log_meetings_member_capacity_Member",
  "log_meetings_member_capacity_Rapporteur",
  "log_meetings_member_capacity_Rapporteur_for_opinion",
  "log_meetings_member_capacity_Shadow_rapporteur",
  "log_meetings_member_capacity_Shadow_rapporteur_for_opinion"
)

controls_str <- paste(controls, collapse = " + ")



# Construct the full formula as a string, then convert to formula
# yit = β0 + β1Treati + β2Postt + β3Cj + β4(Treati*Postt) + β5(Treati*Cj) + β6(Postt*Cj) + β7(Treati*Postt*Cj) + ϵit:
full_formula_str <- paste0("questions ~  treated*post*cj + ", controls_str, " | fe_i + fe_ct + fe_pt")
full_formula <- as.formula(full_formula_str)

# Alternative specification if cj depends on treated (avoids collinearity):
# full_formula_str_alt <- paste0("questions ~  treated*post + cj*post + ", controls_str, " | fe_i + fe_ct + fe_pt")
# full_formula_alt <- as.formula(full_formula_str_alt)

full_formula_str_no_controls <- paste0("questions ~  treated*post*cj  | fe_i + fe_ct + fe_pt")
full_formula_no_controls <- as.formula(full_formula_str_no_controls)

# # with time
# full_formula_str_with_time <- paste0("questions ~ meetings + month + year + ", controls_str, " | fe_i + fe_ct + fe_pt")
# full_formula_with_time <- as.formula(full_formula_str_with_time)

# full_formula_str_squared_with_time <- paste0("questions ~ meetings + meetings**2 + month + year + ", controls_str, " | fe_i + fe_ct + fe_pt")
# full_formula_squared_with_time <- as.formula(full_formula_str_squared_with_time)


# =============================
# B) DDD with PPML (fepois) - all domains
# =============================
m_ddd_ppml <- fepois(
  full_formula,
  data    = df,
  cluster = ~cl_dt
)

m_ddd_ppml_no_controls <- fepois(
  full_formula_no_controls,
  data    = df,
  cluster = ~cl_dt
)

m_ddd_ppml_treated_after_2020_before_2024 <- fepois(
  full_formula_no_controls,
  data    = df_treated_after_2020_before_2024,
  cluster = ~cl_dt
)


# Per domain

results <- list(
  "DDD PPML" = m_ddd_ppml,
  "DDD PPML (no controls)" = m_ddd_ppml_no_controls,
  "DDD PPML (treated after 2020 and before 2024)" = m_ddd_ppml_treated_after_2020_before_2024
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
modelsummary::msummary(
  results,
  gof_omit = "IC|Log|Adj|Pseudo|Within|meps_|month|year",
  coef_omit = paste0("^", paste(controls, collapse = "|"), "$"),
  stars = TRUE
)
