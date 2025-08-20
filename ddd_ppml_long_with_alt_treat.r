# ============================================
# DDD FE + PPML
# ============================================

# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)


treatments <- list(
  "",
  "_meetings_l_category_Business",
  "_meetings_l_category_NGOs",
  "_meetings_l_category_Other",
  "_meetings_l_budget_cat_lower",
  "_meetings_l_budget_cat_middle",
  "_meetings_l_budget_cat_upper",
  "_meetings_l_days_since_registration_lower",
  "_meetings_l_days_since_registration_middle",
  "_meetings_l_days_since_registration_upper"
)

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
  "meps_WORKING_GROUP___MEMBER_BUREAU"
)

# Build the controls part of the formula as a string
controls_str <- paste(controls, collapse = " + ")

# Construct the full formula as a string, then convert to formula
full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_ct + fe_pt + fe_dt")
full_formula <- as.formula(full_formula_str)

full_formula_str_squared <- paste0("questions ~ meetings + meetings**2 + ", controls_str, " | fe_ct + fe_pt + fe_dt")
full_formula_squared <- as.formula(full_formula_str_squared)

results <- list()
for (treatment in treatments) {
  df <- read.csv(paste0("df_long", treatment, ".csv"), stringsAsFactors = TRUE)

  # 1) Build the three FE identifiers (member×domain, member×time, domain×time)
  df$fe_i <- df$member_id # μ_id
  df$fe_ct <- df$country_time # μ_ct
  df$fe_pt <- df$party_time # μ_pt
  df$fe_dt <- df$domain_time # μ_dt
  df$cl_dt <- df$domain_time

  m_ddd_ppml <- fepois(
    full_formula,
    data    = df,
    cluster = ~ domain_time
  )

  results[[treatment]] <- m_ddd_ppml
}

modelsummary::msummary(
  results,
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  coef_omit = "meps_",
  stars = TRUE
)
