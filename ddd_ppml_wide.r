# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)

# --- Assume your data.frame is `df` with columns:
# questions (y), meetings (T), member_id, domain, time, plus controls (e.g., x1, x2, ...)
# Make sure types are appropriate (factors/integers for IDs, numeric for y/T/controls).
df <- read.csv('df_wide.csv', stringsAsFactors=TRUE)

# # filter by domain
# df <- df_raw[df_raw$domain == "agriculture", ]

# 1) Build the three FE identifiers (member×domain, member×time, domain×time)
df$fe_i <- df$member_id   # μ_id
df$fe_it <- df$country_time   # μ_ct
df$fe_dt <- df$party_time   # μ_pt

# 2) (Recommended) build a cluster for domain×time for two-way clustering
df$cl_dt <- df$domain_time

# Controls
controls <- c(
    # "meps_POLITICAL_GROUP_5148.0",
    # "meps_POLITICAL_GROUP_5151.0",
    # "meps_POLITICAL_GROUP_5152.0",
    # "meps_POLITICAL_GROUP_5153.0",
    # "meps_POLITICAL_GROUP_5154.0",
    # "meps_POLITICAL_GROUP_5155.0",
    # "meps_POLITICAL_GROUP_5588.0",
    # "meps_POLITICAL_GROUP_5704.0",
    # "meps_POLITICAL_GROUP_6259.0",
    # "meps_POLITICAL_GROUP_6561.0",
    # "meps_POLITICAL_GROUP_7018.0",
    # "meps_POLITICAL_GROUP_7028.0",
    # "meps_POLITICAL_GROUP_7035.0",
    # "meps_POLITICAL_GROUP_7036.0",
    # "meps_POLITICAL_GROUP_7037.0",
    # "meps_POLITICAL_GROUP_7038.0",
    # "meps_POLITICAL_GROUP_7150.0",
    # "meps_POLITICAL_GROUP_7151.0",
    # "meps_COUNTRY_AUT",
    # "meps_COUNTRY_BEL",
    # "meps_COUNTRY_BGR",
    # "meps_COUNTRY_CYP",
    # "meps_COUNTRY_CZE",
    # "meps_COUNTRY_DEU",
    # "meps_COUNTRY_DNK",
    # "meps_COUNTRY_ESP",
    # "meps_COUNTRY_EST",
    # "meps_COUNTRY_FIN",
    # "meps_COUNTRY_FRA",
    # "meps_COUNTRY_GBR",
    # "meps_COUNTRY_GRC",
    # "meps_COUNTRY_HRV",
    # "meps_COUNTRY_HUN",
    # "meps_COUNTRY_IRL",
    # "meps_COUNTRY_ITA",
    # "meps_COUNTRY_LTU",
    # "meps_COUNTRY_LUX",
    # "meps_COUNTRY_LVA",
    # "meps_COUNTRY_MLT",
    # "meps_COUNTRY_NLD",
    # "meps_COUNTRY_POL",
    # "meps_COUNTRY_PRT",
    # "meps_COUNTRY_ROU",
    # "meps_COUNTRY_SVK",
    # "meps_COUNTRY_SVN",
    # "meps_COUNTRY_SWE",
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
    # "meps_EU_POLITICAL_GROUP___TREASURER",
    # "meps_EU_POLITICAL_GROUP___TREASURER_CO",
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

# 3) Build "treated" variable
df$treated <- df$meetings > 0

# 4) Build the formula
# Build the controls part of the formula as a string
controls_str <- paste(controls, collapse = " + ")

# Construct the full formula as a string, then convert to formula
full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_i + fe_it + fe_dt")
full_formula <- as.formula(full_formula_str)

# =========================
# A) DDD with OLS (feols)
# =========================

m_ddd_ols <- feols(
  full_formula,
  data    = df,
  cluster = ~ cl_dt  # two-way clustering: by member and by domain×time
)

# =============================
# B) DDD with PPML (fepois) - all domains
# =============================
# PPML handles zeros in `questions` naturally and uses a log link.
m_ddd_ppml <- fepois(
  full_formula,
  data    = df,
  cluster = ~ cl_dt
)


full_formula_str_squared <- paste0("questions ~ meetings + meetings**2 + ", controls_str, " | fe_i + fe_it + fe_dt")
full_formula_squared <- as.formula(full_formula_str_squared)
m_ddd_ppml_squared <- fepois(
  full_formula_squared,
  data    = df,
  cluster = ~ cl_dt
)

# Nice side-by-side table
modelsummary::msummary(
  list("DDD OLS" = m_ddd_ols, "DDD PPML" = m_ddd_ppml, "DDD PPML Squared" = m_ddd_ppml_squared),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE
)


# ============================
# C) Compare domains
# ============================

domains <- unique(df$domain)

# function to run the loop
run_loop <- function(df, full_formula) {
    results <- list()
    for (domain in domains) {
        df_domain <- df[df$domain == domain, ]
        m_ddd_ppml_domain <- fepois(
            full_formula,
            data    = df_domain,
            cluster = ~ cl_dt
        )
    results[[domain]] <- m_ddd_ppml_domain
    }
    return(results)
}

results <- run_loop(df, full_formula)

modelsummary::msummary(results, gof_omit = "IC|Log|Adj|Pseudo|Within", stars = TRUE)

