# ============================================
# DDD FE + PPML with Propensity Score Matching
# ============================================

# Load packages
library(fixest)
library(modelsummary)
library(MatchIt)
library(dplyr)
library(cobalt)   # for balance plots


# Load your dataframe
df <- read.csv("df_long.csv", stringsAsFactors = TRUE)

df$domain_time <- paste(df$domain, df$time_fe, sep = "_")

# Build fixed effects & cluster IDs
df$fe_i  <- df$member_id
df$fe_ct <- df$country_time
df$fe_pt <- df$party_time
df$cl_dt <- df$domain_time

# Treatment indicator
df$treated <- df$meetings > 0

# Extract month & year
df$month <- format(as.Date(df$time_fe), "%m")
df$year  <- format(as.Date(df$time_fe), "%Y")

# Controls vector (paste your full list here)
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

# Controls string for formulas
controls_str <- paste(controls, collapse = " + ")

# -------------------------------
# Step 1: PSM within domain-month
# -------------------------------

# List to store matchit objects for balance check
matchit_objects <- list()

matched_list <- df %>%
  group_by(domain_time) %>%
  group_map(~ {
    # Only run matching if both treated & untreated exist
    if(length(unique(.x$treated)) == 2){
      m <- matchit(
        formula = as.formula(
          paste("treated ~", controls_str)
        ),
        data    = .x,
        method  = "nearest",
        distance = "logit",
        ratio   = 1
      )
      # Save the matchit object for later balance checks
      matchit_objects[.x$domain_time[1]] <<- m
      return(match.data(m))
    } else {
      return(NULL) # Skip if no variation in treatment
    }
  })

# Combined matched dataset
df_matched <- bind_rows(matched_list)


# ------------------------------
# Step 2: Check balance after PSM
# ------------------------------
cat("\nBalance check for first matched set:\n")
if(length(matchit_objects) > 0){
  first_key <- names(matchit_objects)[1]
  print(summary(matchit_objects[[first_key]])) # standardized differences before & after matching
}

# ------------------------------
# Step 3: Build formulas
# ------------------------------
full_formula <- as.formula(
  paste0("questions ~ meetings + ", controls_str, " | fe_i + fe_ct + fe_pt")
)

# ------------------------------
# Step 4: Run models (full sample)
# ------------------------------
m_full_ols <- feols(
  full_formula,
  data    = df,
  cluster = ~cl_dt
)

m_full_ppml <- fepois(
  full_formula,
  data    = df,
  cluster = ~cl_dt
)

# ------------------------------
# Step 5: Run models (matched sample)
# ------------------------------
m_matched_ols <- feols(
  full_formula,
  data    = df_matched,
  cluster = ~cl_dt
)

m_matched_ppml <- fepois(
  full_formula,
  data    = df_matched,
  cluster = ~cl_dt
)

# ------------------------------
# Step 6: Compare results
# ------------------------------
modelsummary::msummary(
  list(
    "DDD OLS (Full sample)"     = m_full_ols,
    "DDD PPML (Full sample)"    = m_full_ppml,
    "DDD OLS (Matched sample)"  = m_matched_ols,
    "DDD PPML (Matched sample)" = m_matched_ppml
  ),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE
)
