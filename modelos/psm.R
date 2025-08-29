# ============================
# Propensity Score Matching
# ============================

library(MatchIt)
library(cobalt)
library(fixest)
library(modelsummary)
library(ggplot2)

df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# 1) Build the three FE identifiers (member×domain, member×time, domain×time)
df$fe_i <- df$member_id # μ_id
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE) # μ_ct: country × time fixed effect
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE) # μ_pt: party × time fixed effect
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE) # μ_dt: domain × time fixed effect

# 2) (Recommended) build a cluster for domain×time for two-way clustering
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)


# 3) Build variables
# treated
tmp_has_meeting <- df$meetings > 0 & !is.na(df$meetings)
by_member_any <- aggregate(tmp_has_meeting ~ member_id, data = df, FUN = function(x) any(x, na.rm = TRUE))
by_member_any$member_id <- as.character(by_member_any$member_id)
treated_ids <- by_member_any$member_id[by_member_any$tmp_has_meeting]
df$treated <- df$member_id %in% treated_ids

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
    # "log_meetings_l_category_Business",
    # "log_meetings_l_category_NGOs",
    # "log_meetings_l_category_Other",
    # "log_meetings_l_budget_cat_lower",
    # "log_meetings_l_budget_cat_middle",
    # "log_meetings_l_budget_cat_upper",
    # "log_meetings_l_days_since_registration_lower",
    # "log_meetings_l_days_since_registration_middle",
    # "log_meetings_l_days_since_registration_upper",
    # "log_meetings_member_capacity_Committee_chair",
    # "log_meetings_member_capacity_Delegation_chair",
    # "log_meetings_member_capacity_Member",
    # "log_meetings_member_capacity_Rapporteur",
    # "log_meetings_member_capacity_Rapporteur_for_opinion",
    # "log_meetings_member_capacity_Shadow_rapporteur",
    # "log_meetings_member_capacity_Shadow_rapporteur_for_opinion"
)
controls_str <- paste(controls, collapse = " + ")


dir.create("Tese/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("Tese/tables", showWarnings = FALSE, recursive = TRUE)

# Propensity score model: include domain, time and observed covariates
psm_covariates <- c("domain", "Y.m", controls)
psm_rhs <- paste(psm_covariates, collapse = " + ")
psm_formula_str <- paste0("treated ~ ", psm_rhs)
psm_formula <- as.formula(psm_formula_str)

# Nearest-neighbor matching with replacement on logit PS
m_psm <- matchit(
    formula  = psm_formula,
    data     = df,
    method   = "nearest",
    distance = "logit",
    replace  = TRUE,
    ratio    = 1
)

# Balance (love plot)
pdf("Tese/figures/psm_love_plot.pdf", width = 8, height = 10)
love.plot(m_psm, binary = "std", var.order = "unadjusted", abs = TRUE, line = TRUE, stat = "std")
dev.off()
png("Tese/figures/psm_love_plot.png", width = 1000, height = 1200, res = 140)
love.plot(m_psm, binary = "std", var.order = "unadjusted", abs = TRUE, line = TRUE, stat = "std")
dev.off()

# Matched data with weights
df_matched <- match.data(m_psm)

# ATT on matched sample (FE OLS and PPML) using matching weights
psm_fe_formula_str <- paste0("questions ~ treated + ", controls_str, " | fe_ct + fe_pt + fe_dt")
psm_fe_formula <- as.formula(psm_fe_formula_str)

m_psm_feols <- feols(
    psm_fe_formula,
    data    = df_matched,
    weights = ~weights,
    cluster = ~cl_dt
)

m_psm_fepois <- fepois(
    psm_fe_formula,
    data    = df_matched,
    weights = ~weights,
    cluster = ~cl_dt
)

modelsummary::msummary(
    list(
        "PSM OLS"  = m_psm_feols,
        "PSM PPML" = m_psm_fepois
    ),
    gof_omit = "IC|Log|Adj|Pseudo|Within",
    coef_omit = "meps_",
    stars = TRUE,
    output = "Tese/tables/psm_models.tex"
)

# Coefficient plot for ATT (treated)
extract_treated <- function(m) {
    cf <- coef(m)
    if (!("treated" %in% names(cf))) return(list(b = NA_real_, se = NA_real_))
    b <- unname(cf["treated"])
    V <- try(vcov(m), silent = TRUE)
    se <- NA_real_
    if (!inherits(V, "try-error") && all(c("treated", "treated") %in% colnames(V))) {
        se <- sqrt(V["treated", "treated"])
    }
    list(b = b, se = se)
}

df_psm_coef <- data.frame(
    model = c("PSM OLS", "PSM PPML"),
    estimate = NA_real_,
    se = NA_real_,
    stringsAsFactors = FALSE
)

x1 <- extract_treated(m_psm_feols)
x2 <- extract_treated(m_psm_fepois)
df_psm_coef$estimate <- c(x1$b, x2$b)
df_psm_coef$se <- c(x1$se, x2$se)
df_psm_coef$ci_lo <- df_psm_coef$estimate - 1.96 * df_psm_coef$se
df_psm_coef$ci_hi <- df_psm_coef$estimate + 1.96 * df_psm_coef$se

p_psm <- ggplot(df_psm_coef, aes(x = estimate, y = model)) +
    geom_vline(xintercept = 0, color = "gray70") +
    geom_point(color = "#2ca02c", size = 2.8) +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, color = "#2ca02c") +
    labs(x = "Efeito (treated)", y = "Modelo") +
    theme_minimal()

ggsave("Tese/figures/psm_effects.png", p_psm, width = 8, height = 5, dpi = 300)
ggsave("Tese/figures/psm_effects.pdf", p_psm, width = 8, height = 5)

message("PSM outputs saved to Tese/figures and Tese/tables.")