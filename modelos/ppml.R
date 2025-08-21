# ============================================
# DDD FE + PPML
# ============================================

# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)

# --- Assume your data.frame is `df` with columns:
# questions (y), meetings (T), member_id, domain, time, plus controls (e.g., x1, x2, ...)
# Make sure types are appropriate (factors/integers for IDs, numeric for y/T/controls).
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# # filter by domain
# df <- df_raw[df_raw$domain == "agriculture", ]

# 1) Build the three FE identifiers (member×domain, member×time, domain×time)
df$fe_i <- df$member_id # μ_id
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE) # μ_ct: country × time fixed effect
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE) # μ_pt: party × time fixed effect
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE) # μ_dt: domain × time fixed effect

# 2) (Recommended) build a cluster for domain×time for two-way clustering
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)


# 3) Build variables
# df$treated <- df$meetings > 0
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


# 4) Build the formula
# Build the controls part of the formula as a string

# Construct the full formula as a string, then convert to formula
full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_ct + fe_pt + fe_dt")
full_formula <- as.formula(full_formula_str)


# =========================
# A) DDD with OLS (feols)
# =========================

m_ddd_ols <- feols(
    full_formula,
    data    = df,
    cluster = ~cl_dt # two-way clustering: by member and by domain×time
)

# =============================
# B) DDD with PPML (fepois) - all domains
# =============================
m_ddd_ppml <- fepois(
    full_formula,
    data    = df,
    cluster = ~cl_dt
)
# Nice side-by-side table
modelsummary::msummary(
    list(
        "DDD OLS" = m_ddd_ols,
        "DDD PPML" = m_ddd_ppml
    ),
    gof_omit = "IC|Log|Adj|Pseudo|Within",
    coef_omit = "meps_",
    stars = TRUE
)


# ============================
# C) Compare domains
# ============================

domains <- unique(df$domain)

# function to run the loop
results_domains <- list(
    "Geral" = m_ddd_ppml
)

run_domain_loop <- function(df, full_formula) {
    for (domain in domains) {
        df_domain <- df[df$domain == domain, ]
        m_ddd_ppml_domain <- fepois(
            full_formula,
            data    = df_domain,
            cluster = ~cl_dt
        )
        results_domains[[domain]] <- m_ddd_ppml_domain
    }
    return(results_domains)
}

results_domains<- run_domain_loop(df, full_formula)

results_domains <- append(list("Geral" = m_ddd_ppml), results_domains)
modelsummary::msummary(results_domains, gof_omit = "IC|Log|Adj|Pseudo|Within", coef_omit = "meps_", stars = TRUE)


# ============================
# D) Compare different treatments
# ============================

results_alt_treatments <- list("Geral" = m_ddd_ppml)

alt_treatments <- c(
    "l_category_Business",
    "l_category_NGOs",
    "l_category_Other",
    "l_budget_cat_lower",
    "l_budget_cat_middle",
    "l_budget_cat_upper",
    "l_days_since_registration_lower",
    "l_days_since_registration_middle",
    "l_days_since_registration_upper"
)

# Ignore any days_since_registration columns
alt_treatments <- alt_treatments[!grepl("days_since_registration", alt_treatments)]

run_alt_treatment_loop <- function(df) {

    for (treatment in alt_treatments) {
        df_copy <- df
        df_copy$meetings <- df_copy[[treatment]]

        m_ddd_ppml_treatment <- fepois(
            full_formula,
            data    = df_copy,
            cluster = ~cl_dt
        )
        results_alt_treatments[[treatment]] <- m_ddd_ppml_treatment
    }

    return(results_alt_treatments)
}

results_alt_treatments <- run_alt_treatment_loop(df)

modelsummary::msummary(results_alt_treatments, gof_omit = "IC|Log|Adj|Pseudo|Within", coef_omit = "meps_", stars = TRUE)

results_alt_treatments


# ============================
# E) Compare different treatments in different domains
# ============================

results_alt_treatments_domains <- list("Geral" = m_ddd_ppml)

for (treatment in alt_treatments) {
    df_copy <- df
    df_copy$meetings <- df_copy[[treatment]]

    for (domain in domains) {
        df_copy_domain <- df_copy[df_copy$domain == domain, ]
        m_ddd_ppml_treatment_domain <- fepois(
            full_formula,
            data    = df_copy_domain,
            cluster = ~cl_dt
        )
        results_alt_treatments_domains[[paste(treatment, domain)]] <- m_ddd_ppml_treatment_domain
    }
    print(paste(treatment, "done"))
}


modelsummary::msummary(results_alt_treatments_domains, gof_omit = "IC|Log|Adj|Pseudo|Within", coef_omit = "meps_", stars = TRUE)



## ======================================
## F) Graphs to compare coefficients
## ======================================

# Helper to extract coefficient and SE for `meetings`
extract_meetings <- function(m) {
    cf <- coef(m)
    if (!("meetings" %in% names(cf))) return(list(b = NA_real_, se = NA_real_))
    b <- unname(cf["meetings"])
    V <- try(vcov(m), silent = TRUE)
    se <- NA_real_
    if (!inherits(V, "try-error") && all(c("meetings", "meetings") %in% colnames(V))) {
        se <- sqrt(V["meetings", "meetings"])
    }
    list(b = b, se = se)
}

# Baseline estimate for reference line
baseline_est <- NA_real_
if ("meetings" %in% names(coef(m_ddd_ppml))) {
    baseline_est <- unname(coef(m_ddd_ppml)["meetings"])
}

# Pretty labels for treatments and domains
pretty_treatment_label <- function(x) {
    map <- c(
        "baseline" = "Baseline (meetings)",
        "l_category_Business" = "Category: Business",
        "l_category_NGOs" = "Category: NGOs",
        "l_category_Other" = "Category: Other",
        "l_budget_cat_lower" = "Budget: Lower",
        "l_budget_cat_middle" = "Budget: Middle",
        "l_budget_cat_upper" = "Budget: Upper",
        "l_days_since_registration_lower" = "Registration days: Lower",
        "l_days_since_registration_middle" = "Registration days: Middle",
        "l_days_since_registration_upper" = "Registration days: Upper"
    )
    out <- unname(map[x])
    out[is.na(out)] <- x[is.na(out)]
    out
}

pretty_domain_label <- function(x) {
    # Replace underscores with spaces; keep original case to avoid breaking acronyms
    gsub("_", " ", x)
}

# ------------------------------
# F1) Across domains (baseline treatment)
# ------------------------------

df_domains_plot <- data.frame(domain = character(), estimate = numeric(), se = numeric(), stringsAsFactors = FALSE)

for (nm in names(results_domains)) {
    if (nm == "Geral") next
    fit <- results_domains[[nm]]
    x <- extract_meetings(fit)
    if (is.na(x$b)) next
    df_domains_plot <- rbind(df_domains_plot, data.frame(domain = nm, estimate = x$b, se = x$se))
}

if (nrow(df_domains_plot) > 0) {
    df_domains_plot$ci_lo <- df_domains_plot$estimate - 1.96 * df_domains_plot$se
    df_domains_plot$ci_hi <- df_domains_plot$estimate + 1.96 * df_domains_plot$se
    df_domains_plot$domain_label <- pretty_domain_label(df_domains_plot$domain)
    df_domains_plot$domain_label <- factor(df_domains_plot$domain_label, levels = df_domains_plot$domain_label[order(df_domains_plot$estimate)])

    p_domains <- ggplot(df_domains_plot, aes(x = estimate, y = domain_label)) +
        geom_vline(xintercept = 0, color = "black") +
        { if (!is.na(baseline_est)) geom_vline(xintercept = baseline_est, linetype = "dotted", color = "red") } +
        geom_point(color = "#1f77b4", size = 2.8) +
        geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, color = "#1f77b4") +
        labs(
            title = "Meetings effect across domains (PPML)",
            subtitle = "Points are estimates; bars are 95% CIs. Red dotted = overall baseline",
            x = "Estimate (meetings)", y = "Domain"
        ) +
        theme_minimal()

    ggsave("Tese/figures/fig_coeff_domains.png", p_domains, width = 10, height = 7, dpi = 300)
    ggsave("Tese/figures/fig_coeff_domains.pdf", p_domains, width = 10, height = 7)
}

# ------------------------------
# F2) Across treatments (overall)
# ------------------------------

df_treat_overall <- data.frame(treatment = character(), estimate = numeric(), se = numeric(), stringsAsFactors = FALSE)

for (nm in names(results_alt_treatments)) {
    fit <- results_alt_treatments[[nm]]
    x <- extract_meetings(fit)
    if (is.na(x$b)) next
    label <- if (nm == "Geral") "baseline" else nm
    df_treat_overall <- rbind(df_treat_overall, data.frame(treatment = label, estimate = x$b, se = x$se))
}

if (nrow(df_treat_overall) > 0) {
    df_treat_overall$ci_lo <- df_treat_overall$estimate - 1.96 * df_treat_overall$se
    df_treat_overall$ci_hi <- df_treat_overall$estimate + 1.96 * df_treat_overall$se
    df_treat_overall$treatment_label <- pretty_treatment_label(df_treat_overall$treatment)
    df_treat_overall$treatment_label <- factor(df_treat_overall$treatment_label, levels = df_treat_overall$treatment_label[order(df_treat_overall$estimate)])

    p_treat_overall <- ggplot(df_treat_overall, aes(x = estimate, y = treatment_label)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray70") +
        geom_point(color = "#2ca02c", size = 2.8) +
        geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, color = "#2ca02c") +
        labs(
            title = "Meetings effect across treatments (overall PPML)",
            subtitle = "Points are estimates; bars are 95% CIs.",
            x = "Estimate (meetings)", y = "Treatment"
        ) +
        theme_minimal()

    ggsave("Tese/figures/fig_coeff_treatments_overall.png", p_treat_overall, width = 10, height = 7, dpi = 300)
    ggsave("Tese/figures/fig_coeff_treatments_overall.pdf", p_treat_overall, width = 10, height = 7)
}

# ------------------------------
# F3) Across treatments per domain (faceted)
# ------------------------------

df_treat_by_domain <- data.frame(domain = character(), treatment = character(), estimate = numeric(), se = numeric(), stringsAsFactors = FALSE)

for (tr in alt_treatments) {
    for (dm in domains) {
        key <- paste(tr, dm)
        if (!(key %in% names(results_alt_treatments_domains))) next
        fit <- results_alt_treatments_domains[[key]]
        x <- extract_meetings(fit)
        if (is.na(x$b)) next
        df_treat_by_domain <- rbind(df_treat_by_domain, data.frame(domain = dm, treatment = tr, estimate = x$b, se = x$se))
    }
}

if (nrow(df_treat_by_domain) > 0) {
    df_treat_by_domain$ci_lo <- df_treat_by_domain$estimate - 1.96 * df_treat_by_domain$se
    df_treat_by_domain$ci_hi <- df_treat_by_domain$estimate + 1.96 * df_treat_by_domain$se
    df_treat_by_domain$treatment_label <- pretty_treatment_label(df_treat_by_domain$treatment)
    df_treat_by_domain$domain_label <- pretty_domain_label(df_treat_by_domain$domain)

    p_treat_by_domain <- ggplot(df_treat_by_domain, aes(x = estimate, y = treatment_label)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray70") +
        geom_point(color = "#9467bd", size = 2.6) +
        geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.18, color = "#9467bd") +
        labs(
            title = "Meetings effect: treatments by domain (PPML)",
            subtitle = "Points are estimates; bars are 95% CIs.",
            x = "Estimate (meetings)", y = "Treatment"
        ) +
        facet_wrap(~ domain_label, ncol = 3, scales = "free_y") +
        theme_minimal()

    ggsave("Tese/figures/fig_coeff_treatments_by_domain.png", p_treat_by_domain, width = 12, height = 8, dpi = 300)
    ggsave("Tese/figures/fig_coeff_treatments_by_domain.pdf", p_treat_by_domain, width = 12, height = 8)
}
