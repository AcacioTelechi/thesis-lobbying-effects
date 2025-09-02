# ============================================
# DDD FE + PPML
# ============================================

# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)

figures_dir <- file.path("Tese", "figures", "h1_test")
tables_dir <- file.path("Tese", "tables", "h1_test")
outputs_dir <- file.path("outputs", "h1_test")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)


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

# ---- Quadratic PPML model (meetings + meetings^2)
formula_quadratic_str <- paste0("questions ~ meetings + I(meetings^2) + ", controls_str, " | fe_ct + fe_pt + fe_dt")
formula_quadratic <- as.formula(formula_quadratic_str)

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

# Squared model
m_ddd_ppml_squared <- fepois(
    formula_quadratic,
    data    = df,
    cluster = ~cl_dt
)

# Nice side-by-side table
modelsummary::msummary(
    list(
        "DDD OLS" = m_ddd_ols,
        "DDD PPML" = m_ddd_ppml,
        "DDD PPML Squared" = m_ddd_ppml_squared
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

results_domains <- run_domain_loop(df, full_formula)

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
    if (!("meetings" %in% names(cf))) {
        return(list(b = NA_real_, se = NA_real_))
    }
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
baseline_est_se <- NA_real_
if ("meetings" %in% names(coef(m_ddd_ppml))) {
    baseline_est <- unname(coef(m_ddd_ppml)["meetings"])
    baseline_est_se <- sqrt(vcov(m_ddd_ppml)["meetings", "meetings"])
}

# Pretty labels for treatments and domains
pretty_treatment_label <- function(x) {
    map <- c(
        # "baseline" = "Baseline (meetings)",
        "l_category_Business" = "Categoria: Empresa",
        "l_category_NGOs" = "Categoria: ONG",
        "l_category_Other" = "Categoria: Outros",
        "l_budget_cat_lower" = "Orçamento: Baixo",
        "l_budget_cat_middle" = "Orçamento: Médio",
        "l_budget_cat_upper" = "Orçamento: Alto",
        "l_days_since_registration_lower" = "Dias desde registro: Baixo",
        "l_days_since_registration_middle" = "Dias desde registro: Médio",
        "l_days_since_registration_upper" = "Dias desde registro: Alto"
    )
    out <- unname(map[x])
    out[is.na(out)] <- x[is.na(out)]
    out
}

pretty_domain_label <- function(x) {
    # Replace underscores with spaces; keep original case to avoid breaking acronyms
    gsub("_", " ", x)
    map <- c(
        "agriculture" = "Agricultura",
        "education" = "Educação",
        "human_rights" = "Direitos Humanos",
        "environment_and_climate" = "Meio Ambiente e Clima",
        "health" = "Saúde",
        "infrastructure_and_industry" = "Infraestrutura e Indústria",
        "economics_and_trade" = "Economia e Comércio",
        "foreign_and_security_affairs" = "Assuntos Externos e Segurança",
        "technology" = "Tecnologia"
    )
    out <- unname(map[x])
    out[is.na(out)] <- x[is.na(out)]
    out
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
        geom_vline(xintercept = 0, color = "gray70") +
        {
            if (!is.na(baseline_est) && !is.na(baseline_est_se)) {
                b_lo <- baseline_est - 1.96 * baseline_est_se
                b_hi <- baseline_est + 1.96 * baseline_est_se
                list(
                    annotate("rect", xmin = b_lo, xmax = b_hi, ymin = -Inf, ymax = Inf, alpha = 0.15, fill = "#B45C1F"),
                    geom_vline(xintercept = baseline_est, linetype = "dashed", color = "#B45C1F")
                )
            } else {
                NULL
            }
        } +
        geom_point(color = "#1f77b4", size = 2.8) +
        geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, color = "#1f77b4") +
        labs(
            # title = "Meetings effect across domains (PPML)",
            # subtitle = "Points are estimates; bars are 95% CIs. Red dotted = overall baseline",
            x = "Efeito (meetings)", y = "Domínio"
        ) +
        theme_minimal()

    ggsave(file.path(figures_dir, "fig_coeff_domains.png"), p_domains, width = 10, height = 7, dpi = 300)
    ggsave(file.path(figures_dir, "fig_coeff_domains.pdf"), p_domains, width = 10, height = 7)
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
        geom_vline(xintercept = 0, color = "gray70") +
        {
            if (!is.na(baseline_est) && !is.na(baseline_est_se)) {
                b_lo <- baseline_est - 1.96 * baseline_est_se
                b_hi <- baseline_est + 1.96 * baseline_est_se
                list(
                    annotate("rect", xmin = b_lo, xmax = b_hi, ymin = -Inf, ymax = Inf, alpha = 0.15, fill = "#B45C1F"),
                    geom_vline(xintercept = baseline_est, linetype = "dashed", color = "#B45C1F")
                )
            } else {
                NULL
            }
        } +
        geom_point(color = "#1f77b4", size = 2.6) +
        geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.18, color = "#1f77b4") +
        labs(
            # title = "Meetings effect: treatments by domain (PPML)",
            # subtitle = "Points are estimates; bars are 95% CIs.",
            x = "Efeito (meetings)", y = "Tratamento"
        ) +
        facet_wrap(~domain_label, ncol = 3, scales = "free_y") +
        theme_minimal()

    ggsave(file.path(figures_dir, "fig_coeff_treatments_by_domain.png"), p_treat_by_domain, width = 12, height = 8, dpi = 300)
    ggsave(file.path(figures_dir, "fig_coeff_treatments_by_domain.pdf"), p_treat_by_domain, width = 12, height = 8)
}

# ============================
# G) Quadratic PPML model (meetings + meetings^2)
# ============================

# ---- Helper: build grid of meetings within observed support
q95 <- as.numeric(quantile(df$meetings, 0.95, na.rm = TRUE))
max_x <- max(5, floor(q95))
x_grid <- seq(0, max_x, length.out = 100)

# ---- Linear effect curve: factor = exp(beta1 * x)
coefs_lin <- coef(m_ddd_ppml)
vcov_lin  <- vcov(m_ddd_ppml)

if (!"meetings" %in% names(coefs_lin)) stop("Coefficient 'meetings' not found in m_ddd_ppml.")
b1 <- unname(coefs_lin["meetings"])
v11 <- as.numeric(vcov_lin["meetings", "meetings"]) # clustered variance

eta_lin <- b1 * x_grid
se_eta_lin <- sqrt(pmax(0, (x_grid^2) * v11))

factor_lin <- exp(eta_lin)
lower_lin  <- exp(eta_lin - 1.96 * se_eta_lin)
upper_lin  <- exp(eta_lin + 1.96 * se_eta_lin)

df_lin <- data.frame(
  meetings = x_grid,
  factor   = factor_lin,
  lower    = lower_lin,
  upper    = upper_lin
)

p_lin <- ggplot(df_lin, aes(x = meetings, y = factor)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#1f77b4", alpha = 0.15) +
  geom_line(color = "#1f77b4", size = 1) +
  labs(
    x = "Número de reuniões no mês",
    y = "Fator multiplicativo esperado em perguntas",
    title = "Efeito ceteris paribus (PPML): especificação linear",
    subtitle = "Curva: exp(β₁·meetings). Faixa: IC 95% via método delta (cluster em domínio×tempo)."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(figures_dir, "fig_effect_linear_ppml.pdf"), p_lin, width = 8.5, height = 5.2)
ggsave(file.path(figures_dir, "fig_effect_linear_ppml.png"), p_lin, width = 8.5, height = 5.2, dpi = 200)

# ---- Quadratic effect curve: factor = exp(b1*x + b2*x^2)
coefs_quad <- coef(m_ddd_ppml_squared)
vcov_quad  <- vcov(m_ddd_ppml_squared)

# Identify coefficient names robustly
name_b1 <- "meetings"
if (!name_b1 %in% names(coefs_quad)) stop("Coefficient 'meetings' not found in quadratic model.")

possible_sq_names <- names(coefs_quad)[
  grepl("^I\\(meetings\\^2\\)$", names(coefs_quad)) |
  grepl("meetings\\*\\*2", names(coefs_quad)) |
  grepl("meetings\\^2", names(coefs_quad)) |
  grepl("meetings2$|meetings_sq$|meetings_squared$", names(coefs_quad), ignore.case = TRUE)
]

if (length(possible_sq_names) == 0) stop("Squared coefficient for 'meetings' not found. Tried I(meetings^2) / meetings**2 patterns.")
name_b2 <- possible_sq_names[1]

b1_q <- unname(coefs_quad[name_b1])
b2_q <- unname(coefs_quad[name_b2])

V <- matrix(
  c(
    as.numeric(vcov_quad[name_b1, name_b1]),
    as.numeric(vcov_quad[name_b1, name_b2]),
    as.numeric(vcov_quad[name_b2, name_b1]),
    as.numeric(vcov_quad[name_b2, name_b2])
  ),
  nrow = 2, byrow = TRUE
)

eta_quad <- b1_q * x_grid + b2_q * (x_grid^2)

# se(eta) = sqrt([x, x^2] V [x; x^2])
g1 <- x_grid
g2 <- x_grid^2
se_eta_quad <- sqrt(pmax(0, g1^2 * V[1, 1] + 2 * g1 * g2 * V[1, 2] + g2^2 * V[2, 2]))

factor_quad <- exp(eta_quad)
upper_quad  <- exp(eta_quad + 1.96 * se_eta_quad)
lower_quad  <- exp(eta_quad - 1.96 * se_eta_quad)

df_quad <- data.frame(
  meetings = x_grid,
  factor   = factor_quad,
  lower    = lower_quad,
  upper    = upper_quad
)

p_quad <- ggplot(df_quad, aes(x = meetings, y = factor)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#d62728", alpha = 0.15) +
  geom_line(color = "#d62728", size = 1) +
  labs(
    x = "Número de reuniões no mês",
    y = "Fator multiplicativo esperado em perguntas",
    title = "Efeito ceteris paribus (PPML): especificação quadrática",
    subtitle = "Curva: exp(β₁·meetings + β₂·meetings²). Faixa: IC 95% via método delta (cluster em domínio×tempo)."
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(figures_dir, "fig_effect_quadratic_ppml.pdf"), p_quad, width = 8.5, height = 5.2)
ggsave(file.path(figures_dir, "fig_effect_quadratic_ppml.png"), p_quad, width = 8.5, height = 5.2, dpi = 200)

# ---- Export main regression table (m_ddd_ppml) to LaTeX
msummary(
  list("DDD PPML" = m_ddd_ppml),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE,
  output = file.path(tables_dir, "tab_main_ppml.tex")
)

# ---- Export combined summary for both models (full)
msummary(
  list(
    # "DDD PPML (sem FEs)" = m_ddd_ppml_no_fe,
    # "DDD PPML (FE - membro)" = m_ddd_ppml_no_fe_i,
    # "DDD PPML (FE - país)" = m_ddd_ppml_no_fe_ct,
    # "DDD PPML (FE - partido)" = m_ddd_ppml_no_fe_pt,
    "DDD PPML" = m_ddd_ppml,
    "DDD PPML (Quadrático)" = m_ddd_ppml_squared
  ),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE,
  output = file.path(tables_dir, "tab_main_ppml_both_full.tex")
)

# ---- Compact core table (only key coefficients) as LaTeX
star_from_p <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return("†")
  return("")
}

tidy_coef <- function(model, name) {
  ct <- tryCatch(summary(model)$coeftable, error = function(e) NULL)
  if (is.null(ct) || !(name %in% rownames(ct))) return(list(est = NA_real_, se = NA_real_, p = NA_real_))
  list(est = unname(ct[name, "Estimate"]), se = unname(ct[name, "Std. Error"]), p = unname(ct[name, ncol(ct)]))
}

fmt_coef <- function(est, se, p, exp_transform = FALSE) {
  if (is.na(est)) return("")
  if (exp_transform) {
    est <- exp(est)
    # SE on log-scale; we show transformed point estimate and keep SE on log-scale in parentheses
    return(sprintf("%.3f%s\\\\n(%.3f)", est, star_from_p(p), se))
  } else {
    return(sprintf("%.3f%s\\\\n(%.3f)", est, star_from_p(p), se))
  }
}

# Collect entries
c_lin  <- tidy_coef(m_ddd_ppml, "meetings")
c_q1   <- tidy_coef(m_ddd_ppml_squared, "meetings")
# Find squared term name
sq_names <- names(coef(m_ddd_ppml_squared))[grepl("^I\\(meetings\\^2\\)$|meetings\\*\\*2|meetings\\^2|meetings2$|meetings_sq$|meetings_squared$", names(coef(m_ddd_ppml_squared)))]
sq_name <- if (length(sq_names) > 0) sq_names[1] else NA_character_
c_q2   <- if (!is.na(sq_name)) tidy_coef(m_ddd_ppml_squared, sq_name) else list(est = NA_real_, se = NA_real_, p = NA_real_)

N_obs <- function(model) tryCatch(formatC(nobs(model), big.mark = ",", format = "f", digits = 0), error = function(e) "")

latex_core <- paste0(
  "\\begin{tabular}{lcc}\\n",
  "\\toprule\\n",
  " ", " & DDD PPML & DDD PPML (Quadrático) \\\\ ", "\\n",
  "\\midrule\\n",
  "Reuni\\u00F5es & ", 
  fmt_coef(
    c_lin$est, 
    c_lin$se, 
    c_lin$p, 
    exp_transform = FALSE
  ), 
  " & ", 
  fmt_coef(c_q1$est, c_q1$se, c_q1$p, exp_transform = FALSE), 
  " \\\\ ",
   "\\n",
  if (!is.na(c_q2$est)) paste0("Reuni\\u00F5es$^2$ &  & ", fmt_coef(c_q2$est, c_q2$se, c_q2$p, exp_transform = FALSE), " \\\\ \n") else "",
  "\\midrule\\n",
  "Observa\\u00E7\\u00F5es & ", N_obs(m_ddd_ppml), " & ", N_obs(m_ddd_ppml_squared), " \\\\ ", "\\n",
  "Efeitos fixos & membro; pa\\u00EDs$\u00D7$tempo; partido$\u00D7$tempo & membro; pa\\u00EDs$\u00D7$tempo; partido$\u00D7$tempo \\\\ ", "\\n",
  "Cluster & dom\\u00EDnio$\u00D7$tempo; membro & dom\\u00EDnio$\u00D7$tempo; membro \\\\ ", "\\n",
  "\\bottomrule\\n",
  "\\end{tabular}\\n"
)

writeLines(latex_core, con = file.path(tables_dir, "tab_main_ppml_both_core.tex"))

message("Saved figures to:")
message(file.path(figures_dir, "fig_effect_linear_ppml.pdf"))
message(file.path(figures_dir, "fig_effect_quadratic_ppml.pdf"))
message("Saved table to:")
message(file.path(tables_dir, "tab_main_ppml.tex"))
message(file.path(tables_dir, "tab_main_ppml_both_full.tex"))
message(file.path(tables_dir, "tab_main_ppml_both_core.tex"))








