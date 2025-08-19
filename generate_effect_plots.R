# ============================================
# Generate effect plots for Hypothesis 1 (PPML)
# ============================================

suppressPackageStartupMessages({
  library(fixest)
  library(data.table)
  library(ggplot2)
  library(modelsummary)
})

# ---- IO helpers
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

fig_dir <- file.path("Tese", "figures")
tab_dir <- file.path("Tese", "tables")
ensure_dir(fig_dir)
ensure_dir(tab_dir)

# ---- Load data (columns needed for these models)
# Controls
contsrols <- c(
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


needed_cols <- c(
  "questions", "meetings",
  "member_id", "country_time", "party_time", "domain_time",
  controls
)
controls_str <- paste(controls, collapse = " + ")

df <- fread("df_long.csv", select = needed_cols, showProgress = FALSE)
df <- as.data.frame(df)


# Types
df$questions   <- as.numeric(df$questions)
df$meetings    <- as.numeric(df$meetings)
df$member_id   <- as.factor(df$member_id)
df$country_time <- as.factor(df$country_time)
df$party_time   <- as.factor(df$party_time)
df$domain_time  <- as.factor(df$domain_time)

# Derived FE fields consistent with ddd_ppml_long.r
df$fe_i  <- df$member_id
df$fe_ct <- df$country_time
df$fe_pt <- df$party_time
df$fe_dt <- df$domain_time
df$cl_dt <- df$domain_time

# ---- Main PPML model (linear in meetings)
full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_ct + fe_pt + fe_dt")
formula_linear <- as.formula(full_formula_str)

m_ddd_ppml <- fepois(
  formula_linear,
  data    = df,
  cluster = ~cl_dt
)

# ---- Main PPML model with different FEs
full_formula_str <- paste0("questions ~ meetings + ", controls_str)
formula_linear <- as.formula(full_formula_str)
m_ddd_ppml_no_fe <- fepois(
  formula_linear,
  data    = df,
  cluster = ~cl_dt + member_id
)

full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_i")
formula_linear <- as.formula(full_formula_str)
m_ddd_ppml_no_fe_i <- fepois(
  formula_linear,
  data    = df,
  cluster = ~cl_dt + member_id
)

full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_ct")
formula_linear <- as.formula(full_formula_str)
m_ddd_ppml_no_fe_ct <- fepois(
  formula_linear,
  data    = df,
  cluster = ~cl_dt + member_id

)

full_formula_str <- paste0("questions ~ meetings + ", controls_str, " |  fe_pt")
formula_linear <- as.formula(full_formula_str)
m_ddd_ppml_no_fe_pt <- fepois(
  formula_linear,
  data    = df,
  cluster = ~cl_dt + member_id
)


# ---- Quadratic PPML model (meetings + meetings^2)
formula_quadratic_str <- paste0("questions ~ meetings + I(meetings^2) + ", controls_str, " | fe_ct + fe_pt + fe_dt")
formula_quadratic <- as.formula(formula_quadratic_str)
m_ddd_ppml_squared <- fepois(
  formula_quadratic,
  data    = df,
  cluster = ~cl_dt
)

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

ggsave(file.path(fig_dir, "fig8_effect_linear_ppml.pdf"), p_lin, width = 8.5, height = 5.2)
ggsave(file.path(fig_dir, "fig8_effect_linear_ppml.png"), p_lin, width = 8.5, height = 5.2, dpi = 200)

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
lower_quad  <- exp(eta_quad - 1.96 * se_eta_quad)
upper_quad  <- exp(eta_quad + 1.96 * se_eta_quad)

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

ggsave(file.path(fig_dir, "fig9_effect_quadratic_ppml.pdf"), p_quad, width = 8.5, height = 5.2)
ggsave(file.path(fig_dir, "fig9_effect_quadratic_ppml.png"), p_quad, width = 8.5, height = 5.2, dpi = 200)

# ---- Export main regression table (m_ddd_ppml) to LaTeX
msummary(
  list("DDD PPML" = m_ddd_ppml),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE,
  output = file.path(tab_dir, "tab_main_ppml.tex")
)

# ---- Export combined summary for both models (full)
msummary(
  list(
    "DDD PPML (sem FEs)" = m_ddd_ppml_no_fe,
    "DDD PPML (FE - membro)" = m_ddd_ppml_no_fe_i,
    "DDD PPML (FE - país)" = m_ddd_ppml_no_fe_ct,
    "DDD PPML (FE - partido)" = m_ddd_ppml_no_fe_pt,
    "DDD PPML" = m_ddd_ppml,
    "DDD PPML (Quadrático)" = m_ddd_ppml_squared
  ),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  stars = TRUE,
  # output = file.path(tab_dir, "tab_main_ppml_both_full.tex")
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
  "Reuni\\u00F5es & ", fmt_coef(c_lin$est, c_lin$se, c_lin$p, exp_transform = FALSE), " & ", fmt_coef(c_q1$est, c_q1$se, c_q1$p, exp_transform = FALSE), " \\\\ ", "\\n",
  if (!is.na(c_q2$est)) paste0("Reuni\\u00F5es$^2$ &  & ", fmt_coef(c_q2$est, c_q2$se, c_q2$p, exp_transform = FALSE), " \\\\ \n") else "",
  "\\midrule\\n",
  "Observa\\u00E7\\u00F5es & ", N_obs(m_ddd_ppml), " & ", N_obs(m_ddd_ppml_squared), " \\\\ ", "\\n",
  "Efeitos fixos & membro; pa\\u00EDs$\u00D7$tempo; partido$\u00D7$tempo & membro; pa\\u00EDs$\u00D7$tempo; partido$\u00D7$tempo \\\\ ", "\\n",
  "Cluster & dom\\u00EDnio$\u00D7$tempo; membro & dom\\u00EDnio$\u00D7$tempo; membro \\\\ ", "\\n",
  "\\bottomrule\\n",
  "\\end{tabular}\\n"
)

writeLines(latex_core, con = file.path(tab_dir, "tab_main_ppml_both_core.tex"))

message("Saved figures to:")
message(file.path(fig_dir, "fig8_effect_linear_ppml.pdf"))
message(file.path(fig_dir, "fig9_effect_quadratic_ppml.pdf"))
message("Saved table to:")
message(file.path(tab_dir, "tab_main_ppml.tex"))
message(file.path(tab_dir, "tab_main_ppml_both_full.tex"))
message(file.path(tab_dir, "tab_main_ppml_both_core.tex"))


