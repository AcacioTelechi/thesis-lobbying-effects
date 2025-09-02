# install.packages("fixest") # if needed
# install.packages("MASS") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)
library(MASS) # For glm.nb

# Additional packages for marginal means and data export
# install.packages("emmeans") # if needed
# install.packages("data.table") # if needed
library(emmeans)
library(data.table)


# Ensure output directories
figures_dir <- file.path("Tese", "figures", "h2_test")
tables_dir <- file.path("Tese", "tables", "h2_test")
outputs_dir <- file.path("outputs", "h2_test")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)

#===========================================
# 1) Estimate using the PPML from H1
#===========================================

df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# 1) Build the three FE identifiers (member×domain, member×time, domain×time)
df$fe_i <- df$member_id # μ_id
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE) # μ_ct: country × time fixed effect
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE) # μ_pt: party × time fixed effect
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE) # μ_dt: domain × time fixed effect

# 2) build a cluster for domain×time for two-way clustering
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)

# 3) Build controls
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
full_formula_str <- paste0("questions ~ meetings + ", controls_str, " | fe_ct + fe_pt + fe_dt")
full_formula <- as.formula(full_formula_str)

full_formula_squared_str <- paste0("questions ~ meetings + I(meetings^2) + ", controls_str, " | fe_ct + fe_pt + fe_dt")
full_formula_squared <- as.formula(full_formula_squared_str)

m_ddd_ppml <- fepois(
    full_formula,
    data    = df,
    cluster = ~cl_dt
)

m_ddd_ppml_squared <- fepois(
    full_formula_squared,
    data    = df,
    cluster = ~cl_dt
)

# ============================
# 2) Compare different treatments
# ============================

results_alt_treatments <- list("Geral" = m_ddd_ppml, "Geral Squared" = m_ddd_ppml_squared)

alt_treatments <- c(
    "l_category_Business",
    "l_category_NGOs",
    "l_category_Other"
    # "l_budget_cat_lower",
    # "l_budget_cat_middle",
    # "l_budget_cat_upper",
    # "l_days_since_registration_lower",
    # "l_days_since_registration_middle",
    # "l_days_since_registration_upper"
)

run_alt_treatment_loop <- function(df) {
    for (treatment in alt_treatments) {
        df_copy <- df
        df_copy$meetings <- df_copy[[treatment]]

        m_ddd_ppml_treatment <- fepois(
            full_formula,
            data    = df_copy,
            cluster = ~cl_dt
        )

        m_ddd_ppml_treatment_squared <- fepois(
            full_formula_squared,
            data    = df_copy,
            cluster = ~cl_dt
        )

        results_alt_treatments[[treatment]] <- m_ddd_ppml_treatment
        results_alt_treatments[[paste(treatment, "Squared")]] <- m_ddd_ppml_treatment_squared
    }

    return(results_alt_treatments)
}

results_alt_treatments <- run_alt_treatment_loop(df)

modelsummary::msummary(
  results_alt_treatments, 
  gof_omit = "IC|Log|Adj|Pseudo|Within", 
  coef_omit = "meps_", 
  stars = TRUE
  # output = file.path(tables_dir, "tab_treatments_overall.tex")
)

df_treat_overall <- data.frame(treatment = character(), estimate = numeric(), se = numeric(), b_squared = numeric(), se_squared = numeric(), stringsAsFactors = FALSE)

# Helper to extract coefficient and SE for `meetings`
extract_meetings <- function(m) {
    cf <- coef(m)
    if (!("meetings" %in% names(cf))) {
        return(list(b = NA_real_, se = NA_real_, b_squared = NA_real_, se_squared = NA_real_))
    }
    b <- unname(cf["meetings"])
    b_squared <- unname(cf["I(meetings^2)"])
    V <- try(vcov(m), silent = TRUE)
    se <- NA_real_
    se_squared <- NA_real_
    if (!inherits(V, "try-error") %in% colnames(V)) {
        se <- sqrt(V["meetings", "meetings"])
        if ("I(meetings^2)" %in% colnames(V)) {
          se_squared <- sqrt(V["I(meetings^2)", "I(meetings^2)"])
        }
    }
    list(b = b, se = se, b_squared = b_squared, se_squared = se_squared)
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

for (nm in names(results_alt_treatments)) {
    fit <- results_alt_treatments[[nm]]
    x <- extract_meetings(fit)
    if (is.na(x$b)) next
    if (nm == "Geral") next
    if (nm == "Geral Squared") next
    df_treat_overall <- rbind(df_treat_overall, data.frame(treatment = nm, estimate = x$b, se = x$se, b_squared = x$b_squared, se_squared = x$se_squared))
}

# Baseline estimate for reference line
baseline_est <- NA_real_
baseline_est_se <- NA_real_
if ("meetings" %in% names(coef(m_ddd_ppml))) {
    baseline_est <- unname(coef(m_ddd_ppml)["meetings"])
    baseline_est_se <- sqrt(vcov(m_ddd_ppml)["meetings", "meetings"])
}

if (nrow(df_treat_overall) > 0) {
    df_treat_overall$ci_lo <- df_treat_overall$estimate - 1.96 * df_treat_overall$se
    df_treat_overall$ci_hi <- df_treat_overall$estimate + 1.96 * df_treat_overall$se
    df_treat_overall$treatment_label <- pretty_treatment_label(df_treat_overall$treatment)
    df_treat_overall$treatment_label <- factor(df_treat_overall$treatment_label, levels = df_treat_overall$treatment_label[order(df_treat_overall$estimate)])

    p_treat_overall <- ggplot(df_treat_overall, aes(x = estimate, y = treatment_label)) +
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
            # title = "Meetings effect across treatments (overall PPML)",
            # subtitle = "Points are estimates; bars are 95% CIs.",
            x = "Efeito (meetings)", y = "Tratamento"
        ) +
        theme_minimal()

    ggsave(file.path(figures_dir, "fig_coeff_treatments_overall.png"), p_treat_overall, width = 10, height = 7, dpi = 300)
    ggsave(file.path(figures_dir, "fig_coeff_treatments_overall.pdf"), p_treat_overall, width = 10, height = 7)
}

#===========================================
# 3)Estimate the number of meetings
#===========================================

df_meetings <- read.csv("./data/silver/df_meetings_lobbyists.csv", stringsAsFactors = TRUE)

# Group by lobbyist_id and member_id to count meetings
df_grouped <- aggregate(
  x = list(meetings = df_meetings$lobbyist_id),
  by = list(lobbyist_id = df_meetings$lobbyist_id),
  FUN = length
)

# Df for lobbyist data (get unique lobbyist characteristics)
df_lobbyists <- df_meetings[!duplicated(df_meetings$lobbyist_id), ]

# Merge meeting counts with lobbyist characteristics
df_merged <- merge(df_grouped, df_lobbyists, by = "lobbyist_id")

# Set 'NGOs' as the reference category to compare against
df_merged$l_category <- relevel(df_merged$l_category, ref = "NGOs")

# --- Clean predictors to avoid NA/NaN/Inf in model matrix ---
# Ensure numeric controls are numeric and finite
num_controls <- c(
  "l_ln_max_budget",
  "l_agriculture", "l_economics_and_trade", "l_education",
  "l_environment_and_climate", "l_foreign_and_security_affairs",
  "l_health", "l_human_rights", "l_infrastructure_and_industry", "l_technology"
)
for (nm in num_controls) {
  if (!nm %in% names(df_merged)) next
  df_merged[[nm]] <- suppressWarnings(as.numeric(df_merged[[nm]]))
  df_merged[[nm]][!is.finite(df_merged[[nm]])] <- NA
}
# Meetings should be integer and finite
df_merged$meetings <- as.integer(df_merged$meetings)

# Handle factor with missing values
df_merged$l_head_office_country <- as.character(df_merged$l_head_office_country)
df_merged$l_head_office_country[is.na(df_merged$l_head_office_country) | df_merged$l_head_office_country == ""] <- "Unknown"
df_merged$l_head_office_country <- as.factor(df_merged$l_head_office_country)

# Drop unused levels in l_category and ensure factor
df_merged$l_category <- droplevels(as.factor(df_merged$l_category))

# Build model frame and drop rows with any missing values in model vars
model_vars <- c("meetings", "l_category", num_controls, "l_head_office_country")
model_vars <- model_vars[model_vars %in% names(df_merged)]
rows_before <- nrow(df_merged)
model_df <- df_merged[complete.cases(df_merged[, model_vars, drop = FALSE]), ]
rows_after <- nrow(model_df)
cat(sprintf("Rows before cleaning: %d; after cleaning: %d (removed %d)\n", rows_before, rows_after, rows_before - rows_after))

## --- Model 1: Negative Binomial Regression to model meeting frequency ---
## This is preferred over lm for count data.
## The coefficient for 'l_categoryBusiness' will tell us the log-count difference
## in meetings compared to NGOs, holding other variables constant.
model_nb <- glm.nb(
  meetings ~ l_category + l_ln_max_budget + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

model_nb_interaction <- glm.nb(
  meetings ~ l_category + l_ln_max_budget + l_category:l_ln_max_budget + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

# Print model summary
msummary(
  list("NB meetings model" = model_nb, "NB meetings model interaction" = model_nb_interaction), 
  stars = TRUE, 
  gof_omit = "AIC|BIC|Log.Lik.",
  coef_omit = "l_head_office",
)

# Export model summary to LaTeX
msummary(
  list("NB meetings model" = model_nb, "NB meetings model interaction" = model_nb_interaction), 
  stars = TRUE,
  gof_omit = "AIC|BIC|Log.Lik.",
  output = file.path("Tese", "tables", "h2_test", "tab_meetings_nb.tex")
)

# Quick descriptive check: average meetings by category
avg_meetings_by_cat <- aggregate(meetings ~ l_category, data = model_df, FUN = mean)
print(avg_meetings_by_cat)

# -------------------------------------------
# Effect plots
# -------------------------------------------

# --- Center ln(max budget) for clean interpretation at average budget ---
mean_ln_budget <- mean(model_df$l_ln_max_budget, na.rm = TRUE)
model_df$l_ln_max_budget_c <- model_df$l_ln_max_budget - mean_ln_budget

model_nb_interaction_c <- glm.nb(
  meetings ~ l_category + l_ln_max_budget_c + l_category:l_ln_max_budget_c + l_agriculture + l_economics_and_trade + l_education + l_environment_and_climate + l_foreign_and_security_affairs + l_health + l_human_rights + l_infrastructure_and_industry + l_technology + l_head_office_country,
  data = model_df
)

# Export centered model table
msummary(
  list(
    "NB meetings model" = model_nb, 
    "NB meetings model interaction" = model_nb_interaction,
    "NB meetings model (centered)" = model_nb_interaction_c),
  stars = TRUE,
  gof_omit = "AIC|BIC|Log.Lik.",
  coef_omit = "l_head_office",
  output = file.path(tables_dir, "tab_meetings_nb_centered.tex")
)

# Build grid for centered model predictions
budget_seq <- seq(
  quantile(model_df$l_ln_max_budget, 0.05, na.rm = TRUE),
  quantile(model_df$l_ln_max_budget, 0.98, na.rm = TRUE),
  length.out = 60
)
budget_seq
newdata_center <- expand.grid(
  l_category = droplevels(unique(model_df$l_category)),
  l_ln_max_budget_c = budget_seq - mean_ln_budget
)
# Set controls at means / modal for clean predictions
newdata_center$l_agriculture <- mean(model_df$l_agriculture, na.rm = TRUE)
newdata_center$l_economics_and_trade <- mean(model_df$l_economics_and_trade, na.rm = TRUE)
newdata_center$l_education <- mean(model_df$l_education, na.rm = TRUE)
newdata_center$l_environment_and_climate <- mean(model_df$l_environment_and_climate, na.rm = TRUE)
newdata_center$l_foreign_and_security_affairs <- mean(model_df$l_foreign_and_security_affairs, na.rm = TRUE)
newdata_center$l_health <- mean(model_df$l_health, na.rm = TRUE)
newdata_center$l_human_rights <- mean(model_df$l_human_rights, na.rm = TRUE)
newdata_center$l_infrastructure_and_industry <- mean(model_df$l_infrastructure_and_industry, na.rm = TRUE)
newdata_center$l_technology <- mean(model_df$l_technology, na.rm = TRUE)
mode_country <- names(which.max(table(model_df$l_head_office_country)))
newdata_center$l_head_office_country <- factor(mode_country, levels = levels(model_df$l_head_office_country))

# Predictions with standard errors on response scale
pred_center <- predict(model_nb_interaction_c, newdata = newdata_center, type = "link", se.fit = TRUE)
newdata_center$mean_meetings <- exp(pred_center$fit)
newdata_center$ci_lo <- exp(pred_center$fit - 1.96 * pred_center$se.fit)
newdata_center$ci_hi <- exp(pred_center$fit + 1.96 * pred_center$se.fit)
newdata_center$l_ln_max_budget <- newdata_center$l_ln_max_budget_c + mean_ln_budget

# Save CSV
fwrite(newdata_center, file.path(outputs_dir, "pred_meetings_centered_vs_budget_by_category.csv"))

# Plot expected meetings vs ln(budget) with CIs
p_meet_vs_budget_c <- ggplot(newdata_center, aes(x = l_ln_max_budget, y = mean_meetings, color = l_category, fill = l_category)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, color = NA) +
  geom_vline(xintercept = mean_ln_budget, linetype = "dashed", color = "gray60") +
  labs(x = "ln(orçamento máximo)", y = "Reuniões previstas", color = "Categoria", fill = "Categoria",
       subtitle = "Linha pontilhada: orçamento médio (escala log)") +
  theme_minimal()

ggsave(file.path(figures_dir, "fig_pred_meetings_vs_budget_centered_by_category.png"), p_meet_vs_budget_c, width = 9, height = 5.5, dpi = 300)

ggsave(file.path(figures_dir, "fig_pred_meetings_vs_budget_centered_by_category.pdf"), p_meet_vs_budget_c, width = 9, height = 5.5)

# Ratio curve: Business / NGOs expected meetings vs ln(budget)
lvls <- levels(model_df$l_category)
if (all(c("NGOs", "Business") %in% lvls)) {
  wide <- reshape(newdata_center[, c("l_category", "l_ln_max_budget", "mean_meetings", "ci_lo", "ci_hi")],
                  idvar = "l_ln_max_budget", timevar = "l_category", direction = "wide")
  wide$ratio_mean <- wide$mean_meetings.Business / wide$mean_meetings.NGOs
  # Delta method for CI on ratio (approx via log scale)
  # Here, we approximate using bounds: conservative envelope
  wide$ratio_lo <- (wide$ci_lo.Business / wide$ci_hi.NGOs)
  wide$ratio_hi <- (wide$ci_hi.Business / wide$ci_lo.NGOs)

  # Compute crossover lnB* where ratio = 1 using coefficients
  coefs <- coef(model_nb_interaction_c)
  b_biz <- unname(coefs["l_categoryBusiness"])
  b_int <- unname(coefs["l_categoryBusiness:l_ln_max_budget_c"])
  lnB_star_c <- if (!is.na(b_biz) && !is.na(b_int) && b_int != 0) -b_biz / b_int else NA_real_
  lnB_star <- lnB_star_c + mean_ln_budget

  ratio_df <- wide
  fwrite(ratio_df, file.path(outputs_dir, "ratio_business_over_ngo_vs_budget_centered.csv"))

  p_ratio <- ggplot(ratio_df, aes(x = l_ln_max_budget, y = ratio_mean)) +
    geom_hline(yintercept = 1, color = "gray70") +
    geom_line(color = "#9467bd", size = 1) +
    geom_ribbon(aes(ymin = ratio_lo, ymax = ratio_hi), fill = "#9467bd", alpha = 0.15) +
    { if (is.finite(lnB_star)) geom_vline(xintercept = lnB_star, linetype = "dotted", color = "#B45C1F") else NULL } +
    labs(x = "ln(orçamento máximo)", y = "Razão Business/NGOs (reuniões previstas)",
         subtitle = "Linha horizontal: paridade (1). Linha pontilhada: ponto de cruzamento estimado") +
    theme_minimal()

  ggsave(file.path(figures_dir, "fig_ratio_business_over_ngo_vs_budget_centered.png"), p_ratio, width = 9, height = 5.5, dpi = 300)
  ggsave(file.path(figures_dir, "fig_ratio_business_over_ngo_vs_budget_centered.pdf"), p_ratio, width = 9, height = 5.5)
}

#-------------------------------------------
# Add the ppml coefficients for the interaction term
#-------------------------------------------

# -------------------------------------------
# Total effect vs budget = predicted meetings × per-meeting effect (PPML)
# Using in-memory df_treat_overall (columns: treatment, estimate, se, ci_lo, ci_hi)
# -------------------------------------------

# Expect df_treat_overall to have rows for l_category_Business, l_category_NGOs, l_category_Other
pm_map <- df_treat_overall[df_treat_overall$treatment %in% c("l_category_Business", "l_category_NGOs", "l_category_Other"),
                            c("treatment", "estimate", "se", "ci_lo", "ci_hi")]
# Map to category labels used in predictions
pm_map$l_category <- ifelse(pm_map$treatment == "l_category_Business", "Business",
                        ifelse(pm_map$treatment == "l_category_NGOs", "NGOs",
                        ifelse(pm_map$treatment == "l_category_Other", "Other", pm_map$treatment)))
pm_map <- pm_map[, c("l_category", "estimate", "se", "ci_lo", "ci_hi")]
colnames(pm_map) <- c("l_category", "per_meeting_coef", "per_meeting_se", "per_meeting_lo", "per_meeting_hi")

te_df <- merge(newdata_center[, c("l_category", "l_ln_max_budget", "mean_meetings", "ci_lo", "ci_hi")],
                pm_map, by = "l_category", all.x = TRUE)
te_df$total_effect <- te_df$mean_meetings * te_df$per_meeting_coef
te_df$total_lo <- te_df$ci_lo * te_df$per_meeting_coef
te_df$total_hi <- te_df$ci_hi * te_df$per_meeting_coef

# Save CSV
fwrite(te_df, file.path(outputs_dir, "total_effect_vs_budget_by_category.csv"))

# Plot total effect vs ln(budget)
p_te <- ggplot(te_df, aes(x = l_ln_max_budget, y = total_effect, color = l_category, fill = l_category)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = total_lo, ymax = total_hi), alpha = 0.15, color = NA) +
  labs(x = "ln(orçamento máximo)", y = "Efeito total esperado (reuniões × efeito por reunião)", color = "Categoria", fill = "Categoria") +
  theme_minimal() 

ggsave(file.path(figures_dir, "fig_total_effect_vs_budget_by_category.png"), p_te, width = 9, height = 5.5, dpi = 300)
ggsave(file.path(figures_dir, "fig_total_effect_vs_budget_by_category.pdf"), p_te, width = 9, height = 5.5)

# Optional: ratio Business/NGOs
if (all(c("NGOs", "Business") %in% unique(te_df$l_category))) {
  te_wide <- reshape(te_df[te_df$l_category %in% c("NGOs", "Business"),
                            c("l_ln_max_budget", "l_category", "total_effect", "total_lo", "total_hi")],
                      idvar = "l_ln_max_budget", timevar = "l_category", direction = "wide")
  te_wide$ratio_te <- te_wide$total_effect.Business / te_wide$total_effect.NGOs
  te_wide$ratio_te_lo <- te_wide$total_lo.Business / te_wide$total_hi.NGOs
  te_wide$ratio_te_hi <- te_wide$total_hi.Business / te_wide$total_lo.NGOs

  fwrite(te_wide, file.path(outputs_dir, "ratio_total_effect_business_over_ngo_vs_budget.csv"))

  p_ratio_te <- ggplot(te_wide, aes(x = l_ln_max_budget, y = ratio_te)) +
    geom_hline(yintercept = 1, color = "gray70") +
    geom_line(color = "#e377c2", size = 1) +
    geom_ribbon(aes(ymin = ratio_te_lo, ymax = ratio_te_hi), fill = "#e377c2", alpha = 0.15) +
    labs(x = "ln(orçamento máximo)", y = "Razão Business/NGOs (efeito total)") +
    theme_minimal()

  ggsave(file.path(figures_dir, "fig_ratio_total_effect_business_over_ngo_vs_budget.png"), p_ratio_te, width = 9, height = 5.5, dpi = 300)
  ggsave(file.path(figures_dir, "fig_ratio_total_effect_business_over_ngo_vs_budget.pdf"), p_ratio_te, width = 9, height = 5.5)
}

