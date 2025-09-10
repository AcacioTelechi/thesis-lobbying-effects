# ============================================
# LEADS AND LAGS ANALYSIS - H1 PPML MODEL
# ============================================
# This script implements leads and lags tests for the main PPML model
# to examine anticipation effects and persistence of lobbying impacts

# Load required libraries
library(fixest)
library(modelsummary)
library(ggplot2)
library(dplyr)

# Create output directories
figures_dir <- file.path("Tese", "figures", "leads_lags")
tables_dir <- file.path("Tese", "tables", "leads_lags")
outputs_dir <- file.path("outputs", "leads_lags")

for (dir in c(figures_dir, tables_dir, outputs_dir)) {
    if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}

# Load data
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# Prepare data
df$Y.m <- as.Date(paste0(df$Y.m, "-01"))
df$member_id <- as.factor(df$member_id)
df$domain <- as.factor(df$domain)

# Sort data by member_id, domain, and time to ensure proper ordering
df <- df %>%
    arrange(member_id, domain, Y.m)

# ============================================
# CREATE LEADS AND LAGS VARIABLES
# ============================================

print("Creating leads and lags variables...")

# Function to create leads and lags for a given variable
create_leads_lags <- function(data, var_name, n_periods = 3) {
    data <- data %>%
        group_by(member_id, domain) %>%
        arrange(Y.m) %>%
        mutate(
            # Leads (anticipation effects)
            !!paste0(var_name, "_lead3") := lead(!!sym(var_name), 3),
            !!paste0(var_name, "_lead2") := lead(!!sym(var_name), 2),
            !!paste0(var_name, "_lead1") := lead(!!sym(var_name), 1),
            
            # Current period (reference)
            !!paste0(var_name, "_current") := !!sym(var_name),
            
            # Lags (persistence effects)
            !!paste0(var_name, "_lag1") := lag(!!sym(var_name), 1),
            !!paste0(var_name, "_lag2") := lag(!!sym(var_name), 2),
            !!paste0(var_name, "_lag3") := lag(!!sym(var_name), 3)
        ) %>%
        ungroup()
    
    return(data)
}

# Create leads and lags for meetings variable
df <- create_leads_lags(df, "meetings", n_periods = 3)

# ============================================
# BUILD FIXED EFFECTS STRUCTURE
# ============================================

# Create fixed effects identifiers (consistent with h1.R)
df$fe_i <- df$member_id # μ_id (member fixed effects)
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE) # country × time
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE) # party × time  
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE) # domain × time

# Cluster variable for two-way clustering
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)

# ============================================
# CONTROLS SPECIFICATION
# ============================================

# Political groups controls (consistent with h1.R)
political_controls <- grep("meps_POLITICAL_GROUP", names(df), value = TRUE)
country_controls <- grep("meps_COUNTRY", names(df), value = TRUE) 
committee_controls <- grep("meps_COMMITTEE", names(df), value = TRUE)

# Combine all controls
controls <- c(political_controls, country_controls, committee_controls)
controls <- controls[controls %in% names(df)]  # Only include existing columns

controls_str <- paste(controls, collapse = " + ")

print(paste("Using", length(controls), "control variables"))

# ============================================
# LEADS AND LAGS MODEL ESTIMATION
# ============================================

print("Estimating leads and lags models...")

# Model 1: Full leads and lags specification
formula_full_leads_lags <- as.formula(paste0(
    "questions ~ meetings_lead3 + meetings_lead2 + meetings_lead1 + ",
    "meetings_current + meetings_lag1 + meetings_lag2 + meetings_lag3",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_leads_lags_full <- fepois(
    formula_full_leads_lags,
    data = df,
    cluster = ~cl_dt
)

# Model 2: Limited leads and lags (±2 periods)
formula_limited_leads_lags <- as.formula(paste0(
    "questions ~ meetings_lead2 + meetings_lead1 + ",
    "meetings_current + meetings_lag1 + meetings_lag2",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_leads_lags_limited <- fepois(
    formula_limited_leads_lags,
    data = df,
    cluster = ~cl_dt
)

# Model 3: Only leads (test for anticipation)
formula_leads_only <- as.formula(paste0(
    "questions ~ meetings_lead3 + meetings_lead2 + meetings_lead1 + meetings_current",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_leads_only <- fepois(
    formula_leads_only,
    data = df,
    cluster = ~cl_dt
)

# Model 4: Only lags (test for persistence)  
formula_lags_only <- as.formula(paste0(
    "questions ~ meetings_current + meetings_lag1 + meetings_lag2 + meetings_lag3",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_lags_only <- fepois(
    formula_lags_only,
    data = df,
    cluster = ~cl_dt
)

# ============================================
# EXTRACT COEFFICIENTS FOR EVENT STUDY PLOT
# ============================================

print("Extracting coefficients for event study plot...")

extract_coef_se <- function(model, coef_name) {
    if (coef_name %in% names(coef(model))) {
        coef_val <- coef(model)[coef_name]
        se_val <- sqrt(diag(vcov(model)))[coef_name]
        return(c(coef_val, se_val))
    } else {
        return(c(NA, NA))
    }
}

# Extract coefficients from full model
event_study_data <- data.frame(
    period = c(-3, -2, -1, 0, 1, 2, 3),
    coefficient = rep(NA, 7),
    se = rep(NA, 7),
    ci_lower = rep(NA, 7),
    ci_upper = rep(NA, 7)
)

# Extract coefficients
coef_names <- c("meetings_lead3", "meetings_lead2", "meetings_lead1", 
                "meetings_current", "meetings_lag1", "meetings_lag2", "meetings_lag3")

for (i in 1:7) {
    if (!is.na(coef_names[i])) {
        result <- extract_coef_se(m_leads_lags_full, coef_names[i])
        event_study_data$coefficient[i] <- result[1]
        event_study_data$se[i] <- result[2]
        if (!is.na(result[2])) {
            event_study_data$ci_lower[i] <- result[1] - 1.96 * result[2]
            event_study_data$ci_upper[i] <- result[1] + 1.96 * result[2]
        }
    }
}

# Remove rows with missing coefficients
event_study_data <- event_study_data[!is.na(event_study_data$coefficient), ]

print("Event study coefficients:")
print(event_study_data)

# ============================================
# CREATE EVENT STUDY PLOT
# ============================================

print("Creating event study plot...")

if (nrow(event_study_data) > 0) {
    p_event_study <- ggplot(event_study_data, aes(x = period, y = coefficient)) +
        geom_hline(yintercept = 0, color = "gray70", linetype = "dashed") +
        geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", alpha = 0.5) +
        geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                    fill = "#1f77b4", alpha = 0.2, na.rm = TRUE) +
        geom_line(color = "#1f77b4", size = 1, na.rm = TRUE) +
        geom_point(color = "#1f77b4", size = 3, na.rm = TRUE) +
        scale_x_continuous(
            breaks = seq(-3, 3, 1),
            labels = c("-3", "-2", "-1", "0", "+1", "+2", "+3"),
            name = "Períodos relativos ao tratamento"
        ) +
        scale_y_continuous(name = "Coeficiente (log points)") +
        labs(
            title = "Event Study: Efeitos Dinâmicos do Lobbying",
            subtitle = paste0(
                "Modelo PPML com efeitos fixos de país×tempo, partido×tempo, domínio×tempo\n",
                "Período 0 = tratamento contemporâneo; períodos negativos = antecipação; períodos positivos = persistência"
            ),
            caption = "Nota: Faixas representam IC 95% com clustering por domínio×tempo"
        ) +
        theme_minimal(base_size = 12) +
        theme(
            plot.title = element_text(size = 14, face = "bold"),
            plot.subtitle = element_text(size = 10),
            axis.title = element_text(size = 11),
            legend.position = "none",
            panel.grid.minor = element_blank()
        )

    # Save plot
    ggsave(file.path(figures_dir, "event_study_leads_lags.pdf"), p_event_study, 
           width = 10, height = 6)
    ggsave(file.path(figures_dir, "event_study_leads_lags.png"), p_event_study, 
           width = 10, height = 6, dpi = 300)
    
    print(paste("Event study plot saved to:", figures_dir))
}

# ============================================
# HYPOTHESIS TESTS
# ============================================

print("Conducting hypothesis tests...")

# Test 1: No anticipation effects (all leads = 0)
if (all(c("meetings_lead1", "meetings_lead2", "meetings_lead3") %in% names(coef(m_leads_lags_full)))) {
    test_no_anticipation <- wald(m_leads_lags_full, 
                                c("meetings_lead1", "meetings_lead2", "meetings_lead3"))
    print("Test for no anticipation effects:")
    print(test_no_anticipation)
}

# Test 2: No persistence effects (all lags = 0) 
if (all(c("meetings_lag1", "meetings_lag2", "meetings_lag3") %in% names(coef(m_leads_lags_full)))) {
    test_no_persistence <- wald(m_leads_lags_full, 
                               c("meetings_lag1", "meetings_lag2", "meetings_lag3"))
    print("Test for no persistence effects:")
    print(test_no_persistence)
}

# Test 3: No dynamic effects (all leads and lags = 0)
dynamic_vars <- c("meetings_lead1", "meetings_lead2", "meetings_lead3",
                  "meetings_lag1", "meetings_lag2", "meetings_lag3")
existing_dynamic_vars <- dynamic_vars[dynamic_vars %in% names(coef(m_leads_lags_full))]

if (length(existing_dynamic_vars) > 0) {
    test_no_dynamics <- wald(m_leads_lags_full, existing_dynamic_vars)
    print("Test for no dynamic effects:")
    print(test_no_dynamics)
}

# ============================================
# SUMMARY TABLE
# ============================================

print("Creating summary tables...")

# Main results table
models_list <- list(
    "Leads Only" = m_leads_only,
    "Lags Only" = m_lags_only, 
    "Full Model" = m_leads_lags_full
)

# Create summary table
msummary(
    models_list,
    coef_omit = "meps_",  # Omit control variables for clarity
    gof_omit = "IC|Log|Adj|Pseudo|Within",
    stars = TRUE,
    title = "Leads and Lags Analysis - PPML Results"
)

# Export to LaTeX
msummary(
    models_list,
    coef_omit = "meps_",
    gof_omit = "IC|Log|Adj|Pseudo|Within", 
    stars = TRUE,
    output = file.path(tables_dir, "leads_lags_main.tex"),
    title = "Leads and Lags Analysis - PPML Results"
)

# ============================================
# ROBUSTNESS: ALTERNATIVE SPECIFICATIONS
# ============================================

print("Running robustness checks...")

# Robustness 1: Different clustering structure
m_robust_cluster <- fepois(
    formula_full_leads_lags,
    data = df,
    cluster = ~member_id  # Cluster by member instead
)

# Robustness 2: OLS specification for comparison
formula_ols_leads_lags <- as.formula(paste0(
    "questions ~ meetings_lead2 + meetings_lead1 + ",
    "meetings_current + meetings_lag1 + meetings_lag2",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_ols_leads_lags <- feols(
    formula_ols_leads_lags,
    data = df,
    cluster = ~cl_dt
)

# Robustness table
robustness_models <- list(
    "PPML Baseline" = m_leads_lags_limited,
    "PPML (Member Cluster)" = m_robust_cluster,
    "OLS" = m_ols_leads_lags
)

msummary(
    robustness_models,
    coef_omit = "meps_",
    gof_omit = "IC|Log|Adj|Pseudo|Within",
    stars = TRUE,
    output = file.path(tables_dir, "leads_lags_robustness.tex"),
    title = "Leads and Lags - Robustness Checks"
)

# ============================================
# EXPORT RESULTS SUMMARY
# ============================================

# Create results summary
results_summary <- list(
    event_study_data = event_study_data,
    models = models_list,
    n_obs = nobs(m_leads_lags_full),
    n_members = length(unique(df$member_id)),
    n_domains = length(unique(df$domain)),
    time_range = paste(min(df$Y.m), "to", max(df$Y.m))
)

# Save results
saveRDS(results_summary, file.path(outputs_dir, "leads_lags_results.rds"))

# Print summary
cat("\n" %=% "LEADS AND LAGS ANALYSIS COMPLETED" %=% "\n")
cat("Key findings:\n")
if (nrow(event_study_data) > 0) {
    cat("- Event study coefficients estimated for", nrow(event_study_data), "periods\n")
    cat("- Current period coefficient:", round(event_study_data$coefficient[event_study_data$period == 0], 4), "\n")
}
cat("- Sample size:", results_summary$n_obs, "observations\n")
cat("- Time range:", results_summary$time_range, "\n")
cat("- Figures saved to:", figures_dir, "\n")
cat("- Tables saved to:", tables_dir, "\n")

print("Analysis completed successfully!")
