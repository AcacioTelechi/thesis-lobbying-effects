# ============================================
# PPML
# ============================================

# install.packages("fixest") # if needed
library(fixest)
library(modelsummary)
library(ggplot2)


# Create output directories
figures_dir <- file.path("Tese", "figures", "event_study")
tables_dir <- file.path("Tese", "tables", "event_study")
outputs_dir <- file.path("outputs", "event_study")

for (dir in c(figures_dir, tables_dir, outputs_dir)) {
    if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}


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



# =============================
# Staggered-adoption event study (Sun & Abraham) + pre-trends test
# =============================

# Build calendar time from Y.m
if (is.factor(df$Y.m)) df$Y.m <- as.character(df$Y.m)
if (is.numeric(df$Y.m)) {
    year_part <- floor(df$Y.m / 100)
    month_part <- df$Y.m %% 100
    df$time_fe <- as.Date(sprintf("%04d-%02d-01", year_part, month_part))
} else if (is.character(df$Y.m)) {
    if (any(grepl("-", df$Y.m), na.rm = TRUE)) {
        df$time_fe <- as.Date(paste0(df$Y.m, "-01"))
    } else {
        year_part <- suppressWarnings(as.integer(substr(df$Y.m, 1, 4)))
        month_part <- suppressWarnings(as.integer(substr(df$Y.m, 5, 6)))
        df$time_fe <- as.Date(sprintf("%04d-%02d-01", year_part, month_part))
    }
} else if (inherits(df$Y.m, "Date")) {
    df$time_fe <- df$Y.m
} else {
    stop("Unsupported Y.m format for building time_fe")
}

# Quarter-level time and yearly cohorts; sample never-treated to reduce size
df$time_q <- as.Date(cut(df$time_fe, "quarter"))
# update FE and clustering to quarter time
df$fe_dt <- interaction(df$domain, df$time_q, drop = TRUE)
df$cl_dt <- interaction(df$domain, df$time_q, drop = TRUE)

# configurable sampling of never-treated units (keep all treated)
never_sample_frac <- 1
set.seed(123)

## create a agregated by mep to find never treated
tmp_has_meeting <- df$meetings > 0 & !is.na(df$meetings)
by_member_any <- aggregate(tmp_has_meeting ~ member_id, data = df, FUN = function(x) any(x, na.rm = TRUE))
by_member_any$member_id <- as.character(by_member_any$member_id)
treated_ids <- by_member_any$member_id[by_member_any$tmp_has_meeting]
never_ids <- by_member_any$member_id[!by_member_any$tmp_has_meeting]
if (length(never_ids) > 0 && never_sample_frac < 1) {
    n_keep <- max(1L, ceiling(length(never_ids) * never_sample_frac))
    never_ids <- sample(never_ids, size = n_keep, replace = FALSE)
}
keep_ids <- unique(c(treated_ids, never_ids))
df <- df[as.character(df$member_id) %in% keep_ids, , drop = FALSE]

# cohort based on first quarter with meetings > 0
tmp_has_meeting <- df$meetings > 0 & !is.na(df$meetings)
first_treat_df <- aggregate(time_q ~ member_id, data = df[tmp_has_meeting, ], FUN = min)
first_treat_df$member_id <- as.character(first_treat_df$member_id)
df$cohort_q <- first_treat_df$time_q[match(as.character(df$member_id), first_treat_df$member_id)]
# yearly cohort grouping
df$cohort_y <- as.Date(paste0(format(df$cohort_q, "%Y"), "-01-01"))

# Sun & Abraham event-study with restricted leads/lags [-3,3], ref period -1
sa_formula_restricted <- as.formula(paste0("questions ~ sunab(cohort_y, time_q, ref.p = -1, bin.rel = -2:2) + ", controls_str, " | fe_i + fe_dt + fe_ct + fe_pt"))

m_sa_ppml_restricted <- fepois(
    sa_formula_restricted,
    data    = df,
    cluster = ~cl_dt
)

# Joint pre-trends Wald test (robust selection via regex; safe if not found)
try({
    wt <- wald(m_sa_ppml_restricted, keep = "sunab.*::-3$")
    print(wt)
}, silent = TRUE)

## Plot event-study using fixest's iplot and save via devices (robust ggplot/base)
# PNG
png(file.path(figures_dir, "fig_parallel_trends_event_study_qtr_year.png"), width = 10, height = 6, units = "in", res = 300)
iplot(m_sa_ppml_restricted, ref.line = 0)
title(main = "Event study with staggered adoption (PPML, quarter time, yearly cohorts)")
dev.off()
# PDF
pdf(file.path(figures_dir, "fig_parallel_trends_event_study_qtr_year.pdf"), width = 10, height = 6)
iplot(m_sa_ppml_restricted, ref.line = 0)
title(main = "Event study with staggered adoption (PPML, quarter time, yearly cohorts)")
dev.off()

# Also print to interactive session if available
try(iplot(m_sa_ppml_restricted, ref.line = 0, keep="sunab.*::-3$"), silent = TRUE)

# Prepare data for ggplot event-study plot
# Extract Sun & Abraham coefficients and confidence intervals
sa_coefs <- broom::tidy(m_sa_ppml_restricted, conf.int = TRUE)
# Filter for event-study terms (sunab), extract event time from term
sa_coefs <- sa_coefs[grepl("^time_q::", sa_coefs$term), ]
sa_coefs$event_time <- as.numeric(sub("time_q::(-?\\d+).*", "\\1", sa_coefs$term))

# Order by event_time
sa_coefs <- sa_coefs[order(sa_coefs$event_time), ]
sa_coefs <- sa_coefs[sa_coefs$event_time >= -7 & sa_coefs$event_time <= 7, ]


# Plot with ggplot2
p_event_study <- ggplot(sa_coefs, aes(x = event_time, y = estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  geom_point(size = 2, color = "#1f77b4") +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2, color = "#1f77b4") +
  labs(
    x = "Event Time (Quarters relative to treatment)",
    y = "Estimated Effect on Questions (log count)",
    title = "Event Study: Effect of Lobbying Meetings on Parliamentary Questions",
    subtitle = "Sun & Abraham (2020) estimator, PPML, ref = -1 quarter",
    caption = "Estimates with 95% confidence intervals"
  ) +
  theme_minimal(base_size = 12)

# Save plot to thesis figures directory
ggsave(file.path(figures_dir, "fig_parallel_trends_event_study_qtr_year_ggplot.pdf"), p_event_study, width = 10, height = 6)
ggsave(file.path(figures_dir, "fig_parallel_trends_event_study_qtr_year_ggplot.png"), p_event_study, width = 10, height = 6, dpi = 300)

# Analysis for the event-study plot:
cat("\n=== Event-Study Analysis ===\n")
cat("The ggplot-based event-study plot visualizes the estimated effect of lobbying meetings on parliamentary questions across event time.\n")
cat("Pre-treatment periods (event time < 0) allow for visual inspection of parallel trends, while post-treatment periods (event time >= 0) show the dynamic treatment effects.\n")
cat("Confidence intervals provide a sense of statistical uncertainty around each estimate.\n")
