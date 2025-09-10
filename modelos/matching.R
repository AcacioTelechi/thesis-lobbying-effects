# =============================================================
# Pre-trend Matching Test Pipeline
# -------------------------------------------------------------
# Goal: Construct a proof-of-concept pipeline that matches treated
#       units (first month with meetings > 0) to control units based
#       on pre-treatment outcome trends (questions) and key covariates.
#
# Inputs:
#   - ./data/gold/df_long_v2.csv (panel: member_id × domain × Y.m)
#
# Outputs (created if missing):
#   - Tese/figures/matching/
#   - Tese/tables/matching/
#   - outputs/matching/
# Artifacts saved:
#   - outputs/matching/pretrend_match_pairs.csv
#   - outputs/matching/pretrend_match_summary.rds
#   - Tese/tables/matching/pretrend_balance.txt
#   - Tese/figures/matching/pretrend_love_plot.png
#
# Notes:
#   - This file is intentionally verbose for clarity and auditing.
#   - It is a test scaffold; refine windows/constraints as needed.
# =============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
})

# Declare globals to appease linters in tidy-eval pipelines
utils::globalVariables(c(
  "Y.m","member_id","domain","t0","pre_months","data_pre",
  "anchor","t_rel","questions"
))

install_if_missing <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      message("Installing package: ", p)
      install.packages(p, repos = "https://cloud.r-project.org")
    }
  }
}

install_if_missing(c("MatchIt", "cobalt", "ggplot2"))
library(MatchIt)
library(cobalt)
library(ggplot2)

# -------------------------------------------------------------
# Directories
# -------------------------------------------------------------
figures_dir <- file.path("Tese", "figures", "matching")
tables_dir  <- file.path("Tese", "tables", "matching")
outputs_dir <- file.path("outputs", "matching")

for (dir in c(figures_dir, tables_dir, outputs_dir)) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}

# -------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------
message("Loading data ...")
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = FALSE)

# Ensure expected columns exist
required_cols <- c("member_id", "domain", "Y.m", "questions", "meetings","meps_country", "meps_party")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- df %>%
  mutate(
    Y.m = as.Date(paste0(Y.m, "-01")),
    member_id = as.factor(member_id),
    domain = as.factor(domain),
    meps_country = as.factor(meps_country),
    meps_party = as.factor(meps_party)
  ) %>%
  arrange(member_id, domain, Y.m)

# First treatment month per unit (member × domain)
message("Computing treatment onset per unit ...")
first_meet <- df %>%
  group_by(member_id, domain) %>%
  summarize(first_meeting_date = suppressWarnings(min(Y.m[meetings > 0], na.rm = TRUE)), .groups = "drop")

# Handle units with no meetings (min on empty returns Inf)
first_meet <- first_meet %>% mutate(
  first_meeting_date = ifelse(is.finite(first_meeting_date), first_meeting_date, as.Date(NA))
) %>% mutate(first_meeting_date = as.Date(first_meeting_date, origin = "1970-01-01"))

df <- df %>% left_join(first_meet, by = c("member_id", "domain"))

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
pre_window_months  <- 6    # length of pre-period used to compute trends
post_buffer_months <- 3    # ensure controls free of meetings shortly after anchor
min_non_missing    <- 4    # require at least this many months to fit slope reliably

# -------------------------------------------------------------
# Helpers (pure base R date arithmetic)
# -------------------------------------------------------------
add_months <- function(date, k) {
  # Robust, vectorized month shifting without external deps
  d <- as.Date(date)
  if (length(k) == 1L) k <- rep(k, length(d))
  y <- as.integer(format(d, "%Y"))
  m <- as.integer(format(d, "%m"))
  m2 <- m + k
  y2 <- y + (m2 - 1L) %/% 12L
  m2 <- ((m2 - 1L) %% 12L) + 1L
  as.Date(sprintf("%04d-%02d-01", y2, m2))
}

seq_months <- function(end_date, k) {
  # Returns vector of months: (end_date - k + 1) ... end_date
  start_date <- add_months(end_date, -(k - 1L))
  sapply(0:(k - 1L), function(i) add_months(start_date, i)) |> as.Date(origin = "1970-01-01")
}

compute_pretrend <- function(panel_slice) {
  # panel_slice must contain columns: Y.m, questions
  panel_slice <- panel_slice %>% arrange(Y.m)
  # time index 1..n within window
  panel_slice <- panel_slice %>% mutate(t_idx = row_number())
  slope <- NA_real_
  if (nrow(panel_slice) >= min_non_missing && sd(panel_slice$t_idx) > 0) {
    fit <- try(lm(questions ~ t_idx, data = panel_slice), silent = TRUE)
    if (!inherits(fit, "try-error")) slope <- coef(fit)[["t_idx"]]
  }
  tibble(
    y_mean_pre = mean(panel_slice$questions, na.rm = TRUE),
    y_sd_pre   = sd(panel_slice$questions, na.rm = TRUE),
    y_last_pre = dplyr::last(panel_slice$questions),
    y_slope_pre = slope
  )
}

no_meetings_in_window <- function(panel, months_vec) {
  all(panel$meetings[panel$Y.m %in% months_vec] == 0 | is.na(panel$meetings[panel$Y.m %in% months_vec]))
}

# -------------------------------------------------------------
# Build treated sample: one row per unit with a valid pre-window
# -------------------------------------------------------------
message("Constructing treated sample with pre-trend features ...")

treated_units <- df %>%
  filter(!is.na(first_meeting_date)) %>%
  group_by(member_id, domain) %>%
  summarize(t0 = first(first_meeting_date), .groups = "drop")

treated_units <- treated_units %>%
  mutate(
    pre_months = map(t0, ~ seq_months(add_months(.x, -1), pre_window_months))
  )

# Attach static covariates at t0-1 for matching (country, party)
treated_covs <- df %>%
  semi_join(treated_units, by = c("member_id", "domain")) %>%
  mutate(anchor = Y.m) %>%
  inner_join(treated_units %>% transmute(member_id, domain, t0), by = c("member_id", "domain")) %>%
  filter(Y.m == add_months(t0, -1)) %>%
  select(member_id, domain, meps_country, meps_party)

treated_pre <- treated_units %>%
  tidyr::unnest_longer(pre_months) %>%
  rename(Y.m = pre_months) %>%
  left_join(df %>% select(member_id, domain, Y.m, questions), by = c("member_id", "domain", "Y.m")) %>%
  group_by(member_id, domain) %>%
  arrange(Y.m, .by_group = TRUE) %>%
  mutate(t_idx = row_number()) %>%
  summarise(
    y_mean_pre = mean(questions, na.rm = TRUE),
    y_sd_pre   = sd(questions, na.rm = TRUE),
    y_last_pre = dplyr::last(questions),
    y_slope_pre = {
      ok <- !is.na(questions) & is.finite(questions)
      n_ok <- sum(ok)
      if (n_ok >= min_non_missing) {
        x <- seq_len(n_ok)
        y <- questions[ok]
        if (var(x) > 0) cov(x, y) / var(x) else NA_real_
      } else NA_real_
    },
    .groups = "drop"
  ) %>%
  left_join(treated_covs, by = c("member_id", "domain")) %>%
  mutate(treated = 1L)

# Drop units without sufficient pre-period information
treated_pre <- treated_pre %>% filter(!is.na(y_slope_pre) & is.finite(y_slope_pre))

message("Treated units with valid pre-period: ", nrow(treated_pre))

# -------------------------------------------------------------
# Build control sample: pick an anchor month per unit that mimics a
# treated anchor but with zero meetings in the pre-window and shortly after
# -------------------------------------------------------------
message("Constructing control sample with pre-trend features ...")

units <- df %>% distinct(member_id, domain)

choose_control_anchor <- function(unit_panel) {
  # Try calendar months in reverse order to maximize data availability
  # Anchor must have: at least pre_window_months months before, zero meetings in
  # that pre-window and zero in the post_buffer_months months after the anchor.
  dates <- sort(unique(unit_panel$Y.m), decreasing = TRUE)
  for (anchor in dates) {
    pre_months <- seq_months(add_months(anchor, -1), pre_window_months)
    post_months <- sapply(0:(post_buffer_months - 1L), function(i) add_months(anchor, i)) |> as.Date(origin = "1970-01-01")
    have_all_pre  <- all(pre_months %in% unit_panel$Y.m)
    have_all_post <- all(post_months %in% unit_panel$Y.m)
    if (!(have_all_pre && have_all_post)) next
    # require zero meetings in pre and at anchor/post buffer
    sel <- unit_panel$Y.m %in% c(pre_months, post_months)
    meetings_vec <- unit_panel$meetings[sel]
    if (all(meetings_vec == 0 | is.na(meetings_vec))) {
      return(anchor)
    }
  }
  return(as.Date(NA))
}

control_anchors <- df %>%
  anti_join(treated_units, by = c("member_id", "domain")) %>%
  group_by(member_id, domain) %>%
  group_modify(~ tibble(anchor = choose_control_anchor(.x))) %>%
  ungroup() %>%
  filter(!is.na(anchor))

control_pre <- control_anchors %>%
  mutate(pre_months = map(anchor, ~ seq_months(add_months(.x, -1), pre_window_months))) %>%
  tidyr::unnest_longer(pre_months) %>%
  rename(Y.m = pre_months) %>%
  left_join(df %>% select(member_id, domain, Y.m, questions), by = c("member_id", "domain", "Y.m")) %>%
  group_by(member_id, domain) %>%
  arrange(Y.m, .by_group = TRUE) %>%
  mutate(t_idx = row_number()) %>%
  summarise(
    y_mean_pre = mean(questions, na.rm = TRUE),
    y_sd_pre   = sd(questions, na.rm = TRUE),
    y_last_pre = dplyr::last(questions),
    y_slope_pre = {
      ok <- !is.na(questions) & is.finite(questions)
      n_ok <- sum(ok)
      if (n_ok >= min_non_missing) {
        x <- seq_len(n_ok)
        y <- questions[ok]
        if (var(x) > 0) cov(x, y) / var(x) else NA_real_
      } else NA_real_
    },
    .groups = "drop"
  )

# Attach covariates at anchor-1
control_covs <- df %>%
  semi_join(control_anchors, by = c("member_id", "domain")) %>%
  inner_join(control_anchors, by = c("member_id", "domain")) %>%
  filter(Y.m == add_months(anchor, -1)) %>%
  select(member_id, domain, meps_country, meps_party)

control_pre <- control_pre %>%
  left_join(control_covs, by = c("member_id", "domain")) %>%
  mutate(treated = 0L)

message("Control units with valid anchors: ", nrow(control_pre))

# Keep only domains that exist in both groups to allow exact matching by domain
common_domains <- intersect(unique(treated_pre$domain), unique(control_pre$domain))
treated_pre <- treated_pre %>% filter(domain %in% common_domains)
control_pre <- control_pre %>% filter(domain %in% common_domains)

# -------------------------------------------------------------
# Assemble cross-section for matching
# -------------------------------------------------------------
match_df <- bind_rows(
  treated_pre %>% transmute(member_id, domain, treated, y_mean_pre, y_sd_pre, y_last_pre, y_slope_pre, meps_country, meps_party),
  control_pre %>% transmute(member_id, domain, treated, y_mean_pre, y_sd_pre, y_last_pre, y_slope_pre, meps_country, meps_party)
) %>%
  mutate(across(c(meps_country, meps_party, domain), as.factor))

message("Matching sample size: ", nrow(match_df), " (treated: ", sum(match_df$treated == 1), ", controls: ", sum(match_df$treated == 0), ")")

# -------------------------------------------------------------
# Run nearest-neighbor matching on pre-trend features
# -------------------------------------------------------------
set.seed(123)
form <- treated ~ y_mean_pre + y_sd_pre + y_last_pre + y_slope_pre + meps_country + meps_party

m.out <- matchit(
  formula = form,
  data = match_df,
  method = "nearest",
  distance = "glm",
  replace = TRUE,
  ratio = 1,
  exact = ~ domain
)

message("MatchIt finished. Creating diagnostics ...")

# Balance diagnostics
bal <- bal.tab(m.out, un = TRUE, m.threshold = 0.1)
capture.output(print(bal), file = file.path(tables_dir, "pretrend_balance.txt"))

# Love plot
png(file.path(figures_dir, "pretrend_love_plot.png"), width = 1200, height = 1600, res = 150)
print(love.plot(m.out, stats = c("m"), abs = TRUE, var.order = "unadjusted",
                thresholds = c(m = .1), var.names = c(
                  y_mean_pre = "Mean (pre)",
                  y_sd_pre = "SD (pre)",
                  y_last_pre = "Level at -1",
                  y_slope_pre = "Slope (pre)",
                  meps_country = "Country",
                  meps_party = "Party"
                )))
dev.off()

# Extract matched data
matched <- match.data(m.out)

# Save pairs with subclass information
readr::write_csv(matched, file.path(outputs_dir, "pretrend_match_pairs.csv"))

# -------------------------------------------------------------
# Simple visualization: compare average pre-trends (sanity check)
# -------------------------------------------------------------
message("Building simple pre-trend visualization (sanity check) ...")

plot_pretrend <- function(sample_ids, label) {
  # Average trajectory in the last pre_window_months relative to anchor
  # Reconstruct for each unit using the same pre-window we computed features on
  sample <- sample_ids %>% select(member_id, domain)
  if (label == "treated") {
    tmp <- treated_units %>% semi_join(sample, by = c("member_id", "domain")) %>%
      mutate(pre_months = map(t0, ~ seq_months(add_months(.x, -1), pre_window_months))) %>%
      mutate(data_pre = pmap(list(member_id, domain, pre_months), function(i, d, months_vec) {
        df %>% filter(member_id == i, domain == d, Y.m %in% months_vec) %>% arrange(Y.m) %>%
          mutate(t_rel = -rev(seq_len(n())))
      })) %>% tidyr::unnest(cols = data_pre, names_sep = "_")
  } else {
    tmp <- control_anchors %>% semi_join(sample, by = c("member_id", "domain")) %>%
      mutate(pre_months = map(anchor, ~ seq_months(add_months(.x, -1), pre_window_months))) %>%
      mutate(data_pre = pmap(list(member_id, domain, pre_months), function(i, d, months_vec) {
        df %>% filter(member_id == i, domain == d, Y.m %in% months_vec) %>% arrange(Y.m) %>%
          mutate(t_rel = -rev(seq_len(n())))
      })) %>% tidyr::unnest(cols = data_pre, names_sep = "_")
  }
  tmp %>%
    rename(t_rel = data_pre_t_rel, questions = data_pre_questions) %>%
    group_by(t_rel) %>%
    summarize(y = mean(questions, na.rm = TRUE), .groups = "drop") %>%
    mutate(group = label)
}

treated_ids <- matched %>% filter(treated == 1) %>% distinct(member_id, domain)
control_ids <- matched %>% filter(treated == 0) %>% distinct(member_id, domain)

avg_t <- plot_pretrend(treated_ids, "treated")
avg_c <- plot_pretrend(control_ids, "control")

avg_both <- bind_rows(avg_t, avg_c)

g <- ggplot(avg_both, aes(x = t_rel, y = y, color = group)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = seq(-pre_window_months, -1, by = 1)) +
  labs(title = "Average pre-period trajectories (matched sample)",
       x = "Months before anchor (negative)", y = "Questions") +
  theme_minimal(base_size = 12) +
  theme(legend.title = element_blank())

ggsave(filename = file.path(figures_dir, "pretrend_avg_trajectories.png"), plot = g,
       width = 9, height = 5, dpi = 300)

# -------------------------------------------------------------
# Save summary
# -------------------------------------------------------------
summary_list <- list(
  params = list(pre_window_months = pre_window_months, post_buffer_months = post_buffer_months,
                min_non_missing = min_non_missing),
  n_treated = sum(match_df$treated == 1),
  n_controls = sum(match_df$treated == 0),
  n_matched = nrow(matched),
  balance = bal
)

saveRDS(summary_list, file.path(outputs_dir, "pretrend_match_summary.rds"))

message("Pre-trend matching test completed. Outputs saved to:")
message(" - ", outputs_dir)
message(" - ", figures_dir)
message(" - ", tables_dir)
