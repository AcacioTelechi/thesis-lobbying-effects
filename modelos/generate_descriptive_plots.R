library(fixest)
library(modelsummary)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(gridExtra)

# Set figure directory
fig_dir <- "../Tese/figures"

# Create directory if it doesn't exist
if (!dir.exists(fig_dir)) {
    dir.create(fig_dir, recursive = TRUE)
}

# Load data
df <- read.csv("../data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# Data preprocessing
# Ensure Y.m is properly formatted
if (!is.numeric(df$Y.m)) {
    if (is.factor(df$Y.m)) {
        df$Y.m <- as.character(df$Y.m)
    }
    if (all(grepl("^\\d{4}-\\d{2}$", df$Y.m))) {
        # Format: YYYY-MM
        df$Y.m_num <- as.numeric(gsub("-", "", df$Y.m))
    } else if (all(grepl("^\\d{6}$", df$Y.m))) {
        # Format: YYYYMM
        df$Y.m_num <- as.numeric(df$Y.m)
    } else {
        # Fallback: try to parse as date
        df$Y.m_num <- as.numeric(as.Date(paste0(df$Y.m, "-01")))
    }
} else {
    df$Y.m_num <- df$Y.m
}

# Convert to date for better plotting
df$Y.m_date <- as.Date(paste0(df$Y.m, "-01"))

# ============================================
# PLOT 1: Time series of meetings and questions by Y.m
# ============================================

# Aggregate total meetings and questions by time period
df_time_series_total <- df %>%
    group_by(Y.m, Y.m_num, Y.m_date) %>%
    summarise(
        total_meetings = sum(meetings, na.rm = TRUE),
        total_questions = sum(questions, na.rm = TRUE),
        .groups = "drop"
    )

# Create dual-axis plot for meetings and questions
p1_time_series <- ggplot(df_time_series_total, aes(x = Y.m_date)) +
    geom_line(aes(y = total_meetings, color = "Reuniões"), linewidth = 1) +
    geom_line(aes(y = total_questions, color = "Questões"), linewidth = 1) +
    scale_color_manual(values = c("Reuniões" = "#1f77b4", "Questões" = "#ff7f0e")) +
    scale_y_continuous(
        name = "Reuniões",
        sec.axis = sec_axis(~., name = "Questões")
    ) +
    labs(
        x = "Tempo (mensal)",
        # title = "Time Series of Meetings and Questions",
        # subtitle = "Monthly aggregation of total meetings and questions",
        color = "Variável"
    ) +
    theme_minimal(base_size = 12) +
    theme(
        legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

print(p1_time_series)

# Analysis for Plot 1:
# This plot shows the temporal evolution of both meetings and questions over time.
# It allows us to identify potential correlations, trends, and seasonal patterns between
# lobbying activities (meetings) and parliamentary activities (questions).

# ============================================
# PLOT 2: Time series of proportion of individuals attending meetings by Y.m
# ============================================

# Calculate proportion of individuals with meetings by time period
# First, get the total unique individuals in the entire dataset
total_unique_individuals_ever <- df %>%
    distinct(member_id) %>%
    nrow()

# Create a cumulative tracking of unique individuals over time
# First, find the first treatment date for each individual
first_treatment_by_individual <- df %>%
    filter(meetings > 0) %>%
    group_by(member_id) %>%
    summarise(first_treatment_date = min(Y.m_date), .groups = "drop")

# For each month, count how many individuals had their first treatment on or before that date
unique_individuals_by_period <- df %>%
    distinct(Y.m, Y.m_date) %>%
    arrange(Y.m_date) %>%
    mutate(
        cumulative_unique_individuals = sapply(Y.m_date, function(ref_date) {
            sum(first_treatment_by_individual$first_treatment_date <= ref_date)
        }),
        accumulated_proportion = cumulative_unique_individuals / total_unique_individuals_ever
    )

# Calculate monthly proportions
df_meeting_proportion <- df %>%
    group_by(Y.m, Y.m_num, Y.m_date) %>%
    summarise(
        total_individuals = n_distinct(member_id),
        individuals_with_meetings = n_distinct(member_id[meetings > 0]),
        proportion_meetings = individuals_with_meetings / total_individuals,
        .groups = "drop"
    ) %>%
    # Join with cumulative data
    left_join(unique_individuals_by_period, by = c("Y.m", "Y.m_date")) %>%
    # Fill forward the accumulated proportion for months with no new unique individuals
    arrange(Y.m_date) %>%
    fill(accumulated_proportion, .direction = "down") %>%
    # Replace NA values with 0 for the first periods
    mutate(accumulated_proportion = ifelse(is.na(accumulated_proportion), 0, accumulated_proportion))

# Debug output to verify calculations
cat("\n=== DEBUG: ACCUMULATED PROPORTION CALCULATION ===\n")
cat("Total unique individuals in dataset:", total_unique_individuals_ever, "\n")
cat("Total unique individuals who had meetings:", nrow(first_treatment_by_individual), "\n")
cat("Final accumulated proportion:", max(df_meeting_proportion$accumulated_proportion, na.rm = TRUE), "\n")
cat("First few rows of accumulated proportion:\n")
print(head(df_meeting_proportion[, c("Y.m", "proportion_meetings", "accumulated_proportion")], 10))
cat("Last few rows of accumulated proportion:\n")
print(tail(df_meeting_proportion[, c("Y.m", "proportion_meetings", "accumulated_proportion")], 10))

p2_proportion_meetings <- ggplot(df_meeting_proportion, aes(x = Y.m_date)) +
    # Primary axis: monthly proportion
    geom_line(aes(y = proportion_meetings, color = "Proporção Mensal"), linewidth = 1) +
    geom_point(aes(y = proportion_meetings, color = "Proporção Mensal"), size = 2) +
    # Secondary axis: accumulated proportion (scaled independently)
    geom_line(aes(y = accumulated_proportion, color = "Proporção Acumulada"), linewidth = 1) +
    geom_point(aes(y = accumulated_proportion, color = "Proporção Acumulada"), size = 2) +
    scale_color_manual(values = c("Proporção Mensal" = "#1f77b4", "Proporção Acumulada" = "#ff7f0e")) +
    scale_y_continuous(
        name = "Proporção Mensal",
        labels = percent_format(accuracy = 1),
        # Secondary axis with independent scaling to avoid distortion
        sec.axis = sec_axis(
            transform = ~., 
            name = "Proporção Acumulada", 
            labels = percent_format(accuracy = 1),
            breaks = seq(0, max(df_meeting_proportion$accumulated_proportion, na.rm = TRUE), by = 0.1)
        )
    ) +
    labs(
        x = "Tempo (mensal)",
        y = "Proporção de indivíduos com reuniões",
        title = "Proporção de indivíduos com reuniões ao longo do tempo",
        subtitle = "Proporção mensal vs. proporção acumulada de MEPs únicos com reuniões",
        color = "Tipo de Proporção"
    ) +
    theme_minimal(base_size = 12) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom"
    )

print(p2_proportion_meetings)

# Analysis for Plot 2:
# This dual-axis plot shows both the monthly proportion of MEPs who participated in meetings
# and the accumulated proportion of unique MEPs over time. The monthly proportion reveals
# periods of increased or decreased lobbying activity, while the accumulated proportion
# shows the cumulative reach of lobbying activities across the entire MEP population.
# This helps identify both short-term fluctuations and long-term trends in lobbying participation.

# ============================================
# PLOT 3: Correlation of meetings and questions by Y.m
# ============================================

# Calculate correlation by time period
df_correlation <- df %>%
    group_by(Y.m, Y.m_num, Y.m_date) %>%
    summarise(
        correlation = cor(meetings, questions, use = "complete.obs"),
        total_meetings = sum(meetings, na.rm = TRUE),
        total_questions = sum(questions, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    filter(!is.na(correlation))

p3_correlation <- ggplot(df_correlation, aes(x = Y.m_date, y = correlation)) +
    geom_line(linewidth = 1, color = "#d62728") +
    geom_point(size = 2, color = "#d62728") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_y_continuous(limits = c(-1, 1)) +
    labs(
        x = "Time",
        y = "Correlation Coefficient",
        title = "Correlation Between Meetings and Questions Over Time",
        subtitle = "Monthly correlation between individual-level meetings and questions"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3_correlation)

# Analysis for Plot 3:
# This plot shows the temporal evolution of the correlation between meetings and questions.
# It reveals whether the relationship between lobbying and parliamentary activity
# has changed over time, potentially indicating shifts in lobbying effectiveness.

# ============================================
# ADDITIONAL DOMAIN-BASED PLOTS
# ============================================

# Plot 4: Time series by domain for meetings
if ("domain" %in% colnames(df)) {
    df_domain_meetings <- df %>%
        group_by(Y.m, Y.m_num, Y.m_date, domain) %>%
        summarise(
            total_meetings = sum(meetings, na.rm = TRUE),
            .groups = "drop"
        )
    
    # Use faceting instead of color for better readability
    p4_domain_meetings <- ggplot(df_domain_meetings, aes(x = Y.m_date, y = total_meetings)) +
        geom_line(linewidth = 1, color = "#1f77b4") +
        geom_point(size = 1, color = "#1f77b4") +
        facet_wrap(~domain, scales = "free_y", ncol = 3) +
        labs(
            x = "Time",
            y = "Total Meetings",
            title = "Meetings by Policy Domain Over Time",
            subtitle = "Temporal evolution of lobbying activity across different policy areas (faceted by domain)"
        ) +
        theme_minimal(base_size = 10) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            strip.text = element_text(size = 10, face = "bold"),
            panel.grid.minor = element_blank()
        )
    
    print(p4_domain_meetings)
    
    # Analysis for Plot 4:
    # This faceted plot shows how lobbying activity varies across different policy domains over time.
    # Each domain has its own panel, making it easier to identify trends and patterns
    # without the visual clutter of overlapping lines.
}

# Plot 5: Time series by domain for questions
if ("domain" %in% colnames(df)) {
    df_domain_questions <- df %>%
        group_by(Y.m, Y.m_num, Y.m_date, domain) %>%
        summarise(
            total_questions = sum(questions, na.rm = TRUE),
            .groups = "drop"
        )
    
    # Use faceting instead of color for better readability
    p5_domain_questions <- ggplot(df_domain_questions, aes(x = Y.m_date, y = total_questions)) +
        geom_line(linewidth = 1, color = "#ff7f0e") +
        geom_point(size = 1, color = "#ff7f0e") +
        facet_wrap(~domain, scales = "free_y", ncol = 3) +
        labs(
            x = "Time",
            y = "Total Questions",
            title = "Questions by Policy Domain Over Time",
            subtitle = "Temporal evolution of parliamentary activity across different policy areas (faceted by domain)"
        ) +
        theme_minimal(base_size = 10) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            strip.text = element_text(size = 10, face = "bold"),
            panel.grid.minor = element_blank()
        )
    
    print(p5_domain_questions)
    
    # Analysis for Plot 5:
    # This faceted plot shows how parliamentary activity varies across different policy domains over time.
    # Each domain has its own panel, making it easier to identify trends and patterns
    # without the visual clutter of overlapping lines.
}

# Plot 6: Distribution of meetings by domain (boxplot)
if ("domain" %in% colnames(df)) {
    p6_domain_distribution <- ggplot(df[df$meetings > 0,], aes(x = domain, y = meetings)) +
        geom_boxplot(fill = "#1f77b4", alpha = 0.7) +
        labs(
            x = "Policy Domain",
            y = "Number of Meetings",
            title = "Distribution of Meetings by Policy Domain",
            subtitle = "Boxplot showing the spread and central tendency of lobbying activity"
        ) +
        theme_minimal(base_size = 10) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            legend.position = "none"
        )
    
    print(p6_domain_distribution)
    
    # Analysis for Plot 6:
    # This plot shows the distribution of lobbying activity across policy domains.
    # It reveals which domains have the highest variability in lobbying intensity
    # and helps identify outliers or particularly active domains.
}

# Plot 7: Combined meetings and questions by domain (side-by-side comparison)
if ("domain" %in% colnames(df)) {
    # Aggregate data for comparison
    df_domain_comparison <- df %>%
        group_by(domain) %>%
        summarise(
            total_meetings = sum(meetings, na.rm = TRUE),
            total_questions = sum(questions, na.rm = TRUE),
            avg_meetings = mean(meetings, na.rm = TRUE),
            avg_questions = mean(questions, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        pivot_longer(
            cols = c(total_meetings, total_questions),
            names_to = "metric",
            values_to = "value"
        ) %>%
        mutate(
            metric = factor(metric, 
                          levels = c("total_meetings", "total_questions"),
                          labels = c("Total Meetings", "Total Questions"))
        )
    
    p7_domain_comparison <- ggplot(df_domain_comparison, aes(x = domain, y = value, fill = metric)) +
        geom_col(position = "dodge", alpha = 0.8) +
        scale_fill_manual(values = c("Total Meetings" = "#1f77b4", "Total Questions" = "#ff7f0e")) +
        labs(
            x = "Policy Domain",
            y = "Count",
            title = "Total Meetings vs Questions by Policy Domain",
            subtitle = "Side-by-side comparison of lobbying and parliamentary activity across domains",
            fill = "Metric"
        ) +
        theme_minimal(base_size = 10) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            legend.position = "bottom"
        )
    
    print(p7_domain_comparison)
    
    # Analysis for Plot 7:
    # This side-by-side comparison shows the relative intensity of lobbying vs parliamentary activity
    # across different policy domains, making it easy to identify domains where lobbying
    # is more or less proportional to parliamentary attention.
}

# Plot 8: Heatmap of meetings and questions by domain and time
if ("domain" %in% colnames(df)) {
    # Create heatmap data
    df_heatmap <- df %>%
        group_by(Y.m, domain) %>%
        summarise(
            meetings = sum(meetings, na.rm = TRUE),
            questions = sum(questions, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        mutate(
            Y.m = factor(Y.m, levels = sort(unique(Y.m))),
            domain = factor(domain, levels = sort(unique(domain)))
        )
    
    # Meetings heatmap
    p8_meetings_heatmap <- ggplot(df_heatmap, aes(x = Y.m, y = domain, fill = meetings)) +
        geom_tile() +
        scale_fill_gradient2(
            low = "#f7f7f7", 
            mid = "#1f77b4", 
            high = "#08306b",
            midpoint = median(df_heatmap$meetings, na.rm = TRUE)
        ) +
        labs(
            x = "Time Period",
            y = "Policy Domain",
            title = "Meetings Heatmap by Domain and Time",
            subtitle = "Darker colors indicate higher meeting counts",
            fill = "Meetings"
        ) +
        theme_minimal(base_size = 10) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            axis.text.y = element_text(size = 9),
            panel.grid = element_blank()
        )
    
    print(p8_meetings_heatmap)
    
    # Analysis for Plot 8:
    # This heatmap provides a clear visual representation of lobbying intensity across
    # domains and time periods, making it easy to identify temporal patterns and
    # domain-specific trends without the clutter of overlapping lines.
}

# ============================================
# SAVE ALL PLOTS
# ============================================

# Save the three main plots
ggsave(file.path(fig_dir, "fig1_time_series_meetings_questions.pdf"), p1_time_series, width = 10, height = 6)
ggsave(file.path(fig_dir, "fig1_time_series_meetings_questions.png"), p1_time_series, width = 10, height = 6, dpi = 300)

ggsave(file.path(fig_dir, "fig2_proportion_meetings.pdf"), p2_proportion_meetings, width = 10, height = 6)
ggsave(file.path(fig_dir, "fig2_proportion_meetings.png"), p2_proportion_meetings, width = 10, height = 6, dpi = 300)

ggsave(file.path(fig_dir, "fig3_correlation_meetings_questions.pdf"), p3_correlation, width = 10, height = 6)
ggsave(file.path(fig_dir, "fig3_correlation_meetings_questions.png"), p3_correlation, width = 10, height = 6, dpi = 300)

# Save domain-based plots if they exist
if (exists("p4_domain_meetings")) {
    ggsave(file.path(fig_dir, "fig4_domain_meetings.pdf"), p4_domain_meetings, width = 12, height = 8)
    ggsave(file.path(fig_dir, "fig4_domain_meetings.png"), p4_domain_meetings, width = 12, height = 8, dpi = 300)
}

if (exists("p5_domain_questions")) {
    ggsave(file.path(fig_dir, "fig5_domain_questions.pdf"), p5_domain_questions, width = 12, height = 8)
    ggsave(file.path(fig_dir, "fig5_domain_questions.png"), p5_domain_questions, width = 12, height = 8, dpi = 300)
}

if (exists("p6_domain_distribution")) {
    ggsave(file.path(fig_dir, "fig6_domain_distribution.pdf"), p6_domain_distribution, width = 10, height = 6)
    ggsave(file.path(fig_dir, "fig6_domain_distribution.png"), p6_domain_distribution, width = 10, height = 6, dpi = 300)
}

if (exists("p7_domain_comparison")) {
    ggsave(file.path(fig_dir, "fig7_domain_comparison.pdf"), p7_domain_comparison, width = 12, height = 8)
    ggsave(file.path(fig_dir, "fig7_domain_comparison.png"), p7_domain_comparison, width = 12, height = 8, dpi = 300)
}

if (exists("p8_meetings_heatmap")) {
    ggsave(file.path(fig_dir, "fig8_meetings_heatmap.pdf"), p8_meetings_heatmap, width = 12, height = 8)
    ggsave(file.path(fig_dir, "fig8_meetings_heatmap.png"), p8_meetings_heatmap, width = 12, height = 8, dpi = 300)
}

# ============================================
# SUMMARY STATISTICS
# ============================================

cat("\n=== SUMMARY STATISTICS ===\n")
cat("Total observations:", nrow(df), "\n")
cat("Time period:", min(df$Y.m), "to", max(df$Y.m), "\n")
if ("domain" %in% colnames(df)) {
    cat("Number of domains:", length(unique(df$domain)), "\n")
    cat("Domains:", paste(unique(df$domain), collapse = ", "), "\n")
}
cat("Total meetings:", sum(df$meetings, na.rm = TRUE), "\n")
cat("Total questions:", sum(df$questions, na.rm = TRUE), "\n")
cat("Overall correlation:", cor(df$meetings, df$questions, use = "complete.obs"), "\n")

# ============================================
# GENERAL ANALYSIS AND CONCLUSIONS
# ============================================

cat("\n=== GENERAL ANALYSIS ===\n")
cat("The time series analysis reveals several key insights:\n")
cat("1. Temporal patterns in lobbying activity and parliamentary questions\n")
cat("2. Changes in the proportion of MEPs participating in lobbying\n")
cat("3. Evolution of the correlation between lobbying and parliamentary activity\n")
cat("4. Domain-specific variations in both lobbying and parliamentary behavior\n")
cat("5. Potential institutional or policy-driven changes affecting lobbying dynamics\n")
cat("\nThese visualizations provide a comprehensive understanding of the temporal\n")
cat("and domain-specific patterns in EU lobbying and parliamentary activity.\n")
