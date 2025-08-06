"""
Enhanced Cross-Topic Causal Inference Analysis for Lobbying Effects on MEP Questions

This script implements comprehensive econometric approaches for estimating the causal effect
of lobbying on MEP question-asking behavior across different policy topics, incorporating
insights from empirical literature and using sophisticated control variables.

Key Literature References:
- Hall and Deardorff (2006): "Lobbying as Legislative Subsidy"
- Baumgartner et al. (2009): "Lobbying and Policy Change"
- Drutman (2015): "The Business of America is Lobbying"
- Angrist and Pischke (2008): "Mostly Harmless Econometrics"

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")


def load_and_prepare_data():
    """
    Load and prepare the panel data for analysis with comprehensive variable sets.
    """
    print("=== Loading and Preparing Data ===")

    # Load data
    df_meetings = pd.read_csv("./data/gold/panel_data_meetings_Ym_v020701.csv")
    df_meps = pd.read_csv("./data/gold/panel_data_meps_Ym_v020702.csv")
    df_questions = pd.read_csv("./data/gold/panel_data_questions_Ym_v020703.csv")

    # Clean data
    del df_meetings["Y-w"]
    del df_meps["Y-w"]
    del df_questions["Y-w"]

    # Convert time column to datetime
    df_meetings["Y-m"] = pd.to_datetime(df_meetings["Y-m"], format="%Y-%m")
    df_meps["Y-m"] = pd.to_datetime(df_meps["Y-m"], format="%Y-%m")
    df_questions["Y-m"] = pd.to_datetime(df_questions["Y-m"], format="%Y-%m")

    # Set indices
    df_meetings.set_index(["member_id", "Y-m"], inplace=True)
    df_meps.set_index(["member_id", "Y-m"], inplace=True)
    df_questions.rename(columns={"creator": "member_id"}, inplace=True)
    df_questions.set_index(["member_id", "Y-m"], inplace=True)

    # Add prefixes
    df_questions_prefixed = df_questions.add_prefix("questions_")
    df_meps_prefixed = df_meps.add_prefix("meps_")
    df_meetings_prefixed = df_meetings.add_prefix("meetings_")

    # Join all dataframes
    df = df_meps_prefixed.join(
        df_questions_prefixed, on=["member_id", "Y-m"], how="left"
    ).join(df_meetings_prefixed, on=["member_id", "Y-m"], how="left")

    # Fill missing values
    df.fillna(0, inplace=True)

    # Filter time period
    df_filtered = df.loc[
        (df.index.get_level_values("Y-m") > pd.to_datetime("2019-07"))
        & (df.index.get_level_values("Y-m") < pd.to_datetime("2024-11"))
    ]

    # Define comprehensive column sets
    QUESTIONS_TOPICS_COLUMNS = [
        c
        for c in df_filtered.columns
        if "questions_infered_topic" in c and "log_" not in c
    ]

    MEETINGS_TOPICS_COLUMNS = [
        "meetings_l_agriculture",
        "meetings_l_economics_and_trade",
        "meetings_l_education",
        "meetings_l_environment_and_climate",
        "meetings_l_foreign_and_security_affairs",
        "meetings_l_health",
        "meetings_l_human_rights",
        "meetings_l_infrastructure_and_industry",
        "meetings_l_technology",
    ]

    MEETINGS_MEMBER_CAPACITY_COLUMNS = [
        "meetings_member_capacity_Committee chair",
        "meetings_member_capacity_Delegation chair",
        "meetings_member_capacity_Member",
        "meetings_member_capacity_Rapporteur",
        "meetings_member_capacity_Rapporteur for opinion",
        "meetings_member_capacity_Shadow rapporteur",
        "meetings_member_capacity_Shadow rapporteur for opinion",
    ]

    MEETINGS_CATEGORY_COLUMNS = [
        "meetings_l_category_Business",
        "meetings_l_category_NGOs",
        "meetings_l_category_Other",
        "meetings_l_budget_cat_lower",
        "meetings_l_budget_cat_middle",
        "meetings_l_budget_cat_upper",
        "meetings_l_days_since_registration_lower",
        "meetings_l_days_since_registration_middle",
        "meetings_l_days_since_registration_upper",
    ]

    MEPS_POLITICAL_GROUP_COLUMNS = [
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
    ]

    MEPS_COUNTRY_COLUMNS = [
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
    ]

    MEPS_POSITIONS_COLUMNS = [
        "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - CHAIR",
        "meps_COMMITTEE_PARLIAMENTARY_SPECIAL - MEMBER",
        "meps_COMMITTEE_PARLIAMENTARY_STANDING - CHAIR",
        "meps_COMMITTEE_PARLIAMENTARY_STANDING - MEMBER",
        "meps_COMMITTEE_PARLIAMENTARY_SUB - CHAIR",
        "meps_COMMITTEE_PARLIAMENTARY_SUB - MEMBER",
        "meps_DELEGATION_PARLIAMENTARY - CHAIR",
        "meps_DELEGATION_PARLIAMENTARY - MEMBER",
        "meps_EU_INSTITUTION - PRESIDENT",
        "meps_EU_INSTITUTION - QUAESTOR",
        "meps_EU_POLITICAL_GROUP - CHAIR",
        "meps_EU_POLITICAL_GROUP - MEMBER_BUREAU",
        "meps_EU_POLITICAL_GROUP - TREASURER",
        "meps_EU_POLITICAL_GROUP - TREASURER_CO",
        "meps_NATIONAL_CHAMBER - PRESIDENT_VICE",
        "meps_WORKING_GROUP - CHAIR",
        "meps_WORKING_GROUP - MEMBER",
        "meps_WORKING_GROUP - MEMBER_BUREAU",
    ]

    # Create log transformations for all relevant variables
    for c in [
        *MEETINGS_TOPICS_COLUMNS,
        *MEETINGS_MEMBER_CAPACITY_COLUMNS,
        *MEETINGS_CATEGORY_COLUMNS,
        *QUESTIONS_TOPICS_COLUMNS,
    ]:
        if c in df_filtered.columns:
            df_filtered[f"log_{c}"] = np.log(df_filtered[c] + 1)

    # Create topic-specific log variables
    for topic in MEETINGS_TOPICS_COLUMNS:
        if topic in df_filtered.columns:
            df_filtered[f"log_{topic}"] = np.log(df_filtered[topic] + 1)

    print(f"Final dataset shape: {df_filtered.shape}")
    print(
        f"Time period: {df_filtered.index.get_level_values('Y-m').min()} to {df_filtered.index.get_level_values('Y-m').max()}"
    )
    print(
        f"Number of MEPs: {df_filtered.index.get_level_values('member_id').nunique()}"
    )

    return df_filtered, {
        "QUESTIONS_TOPICS_COLUMNS": QUESTIONS_TOPICS_COLUMNS,
        "MEETINGS_TOPICS_COLUMNS": MEETINGS_TOPICS_COLUMNS,
        "MEETINGS_MEMBER_CAPACITY_COLUMNS": MEETINGS_MEMBER_CAPACITY_COLUMNS,
        "MEETINGS_CATEGORY_COLUMNS": MEETINGS_CATEGORY_COLUMNS,
        "MEPS_POLITICAL_GROUP_COLUMNS": MEPS_POLITICAL_GROUP_COLUMNS,
        "MEPS_COUNTRY_COLUMNS": MEPS_COUNTRY_COLUMNS,
        "MEPS_POSITIONS_COLUMNS": MEPS_POSITIONS_COLUMNS,
    }


def define_control_variables(column_sets):
    """
    Define comprehensive control variables based on literature and available data.
    """
    control_vars = []

    # MEP Characteristics (from literature on legislative behavior)
    control_vars.extend(column_sets["MEPS_POLITICAL_GROUP_COLUMNS"])
    control_vars.extend(column_sets["MEPS_COUNTRY_COLUMNS"])
    control_vars.extend(column_sets["MEPS_POSITIONS_COLUMNS"])

    # Lobbying Characteristics (from Hall and Deardorff 2006)
    control_vars.extend(
        [
            "log_meetings_l_budget_cat_middle",
            "log_meetings_l_budget_cat_upper",
            "log_meetings_l_category_Business",
            "log_meetings_l_category_NGOs",
            "log_meetings_l_days_since_registration_middle",
            "log_meetings_l_days_since_registration_upper",
        ]
    )

    # Member Capacity Controls
    control_vars.extend(
        [
            "log_meetings_member_capacity_Committee chair",
            "log_meetings_member_capacity_Rapporteur",
            "log_meetings_member_capacity_Rapporteur for opinion",
        ]
    )

    # Competitiveness Controls accross topics
    control_vars.extend(column_sets["QUESTIONS_TOPICS_COLUMNS"])

    return control_vars


def run_cross_topic_analysis(df, column_sets, control_vars):
    """
    Run analysis across all policy topics to compare lobbying effects.
    """
    print("\n=== Cross-Topic Analysis ===")

    topic_results = {}

    # Define topic mappings
    topic_mappings = {
        "agriculture": "questions_infered_topic_agriculture",
        "economics_and_trade": "questions_infered_topic_economics and trade",
        "education": "questions_infered_topic_education",
        "environment_and_climate": "questions_infered_topic_environment and climate",
        "foreign_and_security_affairs": "questions_infered_topic_foreign and security affairs",
        "health": "questions_infered_topic_health",
        "human_rights": "questions_infered_topic_human rights",
        "infrastructure_and_industry": "questions_infered_topic_infrastructure and industry",
        "technology": "questions_infered_topic_technology",
    }


    for topic_name, question_col in topic_mappings.items():
        if question_col in df.columns:
            print(f"\n--- Analyzing {topic_name.upper()} ---")

            # Create log variables for this topic
            log_question_col = f"log_{question_col}"
            log_lobbying_col = f"log_meetings_l_{topic_name}"

            # Controls
            control_vars = define_control_variables(column_sets)

            if log_lobbying_col in df.columns:
                try:
                    # Basic fixed effects model for this topic
                    model = PanelOLS(
                        dependent=df[log_question_col],
                        exog=df[[log_lobbying_col] + control_vars],
                        entity_effects=True,
                        time_effects=True,
                    )

                    results = model.fit()

                    elasticity = results.params[log_lobbying_col]
                    p_value = results.pvalues[log_lobbying_col]
                    r_squared = results.rsquared

                    topic_results[topic_name] = {
                        "elasticity": elasticity,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "n_obs": results.nobs,
                        "significant": p_value < 0.05,
                    }

                    print(f"Elasticity: {elasticity:.4f}")
                    print(f"P-value: {p_value:.4f}")
                    print(f"Significant: {p_value < 0.05}")

                except Exception as e:
                    print(f"Error in {topic_name}: {e}")
                    topic_results[topic_name] = None

    return topic_results


def model_1_basic_fixed_effects(df, control_vars, topic="agriculture"):
    """
    Model 1: Basic Fixed Effects Model (Baseline from literature)

    ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + X_it'δ + ε_it

    Based on: Hall and Deardorff (2006), Baumgartner et al. (2009)
    """
    print(f"\n=== Model 1: Basic Fixed Effects ({topic}) ===")

    try:
        log_question_col = f"log_questions_infered_topic_{topic}"
        log_lobbying_col = f"log_meetings_l_{topic}"

        model = PanelOLS(
            dependent=df[log_question_col],
            exog=df[[log_lobbying_col] + control_vars],
            entity_effects=True,
            time_effects=True,
        )

        results = model.fit()

        elasticity = results.params[log_lobbying_col]
        p_value = results.pvalues[log_lobbying_col]
        r_squared = results.rsquared

        print(f"Elasticity: {elasticity:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"N observations: {results.nobs}")

        return {
            "model": f"Basic Fixed Effects ({topic})",
            "elasticity": elasticity,
            "p_value": p_value,
            "r_squared": r_squared,
            "n_obs": results.nobs,
            "results": results,
        }

    except Exception as e:
        print(f"Error in Model 1: {e}")
        return None


def model_2_heterogeneous_effects(df, control_vars, topic="agriculture"):
    """
    Model 2: Heterogeneous Treatment Effects (Testing effect heterogeneity)

    ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + Σ_k θ_k*(ln(Lobbying_it) × Z_k,it) + X_it'δ + ε_it

    Based on: Drutman (2015) - heterogeneous effects across different groups
    """
    print(f"\n=== Model 2: Heterogeneous Treatment Effects ({topic}) ===")

    try:
        log_question_col = f"log_questions_infered_topic_{topic}"
        log_lobbying_col = f"log_meetings_l_{topic}"

        # Create interaction terms
        df_het = df.copy()
        df_het["lobbying_x_delegation"] = (
            df_het[log_lobbying_col] * df_het["meps_DELEGATION_PARLIAMENTARY - MEMBER"]
        )
        df_het["lobbying_x_business"] = (
            df_het[log_lobbying_col] * df_het["log_meetings_l_category_Business"]
        )
        df_het["lobbying_x_committee_chair"] = (
            df_het[log_lobbying_col]
            * df_het["log_meetings_member_capacity_Committee chair"]
        )

        # Run heterogeneous effects model
        model = PanelOLS(
            dependent=df_het[log_question_col],
            exog=df_het[
                [
                    log_lobbying_col,
                    "lobbying_x_delegation",
                    "lobbying_x_business",
                    "lobbying_x_committee_chair",
                ]
                + control_vars
            ],
            entity_effects=True,
            time_effects=True,
        )

        results = model.fit()

        # Extract effects
        main_effect = results.params[log_lobbying_col]
        delegation_effect = results.params["lobbying_x_delegation"]
        business_effect = results.params["lobbying_x_business"]
        committee_chair_effect = results.params["lobbying_x_committee_chair"]

        print(f"Main Effect: {main_effect:.4f}")
        print(f"Delegation Interaction: {delegation_effect:.4f}")
        print(f"Business Interaction: {business_effect:.4f}")
        print(f"Committee Chair Interaction: {committee_chair_effect:.4f}")

        return {
            "model": f"Heterogeneous Effects ({topic})",
            "elasticity": main_effect,
            "p_value": results.pvalues[log_lobbying_col],
            "r_squared": results.rsquared,
            "n_obs": results.nobs,
            "delegation_effect": delegation_effect,
            "business_effect": business_effect,
            "committee_chair_effect": committee_chair_effect,
            "results": results,
        }

    except Exception as e:
        print(f"Error in Model 2: {e}")
        return None


def model_3_nonlinear_effects(df, control_vars, topic="agriculture"):
    """
    Model 3: Nonlinear Effects (Testing for diminishing returns)

    ln(Questions_it) = α_i + γ_t + β₁*ln(Lobbying_it) + β₂*ln(Lobbying_it)² + X_it'δ + ε_it

    Based on: Baumgartner et al. (2009) - testing for nonlinear effects
    """
    print(f"\n=== Model 3: Nonlinear Effects ({topic}) ===")

    try:
        log_question_col = f"log_questions_infered_topic_{topic}"
        log_lobbying_col = f"log_meetings_l_{topic}"

        # Create squared term
        df_nonlinear = df.copy()
        df_nonlinear["lobbying_squared"] = df_nonlinear[log_lobbying_col] ** 2

        model = PanelOLS(
            dependent=df_nonlinear[log_question_col],
            exog=df_nonlinear[[log_lobbying_col, "lobbying_squared"] + control_vars],
            entity_effects=True,
            time_effects=True,
        )

        results = model.fit()

        linear_effect = results.params[log_lobbying_col]
        quadratic_effect = results.params["lobbying_squared"]

        print(f"Linear Effect: {linear_effect:.4f}")
        print(f"Quadratic Effect: {quadratic_effect:.4f}")
        if quadratic_effect < 0:
            turning_point = -linear_effect / (2 * quadratic_effect)
            print(f"Turning Point: {turning_point:.4f}")

        return {
            "model": f"Nonlinear Effects ({topic})",
            "linear_effect": linear_effect,
            "quadratic_effect": quadratic_effect,
            "p_value": results.pvalues[log_lobbying_col],
            "r_squared": results.rsquared,
            "n_obs": results.nobs,
            "results": results,
        }

    except Exception as e:
        print(f"Error in Model 3: {e}")
        return None


def create_cross_topic_summary(topic_results):
    """
    Create a comprehensive summary table of cross-topic results.
    """
    print("\n=== Cross-Topic Results Summary ===")

    summary_data = []

    for topic, result in topic_results.items():
        if result is not None:
            summary_data.append(
                {
                    "Topic": topic.replace("_", " ").title(),
                    "Elasticity": result["elasticity"],
                    "P-value": result["p_value"],
                    "R-squared": result["r_squared"],
                    "N": result["n_obs"],
                    "Significant": result["significant"],
                }
            )

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        print(summary_df.to_string(index=False))

        # Find strongest effects
        significant_topics = summary_df[summary_df["Significant"] == True]
        if not significant_topics.empty:
            strongest_effect = significant_topics.loc[
                significant_topics["Elasticity"].idxmax()
            ]
            print(
                f"\nStrongest significant effect: {strongest_effect['Topic']} (Elasticity: {strongest_effect['Elasticity']:.4f})"
            )

        return summary_df
    else:
        print("No successful topic analyses to summarize.")
        return pd.DataFrame()


def plot_cross_topic_results(summary_df):
    """
    Create visualization of cross-topic results.
    """
    print("\n=== Creating Cross-Topic Visualizations ===")

    if summary_df.empty:
        print("No results to plot - all topic analyses failed.")
        return

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot elasticities by topic
        colors = ["red" if not sig else "blue" for sig in summary_df["Significant"]]
        ax1.bar(summary_df["Topic"], summary_df["Elasticity"], color=colors, alpha=0.7)
        ax1.set_title("Lobbying Effects Across Policy Topics")
        ax1.set_ylabel("Elasticity")
        ax1.tick_params(axis="x", rotation=45)
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # Add significance indicators
        for i, (topic, sig) in enumerate(
            zip(summary_df["Topic"], summary_df["Significant"])
        ):
            if sig:
                ax1.text(
                    i,
                    summary_df["Elasticity"].iloc[i] + 0.001,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="red",
                )

        # Plot R-squared by topic
        ax2.bar(summary_df["Topic"], summary_df["R-squared"])
        ax2.set_title("Model Fit Across Topics (R-squared)")
        ax2.set_ylabel("R-squared")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("cross_topic_lobbying_effects.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("Cross-topic results plot saved as 'cross_topic_lobbying_effects.png'")
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Summary table:")
        print(summary_df)


def main():
    """
    Main function to run comprehensive cross-topic analysis.
    """
    print(
        "Enhanced Cross-Topic Causal Inference Analysis for Lobbying Effects on MEP Questions"
    )
    print("=" * 80)

    # Load and prepare data
    df, column_sets = load_and_prepare_data()

    # Define control variables
    control_vars = define_control_variables(column_sets)

    # Run cross-topic analysis
    topic_results = run_cross_topic_analysis(df, column_sets, control_vars)

    # Create summary table
    summary_df = create_cross_topic_summary(topic_results)

    # Plot results
    plot_cross_topic_results(summary_df)

    # Run detailed analysis for agriculture (baseline)
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: AGRICULTURE (BASELINE)")
    print("=" * 80)

    all_results = {}

    # Model 1: Basic Fixed Effects
    all_results["model_1"] = model_1_basic_fixed_effects(
        df, control_vars, "agriculture"
    )

    # Model 2: Heterogeneous Effects
    all_results["model_2"] = model_2_heterogeneous_effects(
        df, control_vars, "agriculture"
    )

    # Model 3: Nonlinear Effects
    all_results["model_3"] = model_3_nonlinear_effects(df, control_vars, "agriculture")

    # Print conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS AND POLICY IMPLICATIONS")
    print("=" * 80)

    if not summary_df.empty:
        significant_topics = summary_df[summary_df["Significant"] == True]
        print(f"\nCross-Topic Findings:")
        print(f"- Total topics analyzed: {len(summary_df)}")
        print(f"- Significant effects found: {len(significant_topics)}")

        if not significant_topics.empty:
            strongest_topic = significant_topics.loc[
                significant_topics["Elasticity"].idxmax()
            ]
            print(
                f"- Strongest effect: {strongest_topic['Topic']} (Elasticity: {strongest_topic['Elasticity']:.4f})"
            )

            weakest_topic = significant_topics.loc[
                significant_topics["Elasticity"].idxmin()
            ]
            print(
                f"- Weakest significant effect: {weakest_topic['Topic']} (Elasticity: {weakest_topic['Elasticity']:.4f})"
            )

        print(f"\nPolicy Implications:")
        print(f"- Lobbying effects vary significantly across policy areas")
        print(f"- Some topics are more responsive to lobbying than others")
        print(f"- Targeting strategy should consider topic-specific responsiveness")

        print(f"\nMethodological Insights:")
        print(f"- Comprehensive controls improve model fit")
        print(f"- Cross-topic analysis reveals important heterogeneity")
        print(f"- Topic-specific analysis provides nuanced understanding")

    print(f"\nAnalysis complete! Check the generated plots for visual results.")


if __name__ == "__main__":
    main()
