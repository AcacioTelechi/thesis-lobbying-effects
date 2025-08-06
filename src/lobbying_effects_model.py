"""
Lobbying Effects Analysis Model Classes

This module contains two main classes:
1. DataBase: Responsible for loading and treating panel data
2. LobbyingEffectsModel: Responsible for running econometric models with different topics
"""

from src.database import DataBase

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from linearmodels import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")


class LobbyingEffectsModel:
    """
    Class responsible for running econometric models with different topics.
    """

    def __init__(self, df_filtered, column_sets):
        """
        Initialize the model with data and column sets.

        Args:
            df_filtered (pd.DataFrame): Filtered panel data
            column_sets (dict): Dictionary containing column sets
        """
        self.df = df_filtered
        self.column_sets = column_sets
        self.topic: str | None = None
        self.control_vars: list[str] = []

    def set_topic(self, topic):
        """
        Set the topic for analysis and create topic-specific controls.

        Args:
            topic (str): Topic name (e.g., "agriculture", "technology", "health")
        """
        self.topic = topic
        self.control_vars = self._create_control_variables()
        print(f"Topic set to: {topic}")
        print(f"Number of control variables: {len(self.control_vars)}")

    def _create_control_variables(self):
        """
        Create comprehensive control variables based on the topic.
        """
        control_vars = []

        # MEP Characteristics
        control_vars.extend(self.column_sets["MEPS_POLITICAL_GROUP_COLUMNS"])
        control_vars.extend(self.column_sets["MEPS_COUNTRY_COLUMNS"])
        control_vars.extend(self.column_sets["MEPS_POSITIONS_COLUMNS"])

        # Lobbying Characteristics (excluding the main topic)
        for topic_col in self.column_sets["MEETINGS_TOPICS_COLUMNS"]:
            if self.topic not in topic_col:
                log_col = f"log_{topic_col}"
                if log_col in self.df.columns:
                    control_vars.append(log_col)

        # Lobbying Categories
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
                "log_meetings_member_capacity_Committee_chair",
                "log_meetings_member_capacity_Rapporteur",
                "log_meetings_member_capacity_Rapporteur_for_opinion",
            ]
        )

        # Other Topics Questions (excluding the main topic)
        for question_col in self.column_sets["QUESTIONS_TOPICS_COLUMNS"]:
            if self.topic not in question_col:
                control_vars.append(question_col)

        # Lobbying in other topics
        for topic_col in self.column_sets["MEETINGS_TOPICS_COLUMNS"]:
            if self.topic not in topic_col:
                control_vars.append(topic_col)

        # Graph
        control_vars.extend(
            [
                c
                for c in self.column_sets["GRAPH_AUTHORITY_COLUMNS"]
                if self.topic not in c and c != "graph_authority"
            ]
        )
        control_vars.extend(
            [
                c
                for c in self.column_sets["GRAPH_PERCENTAGE_COLUMNS"]
                if self.topic not in c
            ]
        )

        return control_vars

    def get_topic_variables(self) -> tuple[str, str, str, str]:
        """
        Get the main topic variables.

        Returns:
            tuple: (question_col, lobbying_col, log_question_col, log_lobbying_col)
        """
        question_col = f"questions_infered_topic_{self.topic}"
        lobbying_col = f"meetings_l_{self.topic}"
        log_question_col = f"log_{question_col}"
        log_lobbying_col = f"log_{lobbying_col}"

        return question_col, lobbying_col, log_question_col, log_lobbying_col

    def model_basic_fixed_effects(self):
        """
        Model 1: Basic Fixed Effects Model

        ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + X_it'δ + ε_it

        Based on: Hall and Deardorff (2006), Baumgartner et al. (2009)
        """
        print(f"\n=== Model 1: Basic Fixed Effects ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            model = PanelOLS(
                dependent=self.df[log_question_col],
                exog=self.df[[log_lobbying_col] + self.control_vars],
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
                "model": f"Basic Fixed Effects ({self.topic})",
                "elasticity": elasticity,
                "p_value": p_value,
                "r_squared": r_squared,
                "n_obs": results.nobs,
                "results": results,
            }

        except Exception as e:
            print(f"Error in Model 1: {e}")
            return None

    def model_heterogeneous_effects(self):
        """
        Model 2: Heterogeneous Treatment Effects

        ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + Σ_k θ_k*(ln(Lobbying_it) × Z_k,it) + X_it'δ + ε_it
        Based on: Drutman (2015) - heterogeneous effects across different groups
        """
        print(f"\n=== Model 2: Heterogeneous Treatment Effects ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            # Create interaction terms
            df_het = self.df.copy()
            df_het["lobbying_x_delegation"] = (
                df_het[log_lobbying_col]
                * df_het["meps_DELEGATION_PARLIAMENTARY_-_MEMBER"]
            )
            df_het["lobbying_x_business"] = (
                df_het[log_lobbying_col] * df_het["log_meetings_l_category_Business"]
            )
            df_het["lobbying_x_committee_chair"] = (
                df_het[log_lobbying_col]
                * df_het["log_meetings_member_capacity_Committee_chair"]
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
                    + self.control_vars
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
                "model": f"Heterogeneous Effects ({self.topic})",
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

    def model_nonlinear_effects(self):
        """
        Model 3: Nonlinear Effects

        ln(Questions_it) = α_i + γ_t + β₁*ln(Lobbying_it) + β₂*ln(Lobbying_it)² + X_it'δ + ε_it
        """
        print(f"\n=== Model 3: Nonlinear Effects ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            # Create squared term
            df_nonlinear = self.df.copy()
            df_nonlinear["lobbying_squared"] = df_nonlinear[log_lobbying_col] ** 2

            model = PanelOLS(
                dependent=df_nonlinear[log_question_col],
                exog=df_nonlinear[
                    [log_lobbying_col, "lobbying_squared"] + self.control_vars
                ],
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
                "model": f"Nonlinear Effects ({self.topic})",
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

    def model_propensity_score_matching(self):
        """
        Model 4: Propensity Score Matching (Addressing selection bias)

        Based on: Imbens and Rubin (2015) - addressing selection bias
        """
        print(f"\n=== Model 4: Propensity Score Matching ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            # Define treatment
            df_psm = self.df.copy()
            df_psm["treatment"] = (df_psm[log_lobbying_col] > 0).astype(int)

            # Covariates for propensity score estimation
            psm_covariates = [
                *self.column_sets["MEPS_POLITICAL_GROUP_COLUMNS"],
                *self.column_sets["MEPS_COUNTRY_COLUMNS"],
                *self.column_sets["MEPS_POSITIONS_COLUMNS"],
            ]

            # Estimate propensity scores
            logit = LogisticRegression(max_iter=1000, random_state=42)
            logit.fit(df_psm[psm_covariates], df_psm["treatment"])
            df_psm["propensity_score"] = logit.predict_proba(df_psm[psm_covariates])[
                :, 1
            ]

            # Separate treated and control
            treated = df_psm[df_psm["treatment"] == 1]
            control = df_psm[df_psm["treatment"] == 0]

            print(f"Treated units: {len(treated)}")
            print(f"Control units: {len(control)}")

            # Nearest neighbor matching (1:1)
            nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
            nn.fit(control[["propensity_score"]])
            distances, indices = nn.kneighbors(treated[["propensity_score"]])

            # Get matched controls
            matched_control = control.iloc[indices.flatten()]
            matched_treated = treated.copy()

            # Combine matched samples
            matched_df = pd.concat([matched_treated, matched_control])

            print(f"Matched sample size: {len(matched_df)}")

            # Run regression on matched sample
            model = PanelOLS(
                dependent=matched_df[log_question_col],
                exog=matched_df[
                    [
                        log_lobbying_col,
                        *[
                            c
                            for c in self.column_sets["QUESTIONS_TOPICS_COLUMNS"]
                            if self.topic not in c
                        ],
                        *[
                            c
                            for c in self.column_sets["MEETINGS_TOPICS_COLUMNS"]
                            if log_lobbying_col not in c
                        ],
                        *self.column_sets["MEETINGS_CATEGORY_COLUMNS"],
                    ]
                ],
                entity_effects=True,
                time_effects=True,
            )

            results = model.fit()

            elasticity = results.params[log_lobbying_col]
            p_value = results.pvalues[log_lobbying_col]

            print(f"PSM Elasticity: {elasticity:.4f}")
            print(f"PSM P-value: {p_value:.4f}")

            return {
                "model": f"Propensity Score Matching ({self.topic})",
                "elasticity": elasticity,
                "p_value": p_value,
                "r_squared": results.rsquared,
                "n_obs": results.nobs,
                "results": results,
            }

        except Exception as e:
            print(f"Error in Model 4: {e}")
            return None

    def model_instrumental_variables(self):
        """
        Model 5: Instrumental Variables (Addressing endogeneity)

        First Stage: ln(Lobbying_it) = π_0 + π_1*Z_it + X_it'π_2 + ν_it
        Second Stage: ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_hat_it) + ε_it

        Based on: Angrist and Pischke (2008) - instrumental variables approach
        """
        print(f"\n=== Model 5: Instrumental Variables ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            # Define treatment
            df_iv = self.df.copy()

            # First stage: Lobbying = f(Instrument, Controls)
            first_stage = PanelOLS(
                dependent=df_iv[log_lobbying_col],
                exog=df_iv[self.control_vars],
                # entity_effects=True,
                # time_effects=True,
            ).fit()

            # Get predicted values
            df_iv["lobbying_hat"] = first_stage.predict()

            # Second stage: Questions = f(Predicted_Lobbying, Controls)
            second_stage = PanelOLS(
                dependent=df_iv[log_question_col],
                exog=df_iv[["lobbying_hat"]],
                entity_effects=True,
                time_effects=True,
            ).fit()

            elasticity = second_stage.params["lobbying_hat"]
            p_value = second_stage.pvalues["lobbying_hat"]

            print(f"First Stage R-squared: {first_stage.rsquared:.4f}")
            print(f"Second Stage R-squared: {second_stage.rsquared:.4f}")
            print(f"\nIV Elasticity: {elasticity:.4f}")
            print(f"IV P-value: {p_value:.4f}")

            return {
                "model": f"Instrumental Variables ({self.topic})",
                "elasticity": elasticity,
                "p_value": p_value,
                "r_squared": second_stage.rsquared,
                "n_obs": len(df_iv),
                "first_stage": first_stage,
                "second_stage": second_stage,
            }

        except Exception as e:
            print(f"Error in Model 5: {e}")
            return None

    def model_staggered_diff_in_diffs(
        self, treatment_threshold: float = 1, min_treatment_periods: int = 3
    ):
        """
        Model 6: Staggered Diff-in-Diff (Event Study)

        ln(Questions_it) = α_i + γ_t + Σ_k β_k*Treatment_i × Post_k,t + X_it'δ + ε_it

        Where Treatment_i indicates when each MEP first receives high lobbying intensity,
        and Post_k,t indicates k periods after treatment.

        Based on: Callaway and Sant'Anna (2021), Sun and Abraham (2021) - staggered DiD
        """
        print(f"\n=== Model 6b: Staggered Diff-in-Diff ({self.topic}) ===")

        try:
            _, _, log_question_col, log_lobbying_col = self.get_topic_variables()

            # Create staggered diff-in-diffs dataset
            df_staggered = self.df.copy()

            # Find first treatment period for each MEP
            df_staggered["high_lobbying"] = (
                df_staggered[log_lobbying_col] > treatment_threshold
            ).astype(int)

            # Group by MEP and find first period with high lobbying
            mep_treatment_dates = {}
            for mep_id in df_staggered.index.get_level_values(0).unique():
                mep_data = df_staggered.loc[mep_id]
                high_lobbying_periods = mep_data[mep_data["high_lobbying"] == 1].index

                if len(high_lobbying_periods) >= min_treatment_periods:
                    first_treatment = high_lobbying_periods[0]
                    mep_treatment_dates[mep_id] = first_treatment
                else:
                    mep_treatment_dates[mep_id] = None

            # Create treatment indicators
            df_staggered["treatment_date"] = df_staggered.index.get_level_values(0).map(
                mep_treatment_dates
            )
            df_staggered["ever_treated"] = (
                df_staggered["treatment_date"].notna().astype(int)
            )

            # Calculate relative time to treatment
            df_staggered["relative_time"] = (
                df_staggered.index.get_level_values(1) - df_staggered["treatment_date"]
            ).dt.days / 30  # Convert to months

            # Create event study indicators (pre-treatment: -3 to -1, post-treatment: 0 to 3)
            event_periods = list(
                range(-min_treatment_periods, min_treatment_periods + 1)
            )

            for period in event_periods:
                if period < 0:
                    # Pre-treatment periods
                    df_staggered[f"pre_{abs(period)}"] = (
                        (df_staggered["ever_treated"] == 1)
                        & (df_staggered["relative_time"].round(0) == period)
                    ).astype(int)
                else:
                    # Post-treatment periods
                    df_staggered[f"post_{period}"] = (
                        (df_staggered["ever_treated"] == 1)
                        & (df_staggered["relative_time"].round(0) == period)
                    ).astype(int)

            # Create treatment × post interaction (standard DiD)
            df_staggered["post_treatment"] = (
                df_staggered["relative_time"] >= 0
            ).astype(int)
            df_staggered["treatment_x_post"] = (
                df_staggered["ever_treated"] * df_staggered["post_treatment"]
            )

            # Summary statistics
            treated_meps = len(
                df_staggered[df_staggered["ever_treated"] == 1]
                .index.get_level_values(0)
                .unique()
            )
            total_meps = len(df_staggered.index.get_level_values(0).unique())

            print(
                f"Treated MEPs: {treated_meps} out of {total_meps} ({treated_meps/total_meps*100:.1f}%)"
            )

            # Run staggered DiD regression
            event_vars = [f"pre_{abs(p)}" for p in range(1, min_treatment_periods + 1)] + [
                f"post_{p}" for p in range(min_treatment_periods)
            ]

            model = PanelOLS(
                dependent=df_staggered[log_question_col],
                exog=df_staggered[
                    ["treatment_x_post"] + event_vars + self.control_vars
                ],
                entity_effects=True,
                time_effects=True,
            )

            results = model.fit()

            # Extract results
            did_coefficient = results.params["treatment_x_post"]
            did_p_value = results.pvalues["treatment_x_post"]

            print(f"Staggered DiD coefficient: {did_coefficient:.4f}")
            print(f"DiD P-value: {did_p_value:.4f}")
            print(f"R-squared: {results.rsquared:.4f}")
            print(f"N observations: {results.nobs}")

            # Event study coefficients
            event_coefficients = {}
            for var in event_vars:
                if var in results.params:
                    event_coefficients[var] = {
                        "coefficient": results.params[var],
                        "p_value": results.pvalues[var],
                    }

            print("\nEvent Study Coefficients:")
            for period in range(-min_treatment_periods, min_treatment_periods + 1):
                if period < 0:
                    var = f"pre_{abs(period)}"
                else:
                    var = f"post_{period}"

                if var in event_coefficients:
                    coef = event_coefficients[var]["coefficient"]
                    p_val = event_coefficients[var]["p_value"]
                    sig = (
                        "***"
                        if p_val < 0.01
                        else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                    )
                    print(f"  Period {period:2d}: {coef:8.4f} {sig}")

            # Parallel trends test (pre-treatment coefficients should be zero)
            pre_coefficients = [
                event_coefficients.get(f"pre_{abs(p)}", {}).get("coefficient", 0)
                for p in range(1, min_treatment_periods + 1)
            ]
            pre_p_values = [
                event_coefficients.get(f"pre_{abs(p)}", {}).get("p_value", 1)
                for p in range(1, min_treatment_periods + 1)
            ]

            # Test if pre-treatment coefficients are jointly zero
            pre_significant = any(p < 0.05 for p in pre_p_values)

            print(f"\nParallel Trends Test:")
            if pre_significant:
                print(
                    "⚠ Some pre-treatment coefficients are significant - parallel trends may be violated"
                )
            else:
                print(
                    "✓ Pre-treatment coefficients are not significant - parallel trends assumption holds"
                )

            return {
                "model": f"Staggered Diff-in-Diff ({self.topic})",
                "elasticity": did_coefficient,
                "p_value": did_p_value,
                "r_squared": results.rsquared,
                "n_obs": results.nobs,
                "treated_meps": treated_meps,
                "total_meps": total_meps,
                "treatment_threshold": treatment_threshold,
                "event_coefficients": event_coefficients,
                "parallel_trends_violated": pre_significant,
                "results": results,
            }

        except Exception as e:
            print(f"Error in Staggered DiD: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_all_models(self, treatment_threshold=1, min_treatment_periods=2):
        """
        Run all available models for the current topic.

        Args:
            treatment_threshold (int): Treatment threshold for staggered diff-in-diffs
            min_treatment_periods (int): Minimum treatment periods for staggered diff-in-diffs

        Returns:
            dict: Dictionary containing results from all models
        """
        if self.topic is None:
            raise ValueError(
                "Topic must be set before running models. Use set_topic() method."
            )

        print(f"\n{'='*60}")
        print(f"RUNNING ALL MODELS FOR TOPIC: {self.topic.upper()}")
        print(f"{'='*60}")

        all_results = {}

        # Run all models
        all_results["basic_fixed_effects"] = self.model_basic_fixed_effects()
        all_results["heterogeneous_effects"] = self.model_heterogeneous_effects()
        all_results["nonlinear_effects"] = self.model_nonlinear_effects()
        all_results["propensity_score_matching"] = (
            self.model_propensity_score_matching()
        )
        all_results["instrumental_variables"] = self.model_instrumental_variables()
        all_results["staggered_diff_in_diffs"] = self.model_staggered_diff_in_diffs(
            treatment_threshold=treatment_threshold,
            min_treatment_periods=min_treatment_periods,
        )

        self.results = all_results
        return all_results

    def create_summary_table(self, results):
        """
        Create a summary table of model results.

        Args:
            results (dict): Results from run_all_models()

        Returns:
            pd.DataFrame: Summary table
        """
        summary_data = []

        for model_name, result in results.items():
            if result is not None and "elasticity" in result:
                summary_data.append(
                    {
                        "Model": result["model"],
                        "Elasticity": result["elasticity"],
                        "P-value": result["p_value"],
                        "R-squared": result["r_squared"],
                        "N": result["n_obs"],
                    }
                )

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            print("\n=== Summary of All Models ===")
            print(summary_df.to_string(index=False))
        else:
            print("No successful models to summarize.")

        return summary_df

    def plot_results(self, summary_df):
        """
        Create visualization of results.

        Args:
            summary_df (pd.DataFrame): Summary table from create_summary_table()
        """
        if summary_df.empty:
            print("No results to plot - all models failed.")
            return

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot elasticities
            ax1.bar(summary_df["Model"], summary_df["Elasticity"])
            ax1.set_title(f"Treatment Effects for {self.topic}")
            ax1.set_ylabel("Elasticity")
            ax1.tick_params(axis="x", rotation=45)
            ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)

            # Plot R-squared
            ax2.bar(summary_df["Model"], summary_df["R-squared"])
            ax2.set_title(f"Model Fit for {self.topic} (R-squared)")
            ax2.set_ylabel("R-squared")
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                f"{self.topic}_lobbying_effects_results.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

            print(f"Results plot saved as '{self.topic}_lobbying_effects_results.png'")
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Summary table:")
            print(summary_df)


def run_cross_topic_analysis(
    topics, time_frequency="monthly", treatment_threshold=1, min_treatment_periods=2
):
    """
    Run analysis across multiple topics.

    Args:
        topics (list): List of topics to analyze
        time_frequency (str): Either "monthly" or "weekly"

    Returns:
        dict: Results for all topics
    """
    print("=== Cross-Topic Analysis ===")

    # Load data once
    database = DataBase()
    df_filtered, column_sets = database.prepare_data(time_frequency)

    # Initialize model
    model = LobbyingEffectsModel(df_filtered, column_sets)

    all_topic_results = {}

    for topic in topics:
        print(f"\n{'='*50}")
        print(f"ANALYZING TOPIC: {topic.upper()}")
        print(f"{'='*50}")

        model.set_topic(topic)
        results = model.run_all_models(
            treatment_threshold=treatment_threshold,
            min_treatment_periods=min_treatment_periods,
        )
        summary_df = model.create_summary_table(results)

        all_topic_results[topic] = {"results": results, "summary": summary_df}

    return all_topic_results


# Example usage
if __name__ == "__main__":
    # Example 1: Single topic analysis
    print("Example 1: Single Topic Analysis")
    print("=" * 50)
    # Load data
    database = DataBase()
    df_filtered, column_sets = database.prepare_data()

    # Create model and analyze agriculture
    model = LobbyingEffectsModel(df_filtered, column_sets)
    model.set_topic("agriculture")
    results = model.run_all_models()
    summary_df = model.create_summary_table(results)
    model.plot_results(summary_df)

    # Example 2: Cross-topic analysis
    print("\n\nExample 2: Cross-Topic Analysis")
    print("=" * 50)

    topics = ["agriculture", "technology", "health", "environment and climate"]
    cross_topic_results = run_cross_topic_analysis(topics)
