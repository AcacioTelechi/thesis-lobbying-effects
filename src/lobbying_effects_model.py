"""
Lobbying Effects Analysis Model Classes

This module contains two main classes:
1. DataBase: Responsible for loading and treating panel data
2. LobbyingEffectsModel: Responsible for running econometric models with different topics
"""
import sys
sys.path.append("D:\\repos\\pessoal\\thesis-lobbying-effects")
from src.database import DataBase

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from linearmodels import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import numpy as np
import subprocess
import json
import os
import tempfile
import re


warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")


class LobbyingEffectsModel:
    """
    Class responsible for running econometric models with different Ltopics.
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

    # =============================
    # Continuous-treatment DDD (long panel) utilities
    # =============================
    def prepare_long_panel(self) -> pd.DataFrame:
        """
        Reshape the current wide panel (member_id × time) into a long panel over domains:
        (member_id × domain × time), with outcome (questions) and treatment (meetings).

        Returns:
            pd.DataFrame with columns: [member_id, domain, time, questions, meetings]
            and a MultiIndex set to (member_id_domain, time) for FE estimation convenience.
        """
        # Detect index names
        if not isinstance(self.df.index, pd.MultiIndex) or len(self.df.index.names) < 2:
            raise ValueError("Expected MultiIndex with ['member_id', time].")

        entity_col = self.df.index.names[0]
        time_col = self.df.index.names[1]

        # Infer available domains by intersecting questions and meetings columns
        questions_prefix = "questions_infered_topic_"
        meetings_prefix = "meetings_l_"

        question_cols = [c for c in self.df.columns if c.startswith(questions_prefix)]
        meeting_cols = [c for c in self.df.columns if c.startswith(meetings_prefix)
                        and  'category' not in c
                        and 'budget' not in c
                        and 'days_since' not in c]

        # Handle renamed columns (spaces replaced by underscores already)
        # Domains are the suffixes after the prefixes
        domains_q = {c.replace(questions_prefix, "").replace(" ", "_") for c in question_cols}
        domains_m = {c.replace(meetings_prefix, "").replace(" ", "_") for c in meeting_cols}
        domains = sorted(list(domains_q.intersection(domains_m)))

        print(f"Number of domains: {len(domains)}")

        print(f"Domains: {domains}")

        if len(domains) == 0:
            raise ValueError(
                "Could not infer domains. Ensure columns like 'questions_infered_topic_<domain>' and 'meetings_l_<domain>' exist."
            )

        # Build long dataframe by vertical concatenation per domain
        long_frames: list[pd.DataFrame] = []
        base_df = self.df.reset_index()
        for d in domains:
            q_col = f"{questions_prefix}{d}"
            m_col = f"{meetings_prefix}{d}"
            if q_col in base_df.columns and m_col in base_df.columns:
                tmp = base_df[[entity_col, time_col, q_col, m_col]].copy()
                tmp.rename(columns={q_col: "questions", m_col: "meetings"}, inplace=True)
                tmp["domain"] = d
                long_frames.append(tmp)

        if not long_frames:
            raise ValueError("No domain frames could be created. Check input columns.")

        df_long = pd.concat(long_frames, ignore_index=True)

        # Sort and create a combined entity key for member_id × domain
        df_long.sort_values([entity_col, "domain", time_col], inplace=True)
        df_long["member_domain"] = (
            df_long[entity_col].astype(str) + "__" + df_long["domain"].astype(str)
        )

        # Set index for PanelOLS compatibility (entity, time)
        df_long.set_index(["member_domain", time_col], inplace=True)

        return df_long

    def model_continuous_ddd_linear(self, include_domain_time_fe: bool = False):
        """
        Continuous-treatment FE model in long panel (pure Python, linear FE fallback).

        Specification (default, memory-friendly):
            y_{i,d,t} = alpha_{i×d} + tau_t + beta * meetings_{i,d,t} + e_{i,d,t}
        where alpha_{i×d} are entity effects using member_id×domain as the entity;
        tau_t are month FE (PanelOLS time_effects=True).

        Optionally (include_domain_time_fe=True), add domain×time dummies as controls
        to better absorb domain-specific monthly shocks. This increases memory usage.

        Returns:
            dict with model summary, coefficient on meetings, p-value, R-squared, N.
        """
        try:
            df_long = self.prepare_long_panel()

            # Build exogenous matrix
            exog_parts = [df_long[["meetings"]]]

            if include_domain_time_fe:
                # Add domain×time dummy variables (can be memory heavy)
                # Recover time level name from index
                time_level = df_long.index.names[1]
                # Retrieve as columns for dummies
                tmp = df_long.reset_index()
                tmp["domain_time"] = tmp["domain"].astype(str) + "__" + tmp[time_level].astype(str)
                dummies = pd.get_dummies(tmp["domain_time"], prefix="dt", drop_first=True)
                # Align back to df_long index
                dummies.index = df_long.index
                exog_parts.append(dummies)

            exog = pd.concat(exog_parts, axis=1)

            # PanelOLS requires aligned dependent/independent with same index
            dependent = df_long["questions"]

            model = PanelOLS(
                dependent=dependent,
                exog=exog,
                entity_effects=True,  # absorbs member_id×domain FE via entity index
                time_effects=True,    # month FE
            )

            results = model.fit()

            beta = results.params.get("meetings", float("nan"))
            p_value = results.pvalues.get("meetings", float("nan"))
            r2 = results.rsquared

            print("\n=== Continuous-Treatment FE (Linear) ===")
            print(f"Coefficient on meetings: {beta:.6f}")
            print(f"P-value: {p_value:.6f}")
            print(f"R-squared: {r2:.4f}")
            print(f"N observations: {results.nobs}")

            return {
                "model": "Continuous FE (Linear)",
                "coefficient": beta,
                "p_value": p_value,
                "r_squared": r2,
                "n_obs": results.nobs,
                "results": results,
                "include_domain_time_fe": include_domain_time_fe,
            }
        except Exception as e:
            print(f"Error in continuous-treatment FE model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def model_continuous_ddd_ppml(
        self,
        include_domain_time_fe: bool = True,
        include_member_fe: bool = True,
        max_fe_columns: int = 8000,
        maxiter: int = 100,
    ):
        """
        PPML estimator (Poisson GLM with log link) for the long panel, using dummy FEs.

        Practical, pure-Python implementation:
          - Always includes a constant and the continuous treatment 'meetings'.
          - By default includes domain×time fixed effects via dummies (manageable size).
          - Optionally includes member fixed effects (member_id) via dummies.

        Notes:
          - We avoid member×domain FEs here due to memory constraints when using dummies.
          - Use the linear FE variant for saturated FEs; this PPML is a strong robustness.

        Returns:
          dict with coefficient on 'meetings', p-value (clustered by member), and fit info.
        """
        try:
            df_long = self.prepare_long_panel()
            time_col = self.df.index.names[1]

            work = df_long.reset_index()  # columns: member_domain, time, questions, meetings, domain

            # Reconstruct member_id from member_domain composite key
            work["member_id"] = work["member_domain"].astype(str).str.split("__").str[0]

            # Ensure core numeric columns
            work["meetings"] = pd.to_numeric(work["meetings"], errors="coerce").astype(float)
            work["questions"] = pd.to_numeric(work["questions"], errors="coerce").astype(float)

            X_parts = [work[["meetings"]]]

            # Domain × time FEs (default)
            if include_domain_time_fe:
                work["domain_time"] = work["domain"].astype(str) + "__" + work[time_col].astype(str)
                dt_dum = pd.get_dummies(
                    work["domain_time"], prefix="dt", drop_first=True, dtype=float
                )
                X_parts.append(dt_dum)

            # Member FE (optional)
            if include_member_fe:
                mem_dum = pd.get_dummies(
                    work["member_id"], prefix="m", drop_first=True, dtype=float
                )
                X_parts.append(mem_dum)

            # Concatenate design matrix
            X = pd.concat(X_parts, axis=1)
            # Enforce float dtype
            X = X.apply(pd.to_numeric, errors="coerce").astype(float)

            # Guardrail for memory: cap number of columns
            if X.shape[1] > max_fe_columns:
                print(
                    f"Warning: Too many FE columns ({X.shape[1]} > {max_fe_columns}). "
                    "Dropping member FE to reduce dimensionality."
                )
                # Rebuild without member FE
                X_parts = [work[["meetings"]]]
                if include_domain_time_fe:
                    X_parts.append(dt_dum)
                X = pd.concat(X_parts, axis=1)
                X = X.apply(pd.to_numeric, errors="coerce").astype(float)
                include_member_fe_final = False
            else:
                include_member_fe_final = include_member_fe

            y = pd.to_numeric(work["questions"], errors="coerce").astype(float)

            # Add constant
            X_const = pd.concat(
                [pd.Series(1.0, index=X.index, name="const"), X], axis=1
            )
            X_const = X_const.apply(pd.to_numeric, errors="coerce").astype(float)

            # Drop rows with NaN or inf in y or X
            mask_y = np.isfinite(y.to_numpy())
            mask_X = np.isfinite(X_const.to_numpy()).all(axis=1)
            is_finite = mask_y & mask_X

            y_clean = y.loc[is_finite]
            X_clean = X_const.loc[is_finite]
            groups_series = work["member_id"].astype("category").cat.codes
            groups = groups_series.loc[is_finite].to_numpy()

            # Fit Poisson GLM (PPML)
            glm_mod = sm.GLM(y_clean, X_clean, family=sm.families.Poisson())
            glm_res = glm_mod.fit(maxiter=maxiter, cov_type="cluster", cov_kwds={"groups": groups})

            beta = glm_res.params.get("meetings", float("nan"))
            p_value = glm_res.pvalues.get("meetings", float("nan"))

            print("\n=== Continuous-Treatment PPML (GLM Poisson) ===")
            print(f"Coefficient on meetings (semi-elasticity): {beta:.6f}")
            print(f"P-value (clustered by member): {p_value:.6f}")
            print(f"N observations: {int(glm_res.nobs)}")
            print(f"K parameters: {len(glm_res.params)}")

            return {
                "model": "Continuous PPML (Poisson GLM)",
                "coefficient": beta,
                "p_value": p_value,
                "n_obs": int(glm_res.nobs),
                "k_params": len(glm_res.params),
                "include_domain_time_fe": include_domain_time_fe,
                "include_member_fe": include_member_fe_final,
                "results": glm_res,
            }
        except Exception as e:
            print(f"Error in PPML model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def model_continuous_ddd_ppml_fixest_rscript(
        self,
        Rscript_path: str | None = None,
        include_member_time_fe: bool = False,
        fe_pattern: str | None = None,
        domain_filter: str | None = None,
        cluster_by: str = "member_id",
        timeout_seconds: int = 300,
        include_controls: bool = True,
        domain_varying_slopes: bool = False,
        include_meetings_squared: bool = False,
        include_lead1_meetings: bool = False,
        include_lag1_meetings: bool = False,
        trim_top_fraction: float | None = None,
        include_lags: int = 0,
        include_leads: int = 0,
        use_share_outcome: bool = False,
        alt_treatment_var: str | None = None,
        vary_by: str | None = None,
    ):
        """
        PPML using R fixest::fepois by calling Rscript (no rpy2 needed).

        Args:
          Rscript_path: Optional full path to Rscript.exe (Windows), e.g., 'D:\\R-4.4.1\\bin\\x64\\Rscript.exe'.
          include_member_time_fe: include member×time fixed effects.
          cluster_by: column to cluster SEs by (default: member_id).

        Returns:
          dict with coefficient, p-value, n_obs; or None on error.
        """
        try:
            df_long = self.prepare_long_panel().reset_index()
            time_col = self.df.index.names[1]

            # Optional domain filter for per-domain estimation
            if domain_filter is not None:
                df_long = df_long[df_long["domain"].astype(str) == str(domain_filter)].copy()
                if df_long.empty:
                    print(f"No rows after filtering for domain='{domain_filter}'.")
                    return None

            # Reconstruct member_id from member_domain
            df_long["member_id"] = df_long["member_domain"].astype(str).str.split("__").str[0]
            df_long["domain_time"] = df_long["domain"].astype(str) + "__" + df_long[time_col].astype(str)
            df_long["member_time"] = df_long["member_id"].astype(str) + "__" + df_long[time_col].astype(str)

            # Ensure numeric outcome and treatment
            df_long["questions"] = pd.to_numeric(df_long["questions"], errors="coerce")
            df_long["meetings"] = pd.to_numeric(df_long["meetings"], errors="coerce")

            # Optional share outcome within member_id × time (reallocation test)
            if use_share_outcome:
                entity_col = self.df.index.names[0]
                # ensure entity_col exists
                if entity_col not in df_long.columns:
                    df_long[entity_col] = df_long["member_domain"].astype(str).str.split("__").str[0]
                totals = df_long.groupby([entity_col, time_col])["questions"].transform("sum")
                df_long.loc[totals > 0, "questions"] = df_long.loc[totals > 0, "questions"] / totals[totals > 0]
                df_long.loc[totals == 0, "questions"] = float("nan")

            # Optional trimming to reduce influence of extreme intensity values
            if isinstance(trim_top_fraction, (int, float)) and 0 < float(trim_top_fraction) < 1:
                frac = float(trim_top_fraction)
                upper = df_long["meetings"].quantile(1 - frac)
                df_long = df_long[df_long["meetings"] <= upper].copy()

            # Optional dynamics: 1-period lead/lag and K leads/lags within member_domain
            if include_lead1_meetings or include_lag1_meetings or (isinstance(include_lags, int) and include_lags > 0) or (isinstance(include_leads, int) and include_leads > 0):
                # Sort to ensure proper shifting
                df_long.sort_values(["member_domain", time_col], inplace=True)
                if include_lead1_meetings:
                    df_long["lead1_meetings"] = (
                        df_long.groupby("member_domain")["meetings"].shift(-1)
                    )
                if include_lag1_meetings:
                    df_long["lag1_meetings"] = (
                        df_long.groupby("member_domain")["meetings"].shift(1)
                    )
                if isinstance(include_lags, int) and include_lags > 0:
                    for k in range(1, include_lags + 1):
                        df_long[f"lag{k}_meetings"] = df_long.groupby("member_domain")["meetings"].shift(k)
                if isinstance(include_leads, int) and include_leads > 0:
                    for k in range(1, include_leads + 1):
                        df_long[f"lead{k}_meetings"] = df_long.groupby("member_domain")["meetings"].shift(-k)

            df_long = df_long.dropna(subset=["questions", "meetings"])  # drop rows with missing core vars

            # Optionally add controls (only meaningful when member_time FE is OFF)
            control_cols: list[str] = []
            if include_controls and not include_member_time_fe:
                potential_sets = [
                    "MEPS_POLITICAL_GROUP_COLUMNS",
                    "MEPS_COUNTRY_COLUMNS",
                    "MEPS_POSITIONS_COLUMNS",
                    # "MEETINGS_CATEGORY_COLUMNS",
                    "MEETINGS_MEMBER_CAPACITY_COLUMNS",
                ]
                for set_name in potential_sets:
                    if set_name in self.column_sets:
                        for c in self.column_sets[set_name]:
                            if c in self.df.columns:
                                control_cols.append(c)

                # Merge controls from the original wide df (indexed by member_id, time)
                orig = self.df.reset_index()
                # Ensure the join key name exists in df_long
                entity_col = self.df.index.names[0]
                if entity_col not in df_long.columns:
                    # Reconstruct entity_col from member_domain
                    df_long[entity_col] = df_long["member_domain"].astype(str).str.split("__").str[0]

                join_cols = [entity_col, time_col]
                merge_cols = [col for col in control_cols if col in orig.columns]
                if merge_cols:
                    df_long = df_long.merge(orig[join_cols + merge_cols], on=join_cols, how="left")

            # Build a simple time FE column for custom FE patterns
            df_long["time_fe"] = df_long[time_col].astype(str)

            # Keep only required columns to minimize IO
            required_cols = [
                "questions",
                "meetings",
                "member_domain",
                "domain_time",
                "domain",
                "time_fe",
            ]
            if include_member_time_fe:
                required_cols.append("member_time")

            # Add optional dynamic and nonlinear columns
            if include_meetings_squared:
                # We'll use formula I(meetings^2) in R; no extra column needed
                pass
            if include_lead1_meetings:
                required_cols.append("lead1_meetings")
            if include_lag1_meetings:
                required_cols.append("lag1_meetings")
            if isinstance(include_lags, int) and include_lags > 0:
                for k in range(1, include_lags + 1):
                    required_cols.append(f"lag{k}_meetings")
            if isinstance(include_leads, int) and include_leads > 0:
                for k in range(1, include_leads + 1):
                    required_cols.append(f"lead{k}_meetings")

            # Ensure cluster columns exist (supports multi-way like 'member_id + time_fe')
            cluster_terms = [c.strip() for c in str(cluster_by).split('+')]
            missing_terms = []
            for term in cluster_terms:
                if term in df_long.columns:
                    if term not in required_cols:
                        required_cols.append(term)
                else:
                    missing_terms.append(term)
            if missing_terms and len(cluster_terms) == 1:
                print(f"Warning: cluster_by='{cluster_by}' not found. Falling back to 'member_id'.")
                cluster_by = "member_id"
                if "member_id" not in required_cols:
                    required_cols.append("member_id")
            elif missing_terms:
                print("Warning: some cluster columns not found and won't be added to CSV: " + ", ".join(missing_terms))

            # Ensure FE variables referenced in fe_pattern are present in the CSV
            if isinstance(fe_pattern, str) and fe_pattern:
                fp = fe_pattern.strip().lower()
                fe_needed: list[str] = []
                if fp == "member+time":
                    fe_needed = ["member_id", "time_fe"]
                elif fp == "member_domain+domain_time":
                    fe_needed = ["member_domain", "domain_time"]
                elif fp == "member_domain+time":
                    fe_needed = ["member_domain", "time_fe"]
                elif fp == "member+domain_time":
                    fe_needed = ["member_id", "domain_time"]
                elif fp == "member_time+domain":
                    fe_needed = ["member_time", "domain"]
                elif fp == "member_time":
                    fe_needed = ["member_time"]
                for term in fe_needed:
                    if term in df_long.columns and term not in required_cols:
                        required_cols.append(term)
            # Add controls if any, and sanitize names for R
            safe_name_map: dict[str, str] = {}
            if include_controls and not include_member_time_fe and control_cols:
                def r_safe(name: str) -> str:
                    # Similar to make.names: replace invalid with '.', prefix 'X' if starts with digit
                    s = re.sub(r"[^0-9A-Za-z_.]", ".", name)
                    if re.match(r"^[0-9]", s):
                        s = f"X{s}"
                    return s

                for c in control_cols:
                    if c in df_long.columns:
                        safe = r_safe(c)
                        # Ensure uniqueness if collision
                        if safe in df_long.columns and safe != c:
                            idx = 1
                            candidate = f"{safe}.{idx}"
                            while candidate in df_long.columns:
                                idx += 1
                                candidate = f"{safe}.{idx}"
                            safe = candidate
                        safe_name_map[c] = safe
                if safe_name_map:
                    df_long.rename(columns=safe_name_map, inplace=True)
                present_cols = set(df_long.columns) | set(safe_name_map.values())
                required_cols.extend([safe_name_map.get(c, c) for c in control_cols if (safe_name_map.get(c, c) in present_cols)])

            # Ensure alt_treatment_var column exists before slicing
            if alt_treatment_var is not None:
                orig = self.df.reset_index()
                if alt_treatment_var in df_long.columns:
                    if alt_treatment_var not in required_cols:
                        required_cols.append(alt_treatment_var)
                elif alt_treatment_var in orig.columns:
                    # Merge in the alternate treatment
                    entity_col = self.df.index.names[0]
                    if entity_col not in df_long.columns:
                        df_long[entity_col] = df_long["member_domain"].astype(str).str.split("__").str[0]
                    join_cols = [entity_col, time_col]
                    df_long = df_long.merge(orig[join_cols + [alt_treatment_var]], on=join_cols, how="left")
                    required_cols.append(alt_treatment_var)

            df_long = df_long[required_cols].copy()

            # Write temp CSV and R script
            with tempfile.TemporaryDirectory() as td:
                input_csv = os.path.join(td, "ppml_input.csv")
                print(f"Writing input CSV to {input_csv}")
                df_long.to_csv(input_csv, index=False)

                r_script_path = os.path.join(td, "ppml_fixest.R")
                output_json = os.path.join(td, "ppml_output.json")
                fe_rhs = "member_domain + domain_time + member_time" if include_member_time_fe else "member_domain + domain_time"
                # Determine FE RHS according to requested pattern
                if fe_pattern is not None:
                    # Supported patterns: 'member_domain+domain_time', 'member+time', 'member_domain+time', 'member+domain_time'
                    pattern = fe_pattern.strip().lower()
                    if pattern == "member+time":
                        fe_rhs = "member_id + time_fe"
                    elif pattern == "member_domain+time":
                        fe_rhs = "member_domain + time_fe"
                    elif pattern == "member+domain_time":
                        fe_rhs = "member_id + domain_time"
                    elif pattern == "member_time+domain":
                        fe_rhs = "member_time + domain"
                    elif pattern == "member_time":
                        fe_rhs = "member_time"
                    else:
                        fe_rhs = "member_domain + domain_time"
                else:
                    fe_rhs = "member_domain + domain_time + member_time" if include_member_time_fe else "member_domain + domain_time"
                
                # Treatment term: pooled or domain-varying slopes
                if domain_varying_slopes:
                    # baseline slope for reference domain + deltas for others
                    treatment_term = "meetings + i(domain, meetings)"
                else:
                    treatment_term = "meetings"

                # Nonlinearity: add squared meetings
                if include_meetings_squared:
                    treatment_term = treatment_term + " + I(meetings^2)"

                # Dynamics: leads/lags if requested
                if include_lead1_meetings:
                    treatment_term = treatment_term + " + lead1_meetings"
                if include_lag1_meetings:
                    treatment_term = treatment_term + " + lag1_meetings"
                if isinstance(include_lags, int) and include_lags > 0:
                    for k in range(1, include_lags + 1):
                        treatment_term = treatment_term + f" + lag{k}_meetings"
                if isinstance(include_leads, int) and include_leads > 0:
                    for k in range(1, include_leads + 1):
                        treatment_term = treatment_term + f" + lead{k}_meetings"
                # Build controls RHS for R formula (as a single string)
                controls_rhs_str = ""
                if include_controls and not include_member_time_fe and control_cols:
                    rhs_controls = [safe_name_map.get(c, c) for c in control_cols if safe_name_map.get(c, c) in df_long.columns]
                    if rhs_controls:
                        controls_rhs_str = " + " + " + ".join(rhs_controls)
                # Treatment variable override and vary_by interaction support
                treat_var = "meetings"
                if alt_treatment_var is not None and alt_treatment_var in df_long.columns:
                    treat_var = alt_treatment_var
                    if treat_var not in required_cols:
                        required_cols.append(treat_var)
                treatment_term = treatment_term.replace("meetings", treat_var)
                if vary_by is not None and vary_by in df_long.columns:
                    treatment_term = treatment_term.replace(f"i(domain, {treat_var})", f"i({vary_by}, {treat_var})")
                    if vary_by not in required_cols:
                        required_cols.append(vary_by)
                # Build full formula string in Python to avoid R paste/quoting issues
                formula_str = f"questions ~ {treatment_term}{controls_rhs_str} | {fe_rhs}"
                r_code = (
                    """
options(warn=1)
library(fixest); 
library(jsonlite)
df <- read.csv('"""
                    + input_csv.replace("\\", "/")
                    + """', stringsAsFactors=FALSE)
# Ensure proper types
df$questions <- as.numeric(df$questions)
df$meetings <- as.numeric(df$meetings)
df$domain <- factor(df$domain)
if ("agriculture" %in% levels(df$domain)) {
  df$domain <- relevel(df$domain, ref = "agriculture")
}

                fml <- as.formula("""
                    + formula_str.replace("\"", "\\\"")
                    + """)
cl  <- as.formula('~ """
                    + cluster_by
                    + """')
fit <- tryCatch(fepois(fml, data=df, cluster=cl), error=function(e) e)
if (inherits(fit, 'error')) {
  write(toJSON(list(error=fit$message), auto_unbox=TRUE), file='"""
                    + output_json.replace("\\", "/")
                    + """'); quit(status=0)
}
sm <- summary(fit)
cv <- tryCatch(fit$collin.var, error=function(e) NULL)
ct <- sm$coeftable
rn <- rownames(ct)
sq_idx <- which(rn == 'I(meetings^2)')
sq_beta <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, 1])
sq_p <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, ncol(ct)])
# generic leads/lags
lag_idx <- grep('^lag[0-9]+_meetings$', rn)
lag_names <- rn[lag_idx]
lag_coefs <- if (length(lag_idx) == 0) list() else as.list(as.numeric(ct[lag_idx, 1]))
lag_pvals <- if (length(lag_idx) == 0) list() else as.list(as.numeric(ct[lag_idx, ncol(ct)]))
lead_idx <- grep('^lead[0-9]+_meetings$', rn)
lead_names <- rn[lead_idx]
lead_coefs <- if (length(lead_idx) == 0) list() else as.list(as.numeric(ct[lead_idx, 1]))
lead_pvals <- if (length(lead_idx) == 0) list() else as.list(as.numeric(ct[lead_idx, ncol(ct)]))
 # specific 1-step for backward-compat
l1_idx <- which(rn == 'lead1_meetings')
l1_beta <- if (length(l1_idx) == 0) NA else unname(ct[l1_idx, 1])
l1_p <- if (length(l1_idx) == 0) NA else unname(ct[l1_idx, ncol(ct)])
lg1_idx <- which(rn == 'lag1_meetings')
lg1_beta <- if (length(lg1_idx) == 0) NA else unname(ct[lg1_idx, 1])
lg1_p <- if (length(lg1_idx) == 0) NA else unname(ct[lg1_idx, ncol(ct)])
out <- list()
if ("""
                    + ("TRUE" if domain_varying_slopes else "FALSE")
                    + """) {
  # base slope for ref domain (coefficient of 'meetings')
  base_idx <- which(rn == 'meetings')
  base_coef <- if (length(base_idx) == 0) NA else unname(ct[base_idx, 1])
  base_p <- if (length(base_idx) == 0) NA else unname(ct[base_idx, ncol(ct)])
  # deltas for other domains
  d_idx <- grepl(':meetings$', rn)
  d_rows <- rn[d_idx]
  d_names <- sub('^.*::', '', d_rows)
  d_names <- sub(':meetings$', '', d_names)
  d_coefs <- if (any(d_idx)) as.numeric(ct[d_idx, 1]) else numeric(0)
  d_pvals <- if (any(d_idx)) as.numeric(ct[d_idx, ncol(ct)]) else numeric(0)
  out$base_domain <- as.character(levels(as.factor(df$domain))[1])
  out$base_coef <- base_coef
  out$base_p <- base_p
  out$delta_domains <- as.list(d_names)
  out$delta_coefs <- as.list(d_coefs)
  out$delta_pvals <- as.list(d_pvals)
  out$n_obs <- as.integer(nobs(fit))
  out$collin_var <- as.list(cv)
  out$squared_coef <- sq_beta
  out$squared_p <- sq_p
  out$lead1_coef <- l1_beta
  out$lead1_p <- l1_p
  out$lag1_coef <- lg1_beta
  out$lag1_p <- lg1_p
  out$lag_names <- as.list(lag_names)
  out$lag_coefs <- lag_coefs
  out$lag_pvals <- lag_pvals
  out$lead_names <- as.list(lead_names)
  out$lead_coefs <- lead_coefs
  out$lead_pvals <- lead_pvals
} else {
  idx <- which(rn == 'meetings')
  beta <- if (length(idx) == 0) NA else unname(ct[idx, 1])
  p <- if (length(idx) == 0) NA else unname(ct[idx, ncol(ct)])
  out$beta <- beta
  out$p_value <- p
  out$n_obs <- as.integer(nobs(fit))
  out$squared_coef <- sq_beta
  out$squared_p <- sq_p
  out$lead1_coef <- l1_beta
  out$lead1_p <- l1_p
  out$lag1_coef <- lg1_beta
  out$lag1_p <- lg1_p
  out$lag_names <- as.list(lag_names)
  out$lag_coefs <- lag_coefs
  out$lag_pvals <- lag_pvals
  out$lead_names <- as.list(lead_names)
  out$lead_coefs <- lead_coefs
  out$lead_pvals <- lead_pvals
}
write(toJSON(out, auto_unbox=TRUE), file='"""
                    + output_json.replace("\\", "/")
                    + """')
quit(status=0)
"""
                )
                print(f"Writing R script to {r_script_path}")
                with open(r_script_path, "w", encoding="utf-8") as f:
                    f.write(r_code)

                # Determine Rscript command
                chosen_rscript = None
                if Rscript_path:
                    # Auto-correct if a path to R.exe was provided instead of Rscript.exe
                    base = os.path.basename(Rscript_path).lower()
                    if base in ("r.exe", "r"):
                        candidate = os.path.join(os.path.dirname(Rscript_path), "Rscript.exe")
                        if os.path.exists(candidate):
                            chosen_rscript = candidate
                        else:
                            # Try x64 subfolder
                            candidate2 = os.path.join(os.path.dirname(Rscript_path), "x64", "Rscript.exe")
                            if os.path.exists(candidate2):
                                chosen_rscript = candidate2
                    else:
                        chosen_rscript = Rscript_path
                if not chosen_rscript:
                    chosen_rscript = "Rscript"

                cmd = [chosen_rscript]
                # Use forward slashes to avoid quoting issues on Windows
                r_script_path_arg = r_script_path.replace("\\", "/")
                cmd += ["--vanilla", r_script_path_arg]

                print(f"Running Rscript with command: {cmd}")
                # Run with timeout and captured output; set cwd to temp dir to avoid permission issues
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=td)
                print(f"Rscript output: {proc.stdout}")
                print(f"Rscript error: {proc.stderr}")
                if proc.returncode != 0:
                    print("Rscript failed:")
                    print(proc.stderr)
                    return None

                # Read JSON from file written by R
                if not os.path.exists(output_json):
                    print("R did not produce output JSON. Stdout/Stderr:")
                    print(proc.stdout)
                    print(proc.stderr)
                    return None

                with open(output_json, "r", encoding="utf-8") as jf:
                    res = json.load(jf)

                if "error" in res:
                    print(f"R fixest error: {res['error']}")
                    return None

                if domain_varying_slopes:
                    # Assemble per-domain slopes from base + deltas
                    base_domain = res.get("base_domain")
                    base_coef = res.get("base_coef")
                    base_p = res.get("base_p")
                    delta_domains = res.get("delta_domains", []) or []
                    delta_coefs = res.get("delta_coefs", []) or []
                    delta_pvals = res.get("delta_pvals", []) or []
                    # Optional dynamics/nonlinearity extras
                    squared_coef = res.get("squared_coef")
                    squared_p = res.get("squared_p")
                    lag_names = res.get("lag_names")
                    lag_coefs = res.get("lag_coefs")
                    lag_pvals = res.get("lag_pvals")
                    lead_names = res.get("lead_names")
                    lead_coefs = res.get("lead_coefs")
                    lead_pvals = res.get("lead_pvals")
                    slope_by_domain = {}
                    p_by_domain = {}
                    if base_domain is not None and base_coef is not None:
                        slope_by_domain[base_domain] = base_coef
                        p_by_domain[base_domain] = base_p
                    for name, dc, pv in zip(delta_domains, delta_coefs, delta_pvals):
                        if name is None or dc is None:
                            continue
                        slope_by_domain[name] = (base_coef if base_coef is not None else 0.0) + dc
                        p_by_domain[name] = pv
                    n_obs = res.get("n_obs", None)
                    print("\n=== Domain-varying slopes (fixest::fepois) ===")
                    print(f"N observations: {n_obs}")
                    return {
                        "model": "Continuous PPML (fixest via Rscript)",
                        "domain_varying_slopes": True,
                        "slope_by_domain": slope_by_domain,
                        "p_by_domain": p_by_domain,
                        "base_domain": base_domain,
                        "n_obs": n_obs,
                        "include_member_time_fe": include_member_time_fe,
                        "cluster_by": cluster_by,
                        "squared_coef": squared_coef,
                        "squared_p": squared_p,
                        "lag_names": lag_names,
                        "lag_coefs": lag_coefs,
                        "lag_pvals": lag_pvals,
                        "lead_names": lead_names,
                        "lead_coefs": lead_coefs,
                        "lead_pvals": lead_pvals,
                    }
                else:
                    beta = res.get("beta", None)
                    p_value = res.get("p_value", None)
                    n_obs = res.get("n_obs", None)
                    print("\n=== Continuous-Treatment PPML (Rscript fixest::fepois) ===")
                    print(f"Coefficient on meetings (semi-elasticity): {beta}")
                    print(f"P-value (clustered by {cluster_by}): {p_value}")
                    print(f"N observations: {n_obs}")
                    return {
                        "model": "Continuous PPML (fixest via Rscript)",
                        "coefficient": beta,
                        "p_value": p_value,
                        "n_obs": n_obs,
                        "include_member_time_fe": include_member_time_fe,
                        "cluster_by": cluster_by,
                        "squared_coef": res.get("squared_coef"),
                        "squared_p": res.get("squared_p"),
                        "lag_names": res.get("lag_names"),
                        "lag_coefs": res.get("lag_coefs"),
                        "lag_pvals": res.get("lag_pvals"),
                        "lead_names": res.get("lead_names"),
                        "lead_coefs": res.get("lead_coefs"),
                        "lead_pvals": res.get("lead_pvals"),
                    }

        except Exception as e:
            print(f"Error in Rscript-backed PPML: {e}")
            import traceback
            traceback.print_exc()
            return None

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
    # print("Example 1: Single Topic Analysis")
    # print("=" * 50)
    # # Load data
    # database = DataBase()
    # df_filtered, column_sets = database.prepare_data()

    # # Create model and analyze agriculture
    # model = LobbyingEffectsModel(df_filtered, column_sets)
    # model.set_topic("agriculture")
    # results = model.run_all_models()
    # summary_df = model.create_summary_table(results)
    # model.plot_results(summary_df)

    # # Example 2: Cross-topic analysis
    # print("\n\nExample 2: Cross-Topic Analysis")
    # print("=" * 50)

    # topics = ["agriculture", "technology", "health", "environment and climate"]
    # cross_topic_results = run_cross_topic_analysis(topics)

    # Run PPML
    db = DataBase()
    df_filtered, column_sets = db.prepare_data(
                time_frequency="monthly",
                start_date="2019-07",
                end_date="2024-11"
            )

    le_model = LobbyingEffectsModel(df_filtered, column_sets)

    res_tech = le_model.model_continuous_ddd_ppml_fixest_rscript(
        Rscript_path=r"D:\\R-4.5.1\\bin\\Rscript.exe",
        fe_pattern="member+time",
        # domain_filter="technology",
        domain_varying_slopes=True,
        include_controls=True,
        cluster_by="member_id + time_fe",
    )
    print(res_tech)