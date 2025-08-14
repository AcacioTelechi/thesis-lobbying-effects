"""
Lobbying Effects Analysis Model Classes

This module contains two main classes:
1. DataBase: Responsible for loading and treating panel data
2. LobbyingEffectsModel: Responsible for running econometric models with different topics
"""

import sys

sys.path.append("D:\\repos\\pessoal\\thesis-lobbying-effects")
from src.database import LongDatabase

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
from typing import Type

import pyhdfe

warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")



def model_continuous_ddd_linear(db: LongDatabase):
    """
    y_{idt} = β T_{idt} + μ_{id} + μ_{it} + μ_{dt} + X_{idt}' θ + ε_{idt}
    """
    df_long = db.get_df().copy()

    # Identify relevant columns for FE and X
    main_cols = ["questions", "meetings", "member_id", "domain", db.get_time_col()]
    control_cols = db.get_control_cols()
    control_cols = [c for c in control_cols if c in df_long.columns]

    all_cols = main_cols + control_cols

    # Drop any rows with NA in any relevant column
    df_long = df_long.dropna(subset=all_cols).reset_index(drop=True)

    time_col = db.get_time_col()

    # Fixed effects
    df_long["fe_id"] = (
        df_long["member_id"].astype(str) + "_" + df_long["domain"].astype(str)
    ).astype("category")

    df_long["fe_it"] = (
        df_long["member_id"].astype(str) + "_" + df_long[time_col].astype(str)
    ).astype("category")

    df_long["fe_dt"] = (
        df_long["domain"].astype(str) + "_" + df_long[time_col].astype(str)
    ).astype("category")

    # Create FE DataFrame
    fe_df = df_long[["fe_id", "fe_it", "fe_dt"]]

    # Initialize pyhdfe absorber
    absorber = pyhdfe.create(ids=fe_df)

    # Partial out FEs
    y_resid = absorber.residualize(df_long[["questions"]].astype(float)).squeeze()
    X = df_long[["meetings"] + control_cols].astype(float)
    X_resid = absorber.residualize(X)

    # Add constant
    X_resid = sm.add_constant(X_resid)

    # OLS with cluster at member_id
    model = sm.OLS(y_resid, X_resid)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df_long["member_id"]})

    return res



def model_continuous_ddd_ppml_fixest_rscript(
    db: LongDatabase,
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
        df_long = db.get_df()

        # Optional domain filter for per-domain estimation
        if domain_filter is not None:
            df_long = db.filter_by_domain(domain_filter)

        # Keep only required columns to minimize IO
        required_cols = db.get_required_cols()

        # Ensure cluster columns exist (supports multi-way like 'member_id + time_fe')
        cluster_terms = [c.strip() for c in str(cluster_by).split("+")]
        missing_terms = []
        for term in cluster_terms:
            if term in df_long.columns:
                if term not in required_cols:
                    required_cols.append(term)
            else:
                missing_terms.append(term)
        if missing_terms and len(cluster_terms) == 1:
            print(
                f"Warning: cluster_by='{cluster_by}' not found. Falling back to 'member_id'."
            )
            cluster_by = "member_id"
            if "member_id" not in required_cols:
                required_cols.append("member_id")
        elif missing_terms:
            print(
                "Warning: some cluster columns not found and won't be added to CSV: "
                + ", ".join(missing_terms)
            )

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
        control_cols = db.get_control_cols()
        
        df_long = df_long[required_cols].copy()

        # Write temp CSV and R script
        with tempfile.TemporaryDirectory() as td:
            input_csv = os.path.join(td, "ppml_input.csv")
            print(f"Writing input CSV to {input_csv}")
            df_long.to_csv(input_csv, index=False)

            r_script_path = os.path.join(td, "ppml_fixest.R")
            output_json = os.path.join(td, "ppml_output.json")
            fe_rhs = (
                "member_domain + domain_time + member_time"
                if include_member_time_fe
                else "member_domain + domain_time"
            )
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
                fe_rhs = (
                    "member_domain + domain_time + member_time"
                    if include_member_time_fe
                    else "member_domain + domain_time"
                )

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
                rhs_controls = [c for c in control_cols if c in df_long.columns]
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
                treatment_term = treatment_term.replace(
                    f"i(domain, {treat_var})", f"i({vary_by}, {treat_var})"
                )
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
                + formula_str.replace('"', '\\"')
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
                    candidate = os.path.join(
                        os.path.dirname(Rscript_path), "Rscript.exe"
                    )
                    if os.path.exists(candidate):
                        chosen_rscript = candidate
                    else:
                        # Try x64 subfolder
                        candidate2 = os.path.join(
                            os.path.dirname(Rscript_path), "x64", "Rscript.exe"
                        )
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
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=td
            )
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
                    slope_by_domain[name] = (
                        base_coef if base_coef is not None else 0.0
                    ) + dc
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
