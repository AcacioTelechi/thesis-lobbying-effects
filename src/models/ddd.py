"""
Lobbying Effects Analysis Model Classes

This module contains two main classes:
1. DataBase: Responsible for loading and treating panel data
2. LobbyingEffectsModel: Responsible for running econometric models with different topics
"""

import sys

sys.path.append("D:\\repos\\pessoal\\thesis-lobbying-effects")
from src.database import LongDatabase
from src.models.utils.r_utils import run_r_script

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
    fe_pattern: str = "member_domain + domain_time",
    domain_filter: str | None = None,
    cluster_by: str = "member_id",
    timeout_seconds: int = 300,
    domain_varying_slopes: bool = False,
    include_meetings_squared: bool = False,
    alt_treatment_var: str | None = None,
    vary_by: str | None = None,
):
    """
    PPML using R fixest::fepois by calling Rscript (no rpy2 needed).

    Args:
        db: LongDatabase instance with prepared data
        Rscript_path: Optional full path to Rscript.exe (Windows), e.g., 'D:\\R-4.4.1\\bin\\x64\\Rscript.exe'.
        fe_pattern: Fixed effects pattern (e.g., "member_domain + domain_time")
        domain_filter: Optional filter to specific domain for estimation
        cluster_by: Column(s) to cluster SEs by (default: member_id, supports multi-way like "member_id + time_fe")
        domain_varying_slopes: Whether to allow treatment effects to vary by domain
        include_meetings_squared: Whether to include quadratic treatment term
        alt_treatment_var: Alternative treatment variable name (replaces 'meetings')
        vary_by: Variable to interact with treatment for varying slopes (replaces 'domain')

    Returns:
        dict with coefficient, p-value, n_obs; or None on error.
    """
    try:
        df_long = db.get_df()

        # Optional domain filter for per-domain estimation
        if domain_filter is not None:
            df_long = db.filter_by_domain(domain_filter)

        # Get required columns from database (already handles controls, FE vars, etc.)
        required_cols = db.get_required_cols()

        # Add cluster variables to required columns if not already present
        cluster_terms = [c.strip() for c in str(cluster_by).split("+")]
        for term in cluster_terms:
            if term in df_long.columns and term not in required_cols:
                required_cols.append(term)
        
        # Add treatment variable if alternative is specified
        if alt_treatment_var and alt_treatment_var in df_long.columns:
            if alt_treatment_var not in required_cols:
                required_cols.append(alt_treatment_var)
        
        # Add vary_by variable if specified
        if vary_by and vary_by in df_long.columns:
            if vary_by not in required_cols:
                required_cols.append(vary_by)

        # Filter to required columns only
        df_long = df_long[required_cols].copy()

        # Write temp CSV and R script
        with tempfile.TemporaryDirectory() as td:
            input_csv = os.path.join(td, "ppml_input.csv")
            print(f"Writing input CSV to {input_csv}")
            df_long.to_csv(input_csv, index=False)

            r_script_path = os.path.join(td, "ppml_fixest.R")
            output_json = os.path.join(td, "ppml_output.json")
            
            # Determine treatment variable
            treat_var = alt_treatment_var if alt_treatment_var else "meetings"

            # Build treatment term
            if domain_varying_slopes:
                interaction_var = vary_by if vary_by else "domain"
                treatment_term = f"{treat_var} + i({interaction_var}, {treat_var})"
            else:
                treatment_term = treat_var

            # Add quadratic term if requested
            if include_meetings_squared:
                treatment_term += f" + I({treat_var}^2)"

            # Add any lag/lead terms already in the data (handled by database)
            control_cols = db.get_control_cols()
            controls_rhs_str = ""
            if control_cols:
                available_controls = [c for c in control_cols if c in df_long.columns]
                if available_controls:
                    controls_rhs_str = " + " + " + ".join(available_controls)

            # Build full formula string
            formula_str = f"questions ~ {treatment_term}{controls_rhs_str} | {fe_pattern}"
            # Generate R script
            r_code = f"""
options(warn=1)
library(fixest)
library(jsonlite)

# Load data
df <- read.csv('{input_csv.replace('\\', '/')}', stringsAsFactors=FALSE)
df$questions <- as.numeric(df$questions)
df${treat_var} <- as.numeric(df${treat_var})

# Set reference levels if applicable
if ('domain' %in% names(df)) {{
    df$domain <- factor(df$domain)
    if ('agriculture' %in% levels(df$domain)) {{
        df$domain <- relevel(df$domain, ref = 'agriculture')
    }}
}}

# Fit model
fml <- as.formula('{formula_str}')
cl <- as.formula('~ {cluster_by}')
fit <- tryCatch(fepois(fml, data=df, cluster=cl), error=function(e) e)

if (inherits(fit, 'error')) {{
    write(toJSON(list(error=fit$message), auto_unbox=TRUE), file='{output_json.replace('\\', '/')}')
    quit(status=0)
}}

# Extract results
sm <- summary(fit)
ct <- sm$coeftable
rn <- rownames(ct)
out <- list()

if ({"TRUE" if domain_varying_slopes else "FALSE"}) {{
    # Domain-varying slopes
    base_idx <- which(rn == '{treat_var}')
    base_coef <- if (length(base_idx) == 0) NA else unname(ct[base_idx, 1])
    base_p <- if (length(base_idx) == 0) NA else unname(ct[base_idx, ncol(ct)])
    
    # Interaction terms
    interaction_pattern <- paste0(':', '{treat_var}', '$')
    d_idx <- grepl(interaction_pattern, rn)
    d_rows <- rn[d_idx]
    d_names <- sub('^.*::', '', d_rows)
    d_names <- sub(paste0(':', '{treat_var}', '$'), '', d_names)
    d_coefs <- if (any(d_idx)) as.numeric(ct[d_idx, 1]) else numeric(0)
    d_pvals <- if (any(d_idx)) as.numeric(ct[d_idx, ncol(ct)]) else numeric(0)
    
    out$base_coef <- base_coef
    out$base_p <- base_p
    out$delta_domains <- as.list(d_names)
    out$delta_coefs <- as.list(d_coefs)
    out$delta_pvals <- as.list(d_pvals)
}} else {{
    # Pooled slope
    idx <- which(rn == '{treat_var}')
    beta <- if (length(idx) == 0) NA else unname(ct[idx, 1])
    p <- if (length(idx) == 0) NA else unname(ct[idx, ncol(ct)])
    out$beta <- beta
    out$p_value <- p
}}

# Quadratic term if present
sq_idx <- which(rn == 'I({treat_var}^2)')
out$squared_coef <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, 1])
out$squared_p <- if (length(sq_idx) == 0) NA else unname(ct[sq_idx, ncol(ct)])

out$n_obs <- as.integer(nobs(fit))

write(toJSON(out, auto_unbox=TRUE), file='{output_json.replace('\\', '/')}')
quit(status=0)
"""

            # Run R script
            res = run_r_script(r_script_path, r_code, output_json, td, Rscript_path, timeout_seconds)

            if res is None:
                print("[WARNING] !!! R script failed !!!")
                return None

            # Process results
            n_obs = res.get("n_obs", None)
            squared_coef = res.get("squared_coef")
            squared_p = res.get("squared_p")

            if domain_varying_slopes:
                # Domain-varying slopes results
                base_coef = res.get("base_coef")
                base_p = res.get("base_p")
                delta_domains = res.get("delta_domains", []) or []
                delta_coefs = res.get("delta_coefs", []) or []
                delta_pvals = res.get("delta_pvals", []) or []
                
                print("\n=== Domain-varying slopes (fixest::fepois) ===")
                print(f"N observations: {n_obs}")
                print(f"Base coefficient: {base_coef} (p={base_p})")
                for name, dc, pv in zip(delta_domains, delta_coefs, delta_pvals):
                    print(f"  + {name}: {dc} (p={pv})")

                return {
                    "model": "Continuous PPML (fixest via Rscript)",
                    "domain_varying_slopes": True,
                    "base_coef": base_coef,
                    "base_p": base_p,
                    "delta_domains": delta_domains,
                    "delta_coefs": delta_coefs,
                    "delta_pvals": delta_pvals,
                    "n_obs": n_obs,
                    "fe_pattern": fe_pattern,
                    "cluster_by": cluster_by,
                    "squared_coef": squared_coef,
                    "squared_p": squared_p,
                }
            else:
                # Pooled slope results
                beta = res.get("beta", None)
                p_value = res.get("p_value", None)
                
                print("\n=== Continuous-Treatment PPML (Rscript fixest::fepois) ===")
                print(f"Coefficient on {treat_var}: {beta}")
                print(f"P-value (clustered by {cluster_by}): {p_value}")
                print(f"N observations: {n_obs}")

                return {
                    "model": "Continuous PPML (fixest via Rscript)",
                    "coefficient": beta,
                    "p_value": p_value,
                    "n_obs": n_obs,
                    "fe_pattern": fe_pattern,
                    "cluster_by": cluster_by,
                    "squared_coef": squared_coef,
                    "squared_p": squared_p,
                }

    except Exception as e:
        print(f"Error in Rscript-backed PPML: {e}")
        import traceback

        traceback.print_exc()
        return None
