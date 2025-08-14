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


warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("seaborn-v0_8")


def model_continuous_ddd_linear(
    db: LongDatabase,
):
    """
    Continuous-treatment DDD using member×domain and time fixed effects.

        Specification:
            questions_{i,d,t} = beta * meetings_{i,d,t}
                                 + alpha_{i×d} (member×domain FE)
                                 + gamma_t (time FE)
                                 + X'_{i,d,t} * theta
                                 + u_{i,d,t}

        Notes:
        - Using member×domain FE absorbs both member and domain heterogeneity (and their interaction),
          which is stronger than separate additive member and domain FE, and avoids the 3-FE limit.
        - Controls from LongDatabase (if present) are included.

        Args:
            db: LongDatabase object (already prepared; see LongDatabase.prepare_long_panel)

        Returns:
            linearmodels PanelOLSResults or None on error
    """
    try:
        df_long = db.get_df().copy()
        time_col = db.get_time_col()

        # Ensure identifiers exist
        if "member_domain" not in df_long.columns:
            # Fallback construct if needed
            if "member_id" in df_long.columns and "domain" in df_long.columns:
                df_long["member_domain"] = (
                    df_long["member_id"].astype(str) + "__" + df_long["domain"].astype(str)
                )
            else:
                raise ValueError("Expected 'member_domain' in long panel.")

        # Coerce target vars to numeric and drop NAs
        df_long["questions"] = pd.to_numeric(df_long["questions"], errors="coerce")
        df_long["meetings"] = pd.to_numeric(df_long["meetings"], errors="coerce")
        df_long = df_long.dropna(subset=["questions", "meetings"])

        # Controls (if available from LongDatabase)
        control_cols = []
        if hasattr(db, "get_control_cols"):
            try:
                control_cols = db.get_control_cols() or []
            except Exception:
                control_cols = []
                
        control_cols = [c for c in control_cols if c in df_long.columns]

        # Build exogenous matrix: treatment + controls
        X = pd.DataFrame(index=df_long.index)
        X["meetings"] = df_long["meetings"].astype(float)
        for c in control_cols:
            X[c] = pd.to_numeric(df_long[c], errors="coerce").astype(float).fillna(0.0)

        # Set panel index to (member×domain, time)
        mi = pd.MultiIndex.from_arrays(
            [df_long["member_domain"].astype(str), df_long[time_col]],
            names=["member_domain", time_col],
        )
        y = df_long["questions"].astype(float)
        y.index = mi
        X.index = mi

        model = PanelOLS(
            y,
            X,
            entity_effects=True,   # member×domain fixed effects
            time_effects=True,     # time fixed effects
        )

        # Robust SEs (switch to clustered if desired)
        res = model.fit(cov_type="robust")
        return res
    except Exception as e:
        print(f"Error in DDD Linear model: {e}")
        import traceback
        traceback.print_exc()
        return None
