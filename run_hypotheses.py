import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.database import DataBase
from src.lobbying_effects_model import LobbyingEffectsModel


R_DEFAULT = r"D:\\R-4.5.1\\bin\\Rscript.exe"
TW_CLUSTER = "member_id + time_fe"


DOMAINS = [
    "agriculture",
    "economics_and_trade",
    "education",
    "environment_and_climate",
    "foreign_and_security_affairs",
    "health",
    "human_rights",
    "infrastructure_and_industry",
    "technology",
]


def make_model(df: pd.DataFrame, column_sets: Dict[str, List[str]]) -> LobbyingEffectsModel:
    return LobbyingEffectsModel(df, column_sets)


def summarize_domain_slopes(res: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if res is None:
        return out
    if res.get("domain_varying_slopes"):
        for d, b in (res.get("slope_by_domain") or {}).items():
            out.append(
                {
                    "tag": tag,
                    "domain": d,
                    "beta": b,
                    "beta_pct": 100.0 * (np.exp(b) - 1.0) if b is not None else None,
                    "p_value": (res.get("p_by_domain") or {}).get(d),
                    "n_obs": res.get("n_obs"),
                }
            )
    else:
        out.append(
            {
                "tag": tag,
                "domain": "pooled",
                "beta": res.get("coefficient") or res.get("beta"),
                "beta_pct": 100.0 * (np.exp((res.get("coefficient") or res.get("beta") or 0.0)) - 1.0),
                "p_value": res.get("p_value"),
                "n_obs": res.get("n_obs"),
            }
        )
    return out


def run_h1(df: pd.DataFrame, column_sets: Dict[str, List[str]], rscript: str) -> List[Dict[str, Any]]:
    model = make_model(df, column_sets)
    res_base = model.model_continuous_ddd_ppml_fixest_rscript(
        Rscript_path=rscript,
        fe_pattern="member+time",
        domain_varying_slopes=True,
        include_controls=True,
        cluster_by=TW_CLUSTER,
    )
    if res_base is None:
        raise ValueError("res_base is None")
    rows = summarize_domain_slopes(res_base, tag="H1_baseline")

    res_lag = model.model_continuous_ddd_ppml_fixest_rscript(
        Rscript_path=rscript,
        fe_pattern="member+time",
        domain_varying_slopes=True,
        include_controls=True,
        cluster_by=TW_CLUSTER,
        include_lags=1,
    )
    if res_lag is None:
        raise ValueError("res_lag is None")
    rows += summarize_domain_slopes(res_lag, tag="H1_lag1")

    res_sq = model.model_continuous_ddd_ppml_fixest_rscript(
        Rscript_path=rscript,
        fe_pattern="member+time",
        domain_varying_slopes=True,
        include_controls=True,
        cluster_by=TW_CLUSTER,
        include_meetings_squared=True,
    )
    if res_sq is None:
        raise ValueError("res_sq is None")
    rows += summarize_domain_slopes(res_sq, tag="H1_quadratic")
    return rows


def split_sample(df: pd.DataFrame, mask: pd.Series) -> Dict[str, pd.DataFrame]:
    return {
        "high": df.loc[mask].copy(),
        "low": df.loc[~mask].copy(),
    }


def run_h2(df: pd.DataFrame, column_sets: Dict[str, List[str]], rscript: str) -> List[Dict[str, Any]]:
    # Resourcefulness proxy: top tercile of graph_authority or meetings_l_budget_cat_upper
    proxy_cols = [c for c in df.columns if c in ("graph_authority", "meetings_l_budget_cat_upper")]
    if not proxy_cols:
        return []
    proxy = df[proxy_cols[0]].astype(float)
    # thr = proxy.quantile(2.0 / 3.0)
    parts = split_sample(df, proxy >= 1)
    rows: List[Dict[str, Any]] = []
    for grp_name, part_df in parts.items():
        model = make_model(part_df, column_sets)
        res = model.model_continuous_ddd_ppml_fixest_rscript(
            Rscript_path=rscript,
            fe_pattern="member+time",
            domain_varying_slopes=True,
            include_controls=True,
            cluster_by=TW_CLUSTER,
        )
        if res is None:
            raise ValueError("res is None")
        rows += summarize_domain_slopes(res, tag=f"H2_resource_{grp_name}")
    return rows


def run_h3(df: pd.DataFrame, column_sets: Dict[str, List[str]], rscript: str) -> List[Dict[str, Any]]:
    # Salience proxy: high_salience_<domain> from DataBase
    rows: List[Dict[str, Any]] = []
    for d in DOMAINS:
        high_col = f"high_salience_{d}"
        if high_col not in df.columns:
            continue
        parts = split_sample(df, df[high_col] == 1)
        for grp_name, part_df in parts.items():
            model = make_model(part_df, column_sets)
            res = model.model_continuous_ddd_ppml_fixest_rscript(
                Rscript_path=rscript,
                fe_pattern="member+time",
                domain_varying_slopes=True,
                include_controls=True,
                cluster_by=TW_CLUSTER,
                domain_filter=d,
            )
            tag = f"H3_{d}_{'highSal' if grp_name=='high' else 'lowSal'}"
            if res is None:
                raise ValueError("res is None")
            rows += summarize_domain_slopes(res, tag=tag)
    return rows


def main() -> None:
    rscript = os.environ.get("RSCRIPT_PATH", R_DEFAULT)
    db = DataBase()
    df_filtered, column_sets = db.prepare_data(
        time_frequency="monthly", start_date="2019-07", end_date="2024-11"
    )

    all_rows: List[Dict[str, Any]] = []
    all_rows += run_h1(df_filtered, column_sets, rscript)
    all_rows += run_h2(df_filtered, column_sets, rscript)
    all_rows += run_h3(df_filtered, column_sets, rscript)

    out_df = pd.DataFrame(all_rows)
    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "hypotheses_results.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Quick JSON summary per tag-domain
    out_json = os.path.join("outputs", "hypotheses_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()


