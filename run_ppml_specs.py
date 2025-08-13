import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd

# Assume this script is run from the project root
from src.database import DataBase
from src.lobbying_effects_model import LobbyingEffectsModel


def as_percent(beta: Optional[float]) -> Optional[float]:
    if beta is None:
        return None
    try:
        return 100.0 * (pow(2.718281828459045, float(beta)) - 1.0)
    except Exception:
        return None


def run_spec(
    model: LobbyingEffectsModel,
    name: str,
    Rscript_path: Optional[str],
    fe_pattern: str,
    cluster_by: str,
    domain_varying_slopes: bool,
    include_controls: bool,
    include_meetings_squared: bool = False,
    include_lead1_meetings: bool = False,
    include_lag1_meetings: bool = False,
    trim_top_fraction: Optional[float] = None,
    domain_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    res = model.model_continuous_ddd_ppml_fixest_rscript(
        Rscript_path=Rscript_path,
        fe_pattern=fe_pattern,
        cluster_by=cluster_by,
        domain_varying_slopes=domain_varying_slopes,
        include_controls=include_controls,
        include_meetings_squared=include_meetings_squared,
        include_lead1_meetings=include_lead1_meetings,
        include_lag1_meetings=include_lag1_meetings,
        trim_top_fraction=trim_top_fraction,
        domain_filter=domain_filter,
    )

    rows: List[Dict[str, Any]] = []
    if res is None:
        rows.append(
            {
                "spec": name,
                "fe_pattern": fe_pattern,
                "cluster_by": cluster_by,
                "domain_varying_slopes": domain_varying_slopes,
                "meetings_squared": include_meetings_squared,
                "lead1": include_lead1_meetings,
                "lag1": include_lag1_meetings,
                "trim_top_fraction": trim_top_fraction,
                "domain": domain_filter or (res.get("base_domain") if isinstance(res, dict) else None),
                "beta": None,
                "beta_pct": None,
                "p_value": None,
                "n_obs": None,
                "squared_coef": None,
                "squared_p": None,
                "lead1_coef": None,
                "lead1_p": None,
                "lag1_coef": None,
                "lag1_p": None,
                "notes": "failed",
            }
        )
        return rows

    if domain_varying_slopes and isinstance(res, dict) and "slope_by_domain" in res:
        slope_by_domain = res.get("slope_by_domain", {}) or {}
        p_by_domain = res.get("p_by_domain", {}) or {}
        # Add global dynamics info (applies to all rows in this spec)
        lag_names = res.get("lag_names") or []
        lag_coefs = res.get("lag_coefs") or []
        lead_names = res.get("lead_names") or []
        lead_coefs = res.get("lead_coefs") or []
        # Compute simple sums
        lag_sum = float(sum([c for c in lag_coefs if c is not None])) if lag_coefs else None
        lead_sum = float(sum([c for c in lead_coefs if c is not None])) if lead_coefs else None
        for d, b in slope_by_domain.items():
            rows.append(
                {
                    "spec": name,
                    "fe_pattern": fe_pattern,
                    "cluster_by": cluster_by,
                    "domain_varying_slopes": domain_varying_slopes,
                    "meetings_squared": include_meetings_squared,
                    "lead1": include_lead1_meetings,
                    "lag1": include_lag1_meetings,
                    "trim_top_fraction": trim_top_fraction,
                    "domain": d,
                    "beta": b,
                    "beta_pct": as_percent(b),
                    "p_value": p_by_domain.get(d),
                    "n_obs": res.get("n_obs"),
                    "squared_coef": res.get("squared_coef"),
                    "squared_p": res.get("squared_p"),
                    "lead1_coef": res.get("lead1_coef"),
                    "lead1_p": res.get("lead1_p"),
                    "lag1_coef": res.get("lag1_coef"),
                    "lag1_p": res.get("lag1_p"),
                    "lag_names": ";".join([str(x) for x in lag_names]) if lag_names else None,
                    "lag_coefs": ";".join([str(x) for x in lag_coefs]) if lag_coefs else None,
                    "lag_sum": lag_sum,
                    "lead_names": ";".join([str(x) for x in lead_names]) if lead_names else None,
                    "lead_coefs": ";".join([str(x) for x in lead_coefs]) if lead_coefs else None,
                    "lead_sum": lead_sum,
                    "notes": "domain-specific",
                }
            )
    else:
        # pooled or per-domain filtered
        b = res.get("coefficient") or res.get("beta")
        p = res.get("p_value")
        rows.append(
            {
                "spec": name,
                "fe_pattern": fe_pattern,
                "cluster_by": cluster_by,
                "domain_varying_slopes": domain_varying_slopes,
                "meetings_squared": include_meetings_squared,
                "lead1": include_lead1_meetings,
                "lag1": include_lag1_meetings,
                "trim_top_fraction": trim_top_fraction,
                "domain": domain_filter or "pooled",
                "beta": b,
                "beta_pct": as_percent(b) if b is not None else None,
                "p_value": p,
                "n_obs": res.get("n_obs"),
                "squared_coef": res.get("squared_coef"),
                "squared_p": res.get("squared_p"),
                "lead1_coef": res.get("lead1_coef"),
                "lead1_p": res.get("lead1_p"),
                "lag1_coef": res.get("lag1_coef"),
                "lag1_p": res.get("lag1_p"),
                "lag_names": ";".join([str(x) for x in (res.get("lag_names") or [])]) if res.get("lag_names") else None,
                "lag_coefs": ";".join([str(x) for x in (res.get("lag_coefs") or [])]) if res.get("lag_coefs") else None,
                "lag_sum": float(sum([c for c in (res.get("lag_coefs") or []) if c is not None])) if res.get("lag_coefs") else None,
                "lead_names": ";".join([str(x) for x in (res.get("lead_names") or [])]) if res.get("lead_names") else None,
                "lead_coefs": ";".join([str(x) for x in (res.get("lead_coefs") or [])]) if res.get("lead_coefs") else None,
                "lead_sum": float(sum([c for c in (res.get("lead_coefs") or []) if c is not None])) if res.get("lead_coefs") else None,
                "notes": "pooled" if domain_filter is None else f"filtered:{domain_filter}",
            }
        )

    return rows


def main():
    # Config
    rscript_default = r"D:\\R-4.5.1\\bin\\Rscript.exe"
    rscript_path = os.environ.get("RSCRIPT_PATH", rscript_default)

    db = DataBase()
    df_filtered, column_sets = db.prepare_data(
        time_frequency="monthly",
        start_date="2019-07",
        end_date="2024-11",
    )

    model = LobbyingEffectsModel(df_filtered, column_sets)

    specs: List[Dict[str, Any]] = [
        {
            "name": "baseline_member+time_cluster_member",
            "fe_pattern": "member+time",
            "cluster_by": "member_id",
            "dvs": True,
            "controls": True,
        },
        {
            "name": "two_way_cluster",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": True,
            "controls": True,
        },
        {
            "name": "ddd_fe_memberDomain+domainTime",
            "fe_pattern": "member_domain+domain_time",
            "cluster_by": "member_id + time_fe",
            "dvs": True,
            "controls": True,
        },
        {
            "name": "nonlinear_squared",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": True,
            "controls": True,
            "sq": True,
        },
        {
            "name": "dynamics_lead1_lag1",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": True,
            "controls": True,
            "lead1": True,
            "lag1": True,
        },
        {
            "name": "trim_top_1pct",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": True,
            "controls": True,
            "trim": 0.01,
        },
        {
            "name": "per_domain_agriculture",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": False,  # no interactions when filtering
            "controls": True,
            "domain_filter": "agriculture",
        },
        {
            "name": "per_domain_technology",
            "fe_pattern": "member+time",
            "cluster_by": "member_id + time_fe",
            "dvs": False,
            "controls": True,
            "domain_filter": "technology",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for s in specs:
        rows.extend(
            run_spec(
                model=model,
                name=s["name"],
                Rscript_path=rscript_path,
                fe_pattern=s["fe_pattern"],
                cluster_by=s["cluster_by"],
                domain_varying_slopes=bool(s.get("dvs", True)),
                include_controls=bool(s.get("controls", True)),
                include_meetings_squared=bool(s.get("sq", False)),
                include_lead1_meetings=bool(s.get("lead1", False)),
                include_lag1_meetings=bool(s.get("lag1", False)),
                trim_top_fraction=s.get("trim"),
                domain_filter=s.get("domain_filter"),
            )
        )

    df = pd.DataFrame(rows)
    df.sort_values(["spec", "domain"], inplace=True)
    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "ppml_specs_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved comparison to {out_csv}")

    # Quick on-screen summary for agriculture and technology if present
    for d in ["agriculture", "technology"]:
        sub = df[df["domain"] == d][["spec", "beta", "beta_pct", "p_value", "n_obs"]]
        if not sub.empty:
            print(f"\n=== {d} ===")
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()


