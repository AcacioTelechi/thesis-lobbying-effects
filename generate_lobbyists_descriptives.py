import os
from typing import List, Dict

import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("outputs", "lobbyists_descriptives")
DATA_PATH = os.path.join(".", "data", "silver", "df_lobbyists.csv")
FIGURES_DIR = os.path.join("Tese", "figures")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_lobbyists_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "l_registration_date" in df.columns:
        df["l_registration_date"] = pd.to_datetime(
            df["l_registration_date"], errors="coerce", dayfirst=True
        )
        df["registration_year"] = df["l_registration_date"].dt.year

    return df


def identify_theme_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "l_registration_date",
        "l_category",
        "l_head_office_country",
        "l_ln_max_budget",
    }

    theme_cols: List[str] = []
    for col in df.columns:
        if not col.startswith("l_"):
            continue
        if col in excluded:
            continue
        # Keep only binary-like numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            theme_cols.append(col)

    return theme_cols


def compute_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    series = df["l_category"].fillna("Desconhecido")
    counts = series.value_counts(dropna=False).rename("count")
    proportions = (counts / counts.sum()).rename("proportion")
    out = pd.concat([counts, proportions], axis=1).reset_index(names=["category"])
    return out


def compute_country_distribution(df: pd.DataFrame) -> pd.DataFrame:
    series = df["l_head_office_country"].fillna("Desconhecido")
    counts = series.value_counts(dropna=False).rename("count")
    proportions = (counts / counts.sum()).rename("proportion")
    out = pd.concat([counts, proportions], axis=1).reset_index(names=["country"])
    return out


def compute_year_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "registration_year" not in df.columns:
        return pd.DataFrame(columns=["year", "count", "proportion"])  # empty
    series = df["registration_year"].fillna(-1).astype(int)
    counts = series.value_counts(dropna=False).sort_index().rename("count")
    proportions = (counts / counts.sum()).rename("proportion")
    out = pd.concat([counts, proportions], axis=1).reset_index(names=["year"])
    out["year"] = out["year"].replace({-1: "Desconhecido"})
    return out


def compute_theme_coverage(df: pd.DataFrame, theme_cols: List[str]) -> pd.DataFrame:
    if not theme_cols:
        return pd.DataFrame(columns=["theme", "count", "proportion"])  # empty

    binary_df = df[theme_cols].copy()
    # Normalize to 0/1 in case of noise
    for col in theme_cols:
        binary_df[col] = (binary_df[col].fillna(0) > 0).astype(int)

    counts = binary_df.sum(axis=0).rename("count")
    proportions = (counts / len(df)).rename("proportion")
    out = pd.concat([counts, proportions], axis=1).reset_index(names=["theme"])
    return out


def compute_themes_per_lobbyist(df: pd.DataFrame, theme_cols: List[str]) -> pd.Series:
    if not theme_cols:
        return pd.Series(name="themes_per_lobbyist", dtype=int)
    binary_df = df[theme_cols].copy()
    for col in theme_cols:
        binary_df[col] = (binary_df[col].fillna(0) > 0).astype(int)
    return binary_df.sum(axis=1).rename("themes_per_lobbyist")


def compute_budget_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "l_ln_max_budget" not in df.columns:
        return pd.DataFrame(columns=["metric", "value"])  # empty

    series = pd.to_numeric(df["l_ln_max_budget"], errors="coerce").dropna()
    summary: Dict[str, float] = {
        "count": float(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if series.shape[0] > 1 else float("nan"),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }
    out = pd.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())})
    return out


def save_table(df: pd.DataFrame, name: str) -> None:
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    tex_path = os.path.join(OUTPUT_DIR, f"{name}.tex")
    df.to_csv(csv_path, index=False)
    # Render a compact LaTeX table
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:,.3f}", escape=True))


def plot_and_save_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str,
                      horizontal: bool = False, annotate_pct: bool = False) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    data = df.copy()
    if horizontal:
        ax.barh(data[x_col].astype(str), data[y_col])
        ax.set_xlabel(y_col)
        ax.set_ylabel("")
    else:
        ax.bar(data[x_col].astype(str), data[y_col])
        ax.set_ylabel(y_col)
        ax.set_xlabel("")

    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")

    if annotate_pct and "proportion" in df.columns:
        if horizontal:
            for i, (label, val, prop) in enumerate(zip(data[x_col], data[y_col], data["proportion"])):
                ax.text(val, i, f" {prop*100:.1f}%", va="center")
        else:
            for i, (label, val, prop) in enumerate(zip(data[x_col], data[y_col], data["proportion"])):
                ax.text(i, val, f"{prop*100:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_and_save_hist(series: pd.Series, title: str, filename: str, bins: int = 20) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty:
        plt.close(fig)
        return

    ax.hist(clean.astype(float), bins=bins, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    ensure_output_dir(FIGURES_DIR)

    df = load_lobbyists_dataframe(DATA_PATH)
    n = df.shape[0]

    # Identify themes
    theme_cols = identify_theme_columns(df)

    # Compute tables
    cat_dist = compute_category_distribution(df)
    country_dist = compute_country_distribution(df)
    year_dist = compute_year_distribution(df)
    theme_cov = compute_theme_coverage(df, theme_cols)
    themes_per = compute_themes_per_lobbyist(df, theme_cols)
    budget_summ = compute_budget_summary(df)

    # Save tables
    save_table(cat_dist, "category_distribution")
    save_table(country_dist, "country_distribution")
    save_table(year_dist, "year_distribution")
    save_table(theme_cov, "theme_coverage")
    save_table(budget_summ, "budget_summary")

    # Plots
    plot_and_save_bar(
        cat_dist,
        x_col="category",
        y_col="count",
        title=f"Distribuição por categoria (n = {n})",
        filename="category_distribution.png",
        horizontal=False,
        annotate_pct=True,
    )

    # If there are many countries, show top 20
    country_plot = country_dist.copy().sort_values("count", ascending=False).head(20)
    plot_and_save_bar(
        country_plot,
        x_col="country",
        y_col="count",
        title=f"País-sede (top 20; n = {n})",
        filename="country_distribution_top20.png",
        horizontal=False,
        annotate_pct=False,
    )

    plot_and_save_bar(
        year_dist,
        x_col="year",
        y_col="count",
        title="Ano de registo",
        filename="year_distribution.png",
        horizontal=False,
        annotate_pct=False,
    )

    # Theme coverage plot (sorted by proportion)
    theme_plot = theme_cov.copy().sort_values("proportion", ascending=False)
    plot_and_save_bar(
        theme_plot,
        x_col="theme",
        y_col="proportion",
        title="Cobertura temática (proporção de entidades)",
        filename="theme_coverage.png",
        horizontal=True,
        annotate_pct=True,
    )

    # Histograms
    if "l_ln_max_budget" in df.columns:
        plot_and_save_hist(
            df["l_ln_max_budget"],
            title="Distribuição de l_ln_max_budget",
            filename="budget_ln_hist.png",
            bins=20,
        )

    if not themes_per.empty:
        plot_and_save_hist(
            themes_per,
            title="Número de temas por lobista",
            filename="themes_per_lobbyist_hist.png",
            bins=min(20, max(5, themes_per.nunique())),
        )

    # Also save a simple CSV with per-entity derived fields used above
    enriched = df.copy()
    if "registration_year" in enriched.columns:
        pass  # already added
    if theme_cols:
        enriched["themes_per_lobbyist"] = themes_per
    enriched_out = os.path.join(OUTPUT_DIR, "df_lobbyists_enriched.csv")
    enriched.to_csv(enriched_out, index=False)

    print(f"Relatórios gerados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


