#!/usr/bin/env python3
"""
Script para gerar gráficos descritivos de alta qualidade para a tese
Análise dos efeitos do lobbying na atividade parlamentar dos MEPs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Configurações para gráficos de alta qualidade
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Configurações para LaTeX/publicação
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "font.family": "serif",
        "text.usetex": True,  # Pode ser True se tiver LaTeX instalado
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def load_data():
    """Carrega os dados do painel"""
    print("Carregando dados...")
    df = pd.read_csv("df_long.csv")

    # Criar variáveis auxiliares
    df["treated"] = (df["meetings"] > 0).astype(int)
    df["has_questions"] = (df["questions"] > 0).astype(int)
    df["has_meetings"] = (df["meetings"] > 0).astype(int)
    df["date"] = pd.to_datetime(df["Y-m"])

    print(f"Dados carregados: {len(df):,} observações")
    return df


def create_output_dir():
    """Cria diretório para os gráficos"""
    output_dir = Path("Tese/figures")
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def plot_zero_inflation_analysis(df, output_dir):
    """Gráfico 1: Análise da inflação de zeros"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1.1 Proporções de zeros vs não-zeros
    categories = [
        "Perguntas\n(Zero)",
        "Perguntas\n(>0)",
        "Reuniões\n(Zero)",
        "Reuniões\n(>0)",
    ]
    proportions = [
        (df["questions"] == 0).mean() * 100,
        (df["questions"] > 0).mean() * 100,
        (df["meetings"] == 0).mean() * 100,
        (df["meetings"] > 0).mean() * 100,
    ]
    colors = ["lightsteelblue", "steelblue", "lightcoral", "firebrick"]

    bars = ax1.bar(
        categories, proportions, color=colors, edgecolor="black", linewidth=0.5
    )
    ax1.set_title("Proporção de Zeros vs Valores Positivos")
    ax1.set_ylabel("Percentual (%)")

    # Adicionar valores nas barras
    for i, (bar, v) in enumerate(zip(bars, proportions)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 1.2 Distribuição de valores positivos - Perguntas
    positive_questions = df[df["questions"] > 0]["questions"]
    ax2.hist(
        positive_questions,
        bins=range(1, int(positive_questions.max()) + 2),
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Distribuição de Perguntas (quando > 0)")
    ax2.set_xlabel("Número de Perguntas")
    ax2.set_ylabel("Frequência")
    ax2.grid(True, alpha=0.3)

    # 1.3 Distribuição de valores positivos - Reuniões
    positive_meetings = df[df["meetings"] > 0]["meetings"]
    # Usar bins menores para melhor visualização
    bins = np.logspace(0, np.log10(positive_meetings.max()), 20)
    ax3.hist(
        positive_meetings,
        bins=bins,
        alpha=0.7,
        color="firebrick",
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_title("Distribuição de Reuniões (quando > 0)")
    ax3.set_xlabel("Número de Reuniões")
    ax3.set_ylabel("Frequência")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)

    # 1.4 Distribuições log-transformadas
    ax4.hist(
        np.log1p(positive_questions),
        bins=20,
        alpha=0.6,
        label="Log(Perguntas+1)",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.hist(
        np.log1p(positive_meetings),
        bins=20,
        alpha=0.6,
        label="Log(Reuniões+1)",
        color="firebrick",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_title("Distribuições Log-Transformadas (valores > 0)")
    ax4.set_xlabel("Log(Valor + 1)")
    ax4.set_ylabel("Frequência")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_zero_inflation_analysis.pdf")
    plt.savefig(output_dir / "fig1_zero_inflation_analysis.png")
    print("✓ Gráfico 1 salvo: Análise da inflação de zeros")
    plt.close()


def plot_time_series_analysis(df, output_dir):
    """Gráfico 2: Análise temporal"""
    # Agregar dados por tempo
    df_time = (
        df.groupby("Y-m")
        .agg(
            {
                "questions": ["sum", "mean"],
                "meetings": ["sum", "mean"],
                "treated": "mean",
            }
        )
        .round(3)
    )

    df_time.columns = ["_".join(col).strip() for col in df_time.columns.values]
    df_time = df_time.reset_index()
    df_time["date"] = pd.to_datetime(df_time["Y-m"])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 2.1 Tendências agregadas
    ax1.plot(
        df_time["date"],
        df_time["questions_sum"],
        "b-",
        linewidth=2.5,
        label="Perguntas Totais",
        marker="o",
        markersize=3,
    )
    ax1.plot(
        df_time["date"],
        df_time["meetings_sum"],
        "r-",
        linewidth=2.5,
        label="Reuniões Totais",
        marker="s",
        markersize=3,
    )

    ax1.set_title("Evolução Temporal: Totais Mensais")
    ax1.set_ylabel("Total de Perguntas / Reuniões")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2.2 Médias por MEP-domínio
    ax2.plot(
        df_time["date"],
        df_time["questions_mean"],
        "b-",
        linewidth=2.5,
        label="Média Perguntas",
        marker="o",
        markersize=3,
    )
    ax2.plot(
        df_time["date"],
        df_time["meetings_mean"],
        "r-",
        linewidth=2.5,
        label="Média Reuniões",
        marker="s",
        markersize=3,
    )

    ax2.set_title("Evolução Temporal: Médias por MEP-Domínio")
    ax2.set_ylabel("Média de Perguntas / Reuniões")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2.3 Taxa de tratamento ao longo do tempo
    ax3.plot(
        df_time["date"],
        df_time["treated_mean"] * 100,
        "g-",
        linewidth=2.5,
        marker="d",
        markersize=4,
    )
    ax3.set_title("Taxa de Tratamento ao Longo do Tempo")
    ax3.set_ylabel(r"\% de MEP-Domínios com Reuniões")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # 2.4 Correlação temporal
    correlations = []
    for period in df_time["Y-m"]:
        period_data = df[df["Y-m"] == period]
        if len(period_data) > 1:
            corr = period_data["questions"].corr(period_data["meetings"])
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)

    ax4.plot(
        df_time["date"], correlations, "purple", linewidth=2.5, marker="o", markersize=4
    )
    ax4.set_title("Correlação Perguntas-Reuniões ao Longo do Tempo")
    ax4.set_ylabel("Coeficiente de Correlação")
    ax4.tick_params(axis="x", rotation=45)
    ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_time_series_analysis.pdf")
    plt.savefig(output_dir / "fig2_time_series_analysis.png")
    print("✓ Gráfico 2 salvo: Análise temporal")
    plt.close()


def plot_domain_heterogeneity(df, output_dir):
    """Gráfico 3: Heterogeneidade por domínio"""
    # Estatísticas por domínio
    domain_stats = (
        df.groupby("domain")
        .agg(
            {
                "questions": ["mean", "std"],
                "meetings": ["mean", "std"],
                "treated": "mean",
            }
        )
        .round(4)
    )

    domain_stats.columns = [
        "_".join(col).strip() for col in domain_stats.columns.values
    ]
    domain_stats = domain_stats.reset_index()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 3.1 Médias por domínio
    x_pos = range(len(domain_stats))

    bars1 = ax1.bar(
        [x - 0.2 for x in x_pos],
        domain_stats["questions_mean"],
        0.4,
        label="Perguntas",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        [x + 0.2 for x in x_pos],
        domain_stats["meetings_mean"],
        0.4,
        label="Reuniões",
        color="firebrick",
        alpha=0.8,
    )

    ax1.set_title("Atividade Média por Domínio de Política")
    ax1.set_ylabel("Média Mensal")
    ax1.set_xlabel("Domínio")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [
            str(d)[:12] + "..." if len(str(d)) > 12 else str(d)
            for d in domain_stats["domain"]
        ],
        rotation=45,
        ha="right",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 3.2 Taxa de tratamento por domínio
    bars = ax2.bar(
        x_pos,
        domain_stats["treated_mean"] * 100,
        color="forestgreen",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Taxa de Tratamento por Domínio")
    ax2.set_ylabel("% com Reuniões de Lobbying")
    ax2.set_xlabel("Domínio")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [
            str(d)[:12] + "..." if len(str(d)) > 12 else str(d)
            for d in domain_stats["domain"]
        ],
        rotation=45,
        ha="right",
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3.3 Box plots - Perguntas por domínio
    domains = df["domain"].unique()
    domain_questions = [
        df[df["domain"] == domain]["questions"].values for domain in domains
    ]

    bp1 = ax3.boxplot(
        domain_questions,
        labels=[str(d)[:8] + "..." if len(str(d)) > 8 else str(d) for d in domains],
        patch_artist=True,
    )

    # Colorir os box plots
    colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_title("Distribuição de Perguntas por Domínio")
    ax3.set_ylabel("Número de Perguntas")
    ax3.set_xlabel("Domínio")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # 3.4 Box plots - Reuniões por domínio (apenas valores > 0)
    domain_meetings = [
        df[(df["domain"] == domain) & (df["meetings"] > 0)]["meetings"].values
        for domain in domains
    ]

    # Filtrar domínios que têm dados
    valid_meetings = [
        (meetings, domain)
        for meetings, domain in zip(domain_meetings, domains)
        if len(meetings) > 0
    ]

    if valid_meetings:
        meetings_data, meeting_domains = zip(*valid_meetings)
        bp2 = ax4.boxplot(
            meetings_data,
            labels=[
                str(d)[:8] + "..." if len(str(d)) > 8 else str(d)
                for d in meeting_domains
            ],
            patch_artist=True,
        )

        # Colorir os box plots
        colors = plt.cm.Set3(np.linspace(0, 1, len(meeting_domains)))
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax4.set_title("Distribuição de Reuniões por Domínio (quando > 0)")
    ax4.set_ylabel("Número de Reuniões")
    ax4.set_xlabel("Domínio")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_domain_heterogeneity.pdf")
    plt.savefig(output_dir / "fig3_domain_heterogeneity.png")
    print("✓ Gráfico 3 salvo: Heterogeneidade por domínio")
    plt.close()


def plot_correlation_analysis(df, output_dir):
    """Gráfico 4: Análise de correlações e scatter plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 4.1 Scatter plot geral (amostra para visualização)
    sample_df = df.sample(min(10000, len(df)), random_state=42)

    ax1.scatter(
        sample_df["meetings"], sample_df["questions"], alpha=0.3, s=8, color="darkblue"
    )

    # Linha de tendência
    if len(sample_df) > 1:
        z = np.polyfit(sample_df["meetings"], sample_df["questions"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, sample_df["meetings"].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    ax1.set_xlabel("Número de Reuniões")
    ax1.set_ylabel("Número de Perguntas")
    ax1.set_title("Relação entre Perguntas e Reuniões (Amostra)")
    ax1.grid(True, alpha=0.3)

    # 4.2 Scatter plot log-scale
    sample_log_meetings = np.log1p(sample_df["meetings"])
    sample_log_questions = np.log1p(sample_df["questions"])

    ax2.scatter(
        sample_log_meetings, sample_log_questions, alpha=0.3, s=8, color="darkgreen"
    )

    # Linha de tendência para log
    if len(sample_df) > 1:
        z_log = np.polyfit(sample_log_meetings, sample_log_questions, 1)
        p_log = np.poly1d(z_log)
        x_log_trend = np.linspace(0, sample_log_meetings.max(), 100)
        ax2.plot(x_log_trend, p_log(x_log_trend), "r--", alpha=0.8, linewidth=2)

    ax2.set_xlabel("Log(Reuniões + 1)")
    ax2.set_ylabel("Log(Perguntas + 1)")
    ax2.set_title("Relação Log-Transformada")
    ax2.grid(True, alpha=0.3)

    # 4.3 Correlação por domínio
    correlations_by_domain = (
        df.groupby("domain")
        .apply(lambda x: x["questions"].corr(x["meetings"]) if len(x) > 1 else 0)
        .reset_index()
    )
    correlations_by_domain.columns = ["domain", "correlation"]
    correlations_by_domain = correlations_by_domain.fillna(0)

    bars = ax3.bar(
        range(len(correlations_by_domain)),
        correlations_by_domain["correlation"],
        color="purple",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_xlabel("Domínio")
    ax3.set_ylabel("Correlação")
    ax3.set_title("Correlação Perguntas-Reuniões por Domínio")
    ax3.set_xticks(range(len(correlations_by_domain)))
    ax3.set_xticklabels(
        [
            str(d)[:10] + "..." if len(str(d)) > 10 else str(d)
            for d in correlations_by_domain["domain"]
        ],
        rotation=45,
        ha="right",
    )
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # Adicionar valores nas barras
    for i, (bar, corr) in enumerate(zip(bars, correlations_by_domain["correlation"])):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.01 if height >= 0 else -0.02),
            f"{corr:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    # 4.4 Matriz de correlação (variáveis principais)
    # Selecionar algumas variáveis para correlação
    corr_vars = ["questions", "meetings"]

    # Adicionar algumas variáveis de controle se existirem
    potential_vars = ["treated", "has_questions", "has_meetings"]
    for var in potential_vars:
        if var in df.columns:
            corr_vars.append(var)

    corr_matrix = df[corr_vars].corr()

    # Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    im = ax4.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Adicionar texto
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            if not mask[i, j]:
                text = ax4.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

    ax4.set_xticks(range(len(corr_vars)))
    ax4.set_yticks(range(len(corr_vars)))
    ax4.set_xticklabels(corr_vars, rotation=45, ha="right")
    ax4.set_yticklabels(corr_vars)
    ax4.set_title("Matriz de Correlação")

    # Colorbar
    plt.colorbar(im, ax=ax4, shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_correlation_analysis.pdf")
    plt.savefig(output_dir / "fig4_correlation_analysis.png")
    print("✓ Gráfico 4 salvo: Análise de correlações")
    plt.close()


def plot_extensive_intensive_margins(df, output_dir):
    """Gráfico 5: Análise das margens extensiva e intensiva"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 5.1 Tabulação cruzada visualizada
    crosstab = (
        pd.crosstab(df["has_questions"], df["has_meetings"], normalize="all") * 100
    )

    im = ax1.imshow(crosstab.values, cmap="Blues", aspect="auto")

    # Adicionar percentuais
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            text = ax1.text(
                j,
                i,
                f"{crosstab.iloc[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if crosstab.iloc[i, j] > 50 else "black",
                fontweight="bold",
                fontsize=12,
            )

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Sem Reuniões", "Com Reuniões"])
    ax1.set_yticklabels(["Sem Perguntas", "Com Perguntas"])
    ax1.set_title("Tabulação Cruzada: Perguntas vs Reuniões (%)")

    # Colorbar
    plt.colorbar(im, ax=ax1, shrink=0.6)

    # 5.2 Margens extensiva vs intensiva
    extensive_questions = (df["questions"] > 0).mean() * 100
    extensive_meetings = (df["meetings"] > 0).mean() * 100
    intensive_questions = df[df["questions"] > 0]["questions"].mean()
    intensive_meetings = df[df["meetings"] > 0]["meetings"].mean()

    categories = ["Perguntas", "Reuniões"]
    extensive_values = [extensive_questions, extensive_meetings]
    intensive_values = [intensive_questions, intensive_meetings]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        extensive_values,
        width,
        label="Extensiva (% com atividade)",
        color="steelblue",
        alpha=0.8,
    )

    # Segundo eixo y para margem intensiva
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(
        x + width / 2,
        intensive_values,
        width,
        label="Intensiva (média se > 0)",
        color="firebrick",
        alpha=0.8,
    )

    ax2.set_xlabel("Variável")
    ax2.set_ylabel("Margem Extensiva (%)", color="steelblue")
    ax2_twin.set_ylabel("Margem Intensiva (média)", color="firebrick")
    ax2.set_title("Decomposição: Margens Extensiva vs Intensiva")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)

    # Adicionar valores nas barras
    for bar, val in zip(bars1, extensive_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for bar, val in zip(bars2, intensive_values):
        height = bar.get_height()
        ax2_twin.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5.3 Distribuição da margem intensiva - Perguntas
    positive_questions = df[df["questions"] > 0]["questions"]
    ax3.hist(
        positive_questions,
        bins=range(1, min(17, int(positive_questions.max()) + 2)),
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_title("Margem Intensiva: Distribuição de Perguntas (quando > 0)")
    ax3.set_xlabel("Número de Perguntas")
    ax3.set_ylabel("Frequência")
    ax3.grid(True, alpha=0.3)

    # Adicionar estatísticas
    mean_q = positive_questions.mean()
    median_q = positive_questions.median()
    ax3.axvline(
        mean_q, color="red", linestyle="--", linewidth=2, label=f"Média: {mean_q:.2f}"
    )
    ax3.axvline(
        median_q,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {median_q:.2f}",
    )
    ax3.legend()

    # 5.4 Distribuição da margem intensiva - Reuniões (escala log)
    positive_meetings = df[df["meetings"] > 0]["meetings"]

    # Usar bins logarítmicos para melhor visualização
    max_meetings = min(50, positive_meetings.max())  # Limitar para melhor visualização
    bins = np.logspace(0, np.log10(max_meetings), 20)

    ax4.hist(
        positive_meetings[positive_meetings <= max_meetings],
        bins=bins,
        alpha=0.7,
        color="firebrick",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_title("Margem Intensiva: Distribuição de Reuniões (quando > 0)")
    ax4.set_xlabel("Número de Reuniões (escala log)")
    ax4.set_ylabel("Frequência")
    ax4.set_xscale("log")
    ax4.grid(True, alpha=0.3)

    # Adicionar estatísticas
    mean_m = positive_meetings.mean()
    median_m = positive_meetings.median()
    ax4.axvline(
        mean_m, color="blue", linestyle="--", linewidth=2, label=f"Média: {mean_m:.2f}"
    )
    ax4.axvline(
        median_m,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {median_m:.2f}",
    )
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_extensive_intensive_margins.pdf")
    plt.savefig(output_dir / "fig5_extensive_intensive_margins.png")
    print("✓ Gráfico 5 salvo: Análise das margens extensiva e intensiva")
    plt.close()


def plot_mep_aggregated_analysis(df, output_dir):
    """Gráfico 6: Análise agregada por MEP"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 6.1 Análise por MEP - Total de MEPs que receberam tratamento
    mep_treatment = (
        df.groupby("member_id")
        .agg(
            {
                "meetings": "sum",
                "questions": "sum",
                "treated": "max",  # Se teve pelo menos uma reunião
            }
        )
        .reset_index()
    )

    # Estatísticas de tratamento por MEP
    total_meps = len(mep_treatment)
    treated_meps = (mep_treatment["treated"] == 1).sum()
    never_treated_meps = total_meps - treated_meps

    # Gráfico de pizza para tratamento por MEP
    labels = [
        f"MEPs Tratados\n({treated_meps:,})",
        f"MEPs Nunca Tratados\n({never_treated_meps:,})",
    ]
    sizes = [treated_meps, never_treated_meps]
    colors = ["lightcoral", "lightsteelblue"]
    explode = (0.05, 0)  # explode the treated slice

    wedges, texts, autotexts = ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.set_title("Distribuição de MEPs por Status de Tratamento\n(Período Completo)")

    # Melhorar aparência do texto
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # 6.2 Distribuição de reuniões totais por MEP (apenas tratados)
    treated_meps_data = mep_treatment[mep_treatment["treated"] == 1]

    ax2.hist(
        treated_meps_data["meetings"],
        bins=30,
        alpha=0.7,
        color="firebrick",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Distribuição de Reuniões Totais\npor MEP (apenas MEPs tratados)")
    ax2.set_xlabel("Número Total de Reuniões")
    ax2.set_ylabel("Número de MEPs")
    ax2.grid(True, alpha=0.3)

    # Adicionar estatísticas
    mean_meetings = treated_meps_data["meetings"].mean()
    median_meetings = treated_meps_data["meetings"].median()
    ax2.axvline(
        mean_meetings,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Média: {mean_meetings:.1f}",
    )
    ax2.axvline(
        median_meetings,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {median_meetings:.1f}",
    )
    ax2.legend()

    # 6.3 Scatter plot: Reuniões vs Perguntas por MEP
    ax3.scatter(
        treated_meps_data["meetings"],
        treated_meps_data["questions"],
        alpha=0.6,
        s=30,
        color="darkgreen",
    )

    # Linha de tendência
    if len(treated_meps_data) > 1:
        z = np.polyfit(treated_meps_data["meetings"], treated_meps_data["questions"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(
            treated_meps_data["meetings"].min(),
            treated_meps_data["meetings"].max(),
            100,
        )
        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        # Calcular correlação
        corr = treated_meps_data["meetings"].corr(treated_meps_data["questions"])
        ax3.text(
            0.05,
            0.95,
            f"Correlação: {corr:.3f}",
            transform=ax3.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax3.set_xlabel("Total de Reuniões por MEP")
    ax3.set_ylabel("Total de Perguntas por MEP")
    ax3.set_title("Relação Total: Reuniões vs Perguntas por MEP")
    ax3.grid(True, alpha=0.3)

    # 6.4 Top 10 MEPs mais ativos
    df_mep_details = pd.read_csv(r"data/mep_detail.csv")
    top_meps = treated_meps_data.nlargest(10, "meetings")

    x_pos = range(len(top_meps))
    bars = ax4.bar(
        x_pos,
        top_meps["meetings"],
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Adicionar valores nas barras
    for i, (bar, meetings) in enumerate(zip(bars, top_meps["meetings"])):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{(meetings)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax4.set_title("Top 10 MEPs com Mais Reuniões\n(Período Completo)")
    ax4.set_xlabel("MEPs")
    ax4.set_ylabel("Total de Reuniões")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(
        [
            f'{df_mep_details.loc[df_mep_details["id"] == top_meps.iloc[i]["member_id"]]["label"].values[0]}'
            for i in range(len(top_meps))
        ],
        rotation=45,
        ha='right',
        fontsize=12,
    )
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "fig6_mep_aggregated_analysis.pdf")
    plt.savefig(output_dir / "fig6_mep_aggregated_analysis.png")
    print("✓ Gráfico 6 salvo: Análise agregada por MEP")
    plt.close()


def plot_domain_treatment_analysis(df, output_dir):
    """Gráfico 7: Análise detalhada de tratamento por domínio"""
    domains = {
        "human_rights": "Direitos Humanos",
        "agriculture": "Agricultura",
        "education": "Educação",
        "health": "Saúde",
        "foreign_and_security_affairs": "Política Externa e Segurança",
        "environment_and_climate": "Ambiente e Clima",
        "economics_and_trade": "Economia e Comércio",
        "technology": "Tecnologia",
        "infrastructure_and_industry": "Infraestrutura e Indústria",
    }

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))

    # 7.1 MEPs únicos tratados por domínio
    domain_mep_treatment = (
        df[df["meetings"] > 0].groupby("domain")["member_id"].nunique().reset_index()
    )
    domain_mep_treatment.columns = ["domain", "treated_meps"]

    # Total de MEPs possíveis por domínio (todos os MEPs)
    total_meps_per_domain = df.groupby("domain")["member_id"].nunique().reset_index()
    total_meps_per_domain.columns = ["domain", "total_meps"]

    # Merge para calcular percentuais
    domain_treatment_stats = domain_mep_treatment.merge(
        total_meps_per_domain, on="domain"
    )
    domain_treatment_stats["treatment_rate"] = (
        domain_treatment_stats["treated_meps"]
        / domain_treatment_stats["total_meps"]
        * 100
    )

    # Ordenar por taxa de tratamento
    domain_treatment_stats = domain_treatment_stats.sort_values(
        "treatment_rate", ascending=True
    )


    # 7 Distribuição temporal do primeiro tratamento por domínio
    # Encontrar primeiro tratamento por MEP-domínio
    first_treatment = (
        df[df["meetings"] > 0]
        .groupby(["member_id", "domain"])["Y-m"]
        .min()
        .reset_index()
    )
    first_treatment.columns = ["member_id", "domain", "first_treatment_period"]

    # Contar primeiros tratamentos por domínio e período
    first_treatment["year"] = pd.to_datetime(
        first_treatment["first_treatment_period"]
    ).dt.year
    domain_temporal = first_treatment.groupby(["domain", "year"]).size().reset_index()
    domain_temporal.columns = ["domain", "year", "first_treatments"]

    # Pegar os 3 domínios com mais MEPs tratados para visualização
    all_domains = domain_treatment_stats["domain"].tolist()

    for i, domain in enumerate(all_domains):
        domain_data = domain_temporal[domain_temporal["domain"] == domain]
        if len(domain_data) > 0:
            ax1.plot(
                domain_data["year"],
                domain_data["first_treatments"],
                marker="o",
                linewidth=2,
                label=domains[domain],
                markersize=6,
            )

    ax1.set_xlabel("Ano")
    ax1.set_ylabel("Número de Primeiros Tratamentos")
    ax1.set_title("Evolução dos Primeiros Tratamentos")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_domain_treatment_analysis.pdf")
    plt.savefig(output_dir / "fig7_domain_treatment_analysis.png")
    print("✓ Gráfico 7 salvo: Análise detalhada de tratamento por domínio")
    plt.close()


def main():
    """Função principal"""
    print("=== GERADOR DE GRÁFICOS DESCRITIVOS ===")
    print("Gerando gráficos de alta qualidade para a tese...\n")

    # Carregar dados
    df = load_data()

    # Criar diretório de saída
    output_dir = create_output_dir()
    print(f"Gráficos serão salvos em: {output_dir}\n")

    # Gerar gráficos
    plot_zero_inflation_analysis(df, output_dir)
    plot_time_series_analysis(df, output_dir)
    plot_domain_heterogeneity(df, output_dir)
    plot_correlation_analysis(df, output_dir)
    plot_extensive_intensive_margins(df, output_dir)
    plot_mep_aggregated_analysis(df, output_dir)
    plot_domain_treatment_analysis(df, output_dir)

    print(f"\n✅ CONCLUÍDO!")
    print(f"7 gráficos gerados com sucesso em {output_dir}")
    print("Formatos: PDF (para LaTeX) e PNG (para visualização)")
    print("\nGráficos gerados:")
    print("- fig1_zero_inflation_analysis: Análise da inflação de zeros")
    print("- fig2_time_series_analysis: Análise temporal")
    print("- fig3_domain_heterogeneity: Heterogeneidade por domínio")
    print("- fig4_correlation_analysis: Análise de correlações")
    print("- fig5_extensive_intensive_margins: Margens extensiva e intensiva")
    print("- fig6_mep_aggregated_analysis: Análise agregada por MEP")
    print(
        "- fig7_domain_treatment_analysis: Análise detalhada de tratamento por domínio"
    )


if __name__ == "__main__":
    main()
