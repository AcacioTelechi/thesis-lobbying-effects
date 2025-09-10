#!/usr/bin/env python3
"""
Análise da especialização temática dos MEPs e recálculo da inflação de zeros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Carregar dados"""
    print("Carregando dados...")
    df = pd.read_csv("df_long.csv")
    print(f"Dados carregados: {len(df):,} observações")
    return df

def analyze_mep_specialization(df):
    """Analisar especialização temática dos MEPs baseada apenas em perguntas"""
    print("\n=== ANÁLISE DE ESPECIALIZAÇÃO TEMÁTICA (BASEADA EM PERGUNTAS) ===")
    
    # 1. Calcular atividade por MEP-domínio (agregando tempo)
    mep_domain_activity = df.groupby(['member_id', 'domain']).agg({
        'questions': 'sum',
        'meetings': 'sum'  # Mantemos meetings para usar na análise de zeros
    }).reset_index()
    
    # 2. Para cada MEP, identificar quantos domínios ele atua (baseado APENAS em perguntas)
    mep_active_domains = mep_domain_activity[
        (mep_domain_activity['questions'] > 0)
    ].groupby('member_id').size().reset_index()
    mep_active_domains.columns = ['member_id', 'num_active_domains']
    
    print(f"Número médio de domínios ativos por MEP: {mep_active_domains['num_active_domains'].mean():.2f}")
    print(f"Mediana de domínios ativos por MEP: {mep_active_domains['num_active_domains'].median():.1f}")
    print(f"MEPs que atuam em apenas 1 domínio: {(mep_active_domains['num_active_domains'] == 1).sum()}")
    print(f"MEPs que atuam em 2-3 domínios: {((mep_active_domains['num_active_domains'] >= 2) & (mep_active_domains['num_active_domains'] <= 3)).sum()}")
    print(f"MEPs que atuam em 4+ domínios: {(mep_active_domains['num_active_domains'] >= 4).sum()}")
    
    # 3. Calcular concentração de atividade por MEP baseada APENAS em perguntas (Índice Herfindahl)
    def calculate_herfindahl_questions(group):
        total_questions = group['questions'].sum()
        if total_questions == 0:
            return 0
        shares = group['questions'] / total_questions
        return (shares ** 2).sum()
    
    mep_concentration = mep_domain_activity.groupby('member_id').apply(
        calculate_herfindahl_questions
    ).reset_index()
    mep_concentration.columns = ['member_id', 'herfindahl_index']
    
    print(f"\nÍndice Herfindahl médio (baseado em perguntas): {mep_concentration['herfindahl_index'].mean():.3f}")
    print(f"MEPs altamente especializados (HHI > 0.8): {(mep_concentration['herfindahl_index'] > 0.8).sum()}")
    print(f"MEPs moderadamente especializados (HHI 0.4-0.8): {((mep_concentration['herfindahl_index'] >= 0.4) & (mep_concentration['herfindahl_index'] <= 0.8)).sum()}")
    print(f"MEPs generalistas (HHI < 0.4): {(mep_concentration['herfindahl_index'] < 0.4).sum()}")
    
    return mep_active_domains, mep_concentration, mep_domain_activity

def analyze_corrected_zero_inflation(df, mep_domain_activity):
    """Análise corrigida da inflação de zeros"""
    print("\n=== ANÁLISE CORRIGIDA DE INFLAÇÃO DE ZEROS ===")
    
    # 1. Nível MEP-Tempo (agregando domínios onde MEP é ativo)
    print("\n1. Nível MEP-Tempo (apenas domínios onde MEP é ativo):")
    
    # Identificar MEPs ativos por domínio (baseado APENAS em perguntas)
    active_mep_domains = mep_domain_activity[
        (mep_domain_activity['questions'] > 0)
    ][['member_id', 'domain']]
    
    # Filtrar observações apenas para MEP-domínios onde há atividade
    df_active_domains = df.merge(active_mep_domains, on=['member_id', 'domain'])
    
    # Agregar por MEP-tempo
    mep_time_agg = df_active_domains.groupby(['member_id', 'Y-m']).agg({
        'questions': 'sum',
        'meetings': 'sum'
    }).reset_index()
    
    questions_zeros_mep_time = (mep_time_agg['questions'] == 0).sum()
    meetings_zeros_mep_time = (mep_time_agg['meetings'] == 0).sum()
    total_obs_mep_time = len(mep_time_agg)
    
    print(f"Total observações MEP-tempo (domínios ativos): {total_obs_mep_time:,}")
    print(f"Zeros em perguntas: {questions_zeros_mep_time:,} ({questions_zeros_mep_time/total_obs_mep_time*100:.1f}%)")
    print(f"Zeros em reuniões: {meetings_zeros_mep_time:,} ({meetings_zeros_mep_time/total_obs_mep_time*100:.1f}%)")
    
    # 2. Nível Domínio-Tempo (agregando MEPs ativos)
    print("\n2. Nível Domínio-Tempo (agregando MEPs ativos):")
    
    domain_time_agg = df_active_domains.groupby(['domain', 'Y-m']).agg({
        'questions': 'sum',
        'meetings': 'sum'
    }).reset_index()
    
    questions_zeros_domain_time = (domain_time_agg['questions'] == 0).sum()
    meetings_zeros_domain_time = (domain_time_agg['meetings'] == 0).sum()
    total_obs_domain_time = len(domain_time_agg)
    
    print(f"Total observações domínio-tempo: {total_obs_domain_time:,}")
    print(f"Zeros em perguntas: {questions_zeros_domain_time:,} ({questions_zeros_domain_time/total_obs_domain_time*100:.1f}%)")
    print(f"Zeros em reuniões: {meetings_zeros_domain_time:,} ({meetings_zeros_domain_time/total_obs_domain_time*100:.1f}%)")
    
    # 3. Comparação com análise original (MEP-domínio-tempo)
    print("\n3. Comparação com análise original (MEP-domínio-tempo):")
    questions_zeros_original = (df['questions'] == 0).sum()
    meetings_zeros_original = (df['meetings'] == 0).sum()
    total_obs_original = len(df)
    
    print(f"Original - Total observações: {total_obs_original:,}")
    print(f"Original - Zeros em perguntas: {questions_zeros_original:,} ({questions_zeros_original/total_obs_original*100:.1f}%)")
    print(f"Original - Zeros em reuniões: {meetings_zeros_original:,} ({meetings_zeros_original/total_obs_original*100:.1f}%)")
    
    return mep_time_agg, domain_time_agg

def analyze_specialization_by_questions_only(df):
    """Análise de especialização considerando APENAS proporção de perguntas por domínio"""
    print("\n=== ANÁLISE DE ESPECIALIZAÇÃO BASEADA APENAS EM PERGUNTAS ===")
    
    # 1. Calcular apenas atividade de perguntas por MEP-domínio
    mep_domain_questions = df.groupby(['member_id', 'domain']).agg({
        'questions': 'sum'
    }).reset_index()
    
    # 2. Para cada MEP, calcular proporção de perguntas por domínio
    mep_total_questions = mep_domain_questions.groupby('member_id')['questions'].sum().reset_index()
    mep_total_questions.columns = ['member_id', 'total_questions']
    
    # Merge para calcular proporções
    mep_domain_prop = mep_domain_questions.merge(mep_total_questions, on='member_id')
    mep_domain_prop['proportion'] = mep_domain_prop['questions'] / mep_domain_prop['total_questions']
    
    # 3. Calcular HHI baseado apenas em perguntas
    def calculate_herfindahl_questions_only(group):
        total_questions = group['questions'].sum()
        if total_questions == 0:
            return 0
        shares = group['questions'] / total_questions
        return (shares ** 2).sum()
    
    mep_concentration_questions = mep_domain_questions.groupby('member_id').apply(
        calculate_herfindahl_questions_only
    ).reset_index()
    mep_concentration_questions.columns = ['member_id', 'herfindahl_questions_only']
    
    print(f"HHI médio (apenas perguntas): {mep_concentration_questions['herfindahl_questions_only'].mean():.3f}")
    print(f"MEPs altamente especializados (HHI > 0.8): {(mep_concentration_questions['herfindahl_questions_only'] > 0.8).sum()}")
    print(f"MEPs moderadamente especializados (HHI 0.4-0.8): {((mep_concentration_questions['herfindahl_questions_only'] >= 0.4) & (mep_concentration_questions['herfindahl_questions_only'] <= 0.8)).sum()}")
    print(f"MEPs generalistas (HHI < 0.4): {(mep_concentration_questions['herfindahl_questions_only'] < 0.4).sum()}")
    
    # 4. Identificar domínios ativos baseado apenas em perguntas
    mep_active_domains_questions = mep_domain_questions[
        mep_domain_questions['questions'] > 0
    ].groupby('member_id').size().reset_index()
    mep_active_domains_questions.columns = ['member_id', 'num_active_domains_questions']
    
    print(f"\nNúmero médio de domínios ativos (perguntas): {mep_active_domains_questions['num_active_domains_questions'].mean():.2f}")
    print(f"Mediana de domínios ativos (perguntas): {mep_active_domains_questions['num_active_domains_questions'].median():.1f}")
    
    # 5. Mostrar exemplos de MEPs mais e menos especializados
    print("\n=== EXEMPLOS DE ESPECIALIZAÇÃO ===")
    
    # MEPs mais especializados (HHI alto)
    top_specialized = mep_concentration_questions.nlargest(5, 'herfindahl_questions_only')
    print("\nTop 5 MEPs mais especializados (baseado em perguntas):")
    for _, row in top_specialized.iterrows():
        mep_id = row['member_id']
        hhi = row['herfindahl_questions_only']
        
        # Pegar distribuição de perguntas deste MEP
        mep_dist = mep_domain_questions[mep_domain_questions['member_id'] == mep_id]
        mep_dist = mep_dist[mep_dist['questions'] > 0].sort_values('questions', ascending=False)
        
        print(f"MEP {mep_id} (HHI: {hhi:.3f}):")
        for _, domain_row in mep_dist.head(3).iterrows():
            domain = domain_row['domain']
            questions = domain_row['questions']
            total = mep_total_questions[mep_total_questions['member_id'] == mep_id]['total_questions'].iloc[0]
            prop = questions / total * 100
            print(f"  - {domain}: {questions} perguntas ({prop:.1f}%)")
        print()
    
    return mep_concentration_questions, mep_domain_prop, mep_active_domains_questions

def create_corrected_plots(df, mep_time_agg, domain_time_agg, mep_active_domains, output_dir):
    """Criar gráficos corrigidos"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Figura 1: Comparação de inflação de zeros por nível de agregação
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1.1 Original (MEP-domínio-tempo)
    original_questions_prop = [(df['questions'] == 0).sum(), (df['questions'] > 0).sum()]
    original_meetings_prop = [(df['meetings'] == 0).sum(), (df['meetings'] > 0).sum()]
    
    labels = ['Zeros', 'Positivos']
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, [p/len(df)*100 for p in original_questions_prop], width, 
           label='Perguntas', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, [p/len(df)*100 for p in original_meetings_prop], width, 
           label='Reuniões', alpha=0.8, color='lightcoral')
    ax1.set_title('Original: MEP-Domínio-Tempo\n(Incluindo domínios inativos)')
    ax1.set_ylabel('Percentual (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, (q_prop, m_prop) in enumerate(zip(original_questions_prop, original_meetings_prop)):
        ax1.text(i - width/2, q_prop/len(df)*100 + 1, f'{q_prop/len(df)*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, m_prop/len(df)*100 + 1, f'{m_prop/len(df)*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # 1.2 Corrigido (MEP-tempo, domínios ativos)
    mep_time_questions_prop = [(mep_time_agg['questions'] == 0).sum(), (mep_time_agg['questions'] > 0).sum()]
    mep_time_meetings_prop = [(mep_time_agg['meetings'] == 0).sum(), (mep_time_agg['meetings'] > 0).sum()]
    
    ax2.bar(x - width/2, [p/len(mep_time_agg)*100 for p in mep_time_questions_prop], width, 
           label='Perguntas', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, [p/len(mep_time_agg)*100 for p in mep_time_meetings_prop], width, 
           label='Reuniões', alpha=0.8, color='lightcoral')
    ax2.set_title('Corrigido: MEP-Tempo\n(Apenas domínios ativos)')
    ax2.set_ylabel('Percentual (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, (q_prop, m_prop) in enumerate(zip(mep_time_questions_prop, mep_time_meetings_prop)):
        ax2.text(i - width/2, q_prop/len(mep_time_agg)*100 + 1, f'{q_prop/len(mep_time_agg)*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
        ax2.text(i + width/2, m_prop/len(mep_time_agg)*100 + 1, f'{m_prop/len(mep_time_agg)*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # 1.3 Especialização dos MEPs (número de domínios ativos)
    specialization_data = mep_active_domains['num_active_domains'].value_counts().sort_index()
    
    bars = ax3.bar(specialization_data.index, specialization_data.values, 
                   alpha=0.8, color='forestgreen', edgecolor='black', linewidth=0.5)
    ax3.set_title('Especialização Temática dos MEPs\n(Número de Domínios Ativos)')
    ax3.set_xlabel('Número de Domínios Ativos')
    ax3.set_ylabel('Número de MEPs')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, specialization_data.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 1.4 Distribuição temporal da atividade (MEP-tempo vs original)
    # Calcular totais mensais
    original_monthly = df.groupby('Y-m').agg({
        'questions': 'sum',
        'meetings': 'sum'
    })
    
    corrected_monthly = mep_time_agg.groupby('Y-m').agg({
        'questions': 'sum',
        'meetings': 'sum'
    })
    
    # Converter índice para datetime para melhor visualização
    original_monthly.index = pd.to_datetime(original_monthly.index)
    corrected_monthly.index = pd.to_datetime(corrected_monthly.index)
    
    ax4.plot(original_monthly.index, original_monthly['meetings'], 
            label='Original (MEP-Dom-Tempo)', alpha=0.7, linewidth=2)
    ax4.plot(corrected_monthly.index, corrected_monthly['meetings'], 
            label='Corrigido (MEP-Tempo)', alpha=0.7, linewidth=2)
    
    ax4.set_title('Evolução Temporal: Reuniões Totais\n(Comparação de Métodos)')
    ax4.set_xlabel('Período')
    ax4.set_ylabel('Total de Reuniões')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_corrected_zero_inflation_analysis.pdf')
    plt.savefig(output_dir / 'fig_corrected_zero_inflation_analysis.png')
    print("✓ Gráfico de análise corrigida de inflação de zeros salvo")
    plt.close()

def main():
    """Função principal"""
    print("=== ANÁLISE DE ESPECIALIZAÇÃO E INFLAÇÃO DE ZEROS CORRIGIDA ===")
    
    # Carregar dados
    df = load_data()
    
    # Analisar especialização (método original: perguntas + reuniões)
    mep_active_domains, mep_concentration, mep_domain_activity = analyze_mep_specialization(df)
    
    # Analisar especialização baseada APENAS em perguntas
    mep_concentration_questions, mep_domain_prop, mep_active_domains_questions = analyze_specialization_by_questions_only(df)
    
    # Análise corrigida de inflação de zeros
    mep_time_agg, domain_time_agg = analyze_corrected_zero_inflation(df, mep_domain_activity)
    
    # Criar gráficos
    output_dir = Path("Tese/figures")
    output_dir.mkdir(exist_ok=True)
    
    create_corrected_plots(df, mep_time_agg, domain_time_agg, mep_active_domains, output_dir)
    
    # Salvar dados agregados para uso posterior
    mep_time_agg.to_csv("mep_time_aggregated.csv", index=False)
    domain_time_agg.to_csv("domain_time_aggregated.csv", index=False)
    mep_active_domains.to_csv("mep_specialization.csv", index=False)
    mep_concentration_questions.to_csv("mep_specialization_questions_only.csv", index=False)
    mep_domain_prop.to_csv("mep_domain_proportions.csv", index=False)
    
    print(f"\n✅ ANÁLISE CONCLUÍDA!")
    print("Arquivos gerados:")
    print("- fig_corrected_zero_inflation_analysis.pdf/png")
    print("- mep_time_aggregated.csv")
    print("- domain_time_aggregated.csv")
    print("- mep_specialization.csv")
    print("- mep_specialization_questions_only.csv")
    print("- mep_domain_proportions.csv")

if __name__ == "__main__":
    main()
