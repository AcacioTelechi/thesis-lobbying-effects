# Guia de Implementação em R - Leads and Lags Analysis

## Pré-requisitos

### Pacotes Necessários
```r
# Instalar pacotes se necessário
install.packages(c("fixest", "modelsummary", "ggplot2", "dplyr"))

# Carregar bibliotecas
library(fixest)
library(modelsummary) 
library(ggplot2)
library(dplyr)
```

### Estrutura dos Dados
O arquivo `df_long_v2.csv` deve conter as seguintes variáveis essenciais:
- `member_id`: Identificador único do parlamentar
- `Y.m`: Data no formato YYYY-MM (será convertida para Date)
- `domain`: Domínio político (agricultura, economia, etc.)
- `meetings`: Número de reuniões (variável de tratamento)
- `questions`: Número de perguntas parlamentares (variável dependente)
- Variáveis de controle (`meps_POLITICAL_GROUP_*`, `meps_COUNTRY_*`, etc.)

## Implementação Paso a Paso

### 1. Preparação dos Dados

```r
# Carregar e preparar dados
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# Converter data
df$Y.m <- as.Date(paste0(df$Y.m, "-01"))
df$member_id <- as.factor(df$member_id)
df$domain <- as.factor(df$domain)

# Ordenar por member_id, domain, e tempo
df <- df %>%
    arrange(member_id, domain, Y.m)
```

### 2. Criação de Leads e Lags

```r
# Função para criar leads e lags
create_leads_lags <- function(data, var_name, n_periods = 3) {
    data %>%
        group_by(member_id, domain) %>%
        arrange(Y.m) %>%
        mutate(
            # Leads (valores futuros)
            !!paste0(var_name, "_lead3") := lead(!!sym(var_name), 3),
            !!paste0(var_name, "_lead2") := lead(!!sym(var_name), 2),
            !!paste0(var_name, "_lead1") := lead(!!sym(var_name), 1),
            
            # Current period
            !!paste0(var_name, "_current") := !!sym(var_name),
            
            # Lags (valores passados)
            !!paste0(var_name, "_lag1") := lag(!!sym(var_name), 1),
            !!paste0(var_name, "_lag2") := lag(!!sym(var_name), 2),
            !!paste0(var_name, "_lag3") := lag(!!sym(var_name), 3)
        ) %>%
        ungroup()
}

# Aplicar função
df <- create_leads_lags(df, "meetings", n_periods = 3)
```

### 3. Estrutura de Efeitos Fixos

```r
# Criar identificadores de efeitos fixos
df$fe_i <- df$member_id # Efeitos fixos individuais
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE) # País × tempo
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE) # Partido × tempo
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE) # Domínio × tempo

# Variável de cluster
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)
```

### 4. Especificação do Modelo

```r
# Definir controles
political_controls <- grep("meps_POLITICAL_GROUP", names(df), value = TRUE)
country_controls <- grep("meps_COUNTRY", names(df), value = TRUE)
committee_controls <- grep("meps_COMMITTEE", names(df), value = TRUE)

controls <- c(political_controls, country_controls, committee_controls)
controls <- controls[controls %in% names(df)]
controls_str <- paste(controls, collapse = " + ")

# Fórmula completa de leads e lags
formula_full_leads_lags <- as.formula(paste0(
    "questions ~ meetings_lead3 + meetings_lead2 + meetings_lead1 + ",
    "meetings_current + meetings_lag1 + meetings_lag2 + meetings_lag3",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))
```

### 5. Estimação PPML

```r
# Modelo principal
m_leads_lags_full <- fepois(
    formula_full_leads_lags,
    data = df,
    cluster = ~cl_dt
)

# Visualizar resultados
summary(m_leads_lags_full)
```

### 6. Testes de Hipóteses

```r
# Teste 1: Ausência de antecipação (leads = 0)
test_no_anticipation <- wald(
    m_leads_lags_full, 
    c("meetings_lead1", "meetings_lead2", "meetings_lead3")
)

# Teste 2: Ausência de persistência (lags = 0)
test_no_persistence <- wald(
    m_leads_lags_full, 
    c("meetings_lag1", "meetings_lag2", "meetings_lag3")
)

# Teste 3: Ausência de dinâmica (todos leads e lags = 0)
dynamic_vars <- c("meetings_lead1", "meetings_lead2", "meetings_lead3",
                  "meetings_lag1", "meetings_lag2", "meetings_lag3")
test_no_dynamics <- wald(m_leads_lags_full, dynamic_vars)

# Imprimir resultados
print(test_no_anticipation)
print(test_no_persistence)
print(test_no_dynamics)
```

### 7. Event Study Plot

```r
# Extrair coeficientes e erros padrão
extract_coef_se <- function(model, coef_name) {
    if (coef_name %in% names(coef(model))) {
        coef_val <- coef(model)[coef_name]
        se_val <- sqrt(diag(vcov(model)))[coef_name]
        return(c(coef_val, se_val))
    } else {
        return(c(NA, NA))
    }
}

# Criar dataset para plotting
coef_names <- c("meetings_lead3", "meetings_lead2", "meetings_lead1", 
                "meetings_current", "meetings_lag1", "meetings_lag2", "meetings_lag3")
periods <- c(-3, -2, -1, 0, 1, 2, 3)

event_study_data <- data.frame(
    period = periods,
    coefficient = numeric(7),
    se = numeric(7)
)

for (i in 1:7) {
    result <- extract_coef_se(m_leads_lags_full, coef_names[i])
    event_study_data$coefficient[i] <- result[1]
    event_study_data$se[i] <- result[2]
}

# Calcular intervalos de confiança
event_study_data$ci_lower <- event_study_data$coefficient - 1.96 * event_study_data$se
event_study_data$ci_upper <- event_study_data$coefficient + 1.96 * event_study_data$se

# Criar gráfico
p_event_study <- ggplot(event_study_data, aes(x = period, y = coefficient)) +
    geom_hline(yintercept = 0, color = "gray70", linetype = "dashed") +
    geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", alpha = 0.5) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                fill = "#1f77b4", alpha = 0.2, na.rm = TRUE) +
    geom_line(color = "#1f77b4", size = 1, na.rm = TRUE) +
    geom_point(color = "#1f77b4", size = 3, na.rm = TRUE) +
    scale_x_continuous(
        breaks = seq(-3, 3, 1),
        labels = c("-3", "-2", "-1", "0", "+1", "+2", "+3"),
        name = "Períodos relativos ao tratamento"
    ) +
    labs(
        title = "Event Study: Efeitos Dinâmicos do Lobbying",
        subtitle = "Modelo PPML com efeitos fixos completos",
        y = "Coeficiente (log points)",
        caption = "IC 95% com clustering por domínio×tempo"
    ) +
    theme_minimal()

# Salvar gráfico
ggsave("Tese/figures/leads_lags/event_study_leads_lags.pdf", 
       p_event_study, width = 10, height = 6)
ggsave("Tese/figures/leads_lags/event_study_leads_lags.png", 
       p_event_study, width = 10, height = 6, dpi = 300)
```

### 8. Tabelas de Resultados

```r
# Criar diferentes especificações para comparação
formula_leads_only <- as.formula(paste0(
    "questions ~ meetings_lead3 + meetings_lead2 + meetings_lead1 + meetings_current",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

formula_lags_only <- as.formula(paste0(
    "questions ~ meetings_current + meetings_lag1 + meetings_lag2 + meetings_lag3",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_leads_only <- fepois(formula_leads_only, data = df, cluster = ~cl_dt)
m_lags_only <- fepois(formula_lags_only, data = df, cluster = ~cl_dt)

# Tabela comparativa
models_list <- list(
    "Apenas Leads" = m_leads_only,
    "Apenas Lags" = m_lags_only,
    "Modelo Completo" = m_leads_lags_full
)

msummary(
    models_list,
    coef_omit = "meps_",  # Omitir controles na exibição
    gof_omit = "IC|Log|Adj|Pseudo|Within",
    stars = TRUE,
    output = "Tese/tables/leads_lags/leads_lags_main.tex",
    title = "Análise de Leads e Lags - Resultados PPML"
)
```

## Verificações de Robustez Recomendadas

### 1. Clustering Alternativo
```r
# Clustering por membro
m_robust_member <- fepois(formula_full_leads_lags, data = df, cluster = ~member_id)

# Clustering bidirecional (se disponível)
# m_robust_twoway <- fepois(formula_full_leads_lags, data = df, 
#                          cluster = ~member_id + cl_dt)
```

### 2. Especificação OLS
```r
m_ols_leads_lags <- feols(
    formula_full_leads_lags,
    data = df,
    cluster = ~cl_dt
)
```

### 3. Janelas Temporais Alternativas
```r
# Criar leads/lags ±2 períodos apenas
df <- create_leads_lags(df, "meetings", n_periods = 2)

# Re-estimar com janela mais curta
formula_short <- as.formula(paste0(
    "questions ~ meetings_lead2 + meetings_lead1 + meetings_current + ",
    "meetings_lag1 + meetings_lag2",
    if(length(controls) > 0) paste(" +", controls_str) else "",
    " | fe_ct + fe_pt + fe_dt"
))

m_short_window <- fepois(formula_short, data = df, cluster = ~cl_dt)
```

## Diagnósticos e Verificações

### 1. Verificar Balanceamento do Painel
```r
# Verificar se temos observações suficientes para leads/lags
table(is.na(df$meetings_lead3))
table(is.na(df$meetings_lag3))
```

### 2. Estatísticas Descritivas
```r
# Resumo das variáveis de leads/lags
summary(df[, c("meetings_lead3", "meetings_lead2", "meetings_lead1",
                "meetings_current", "meetings_lag1", "meetings_lag2", "meetings_lag3")])
```

### 3. Convergência do Modelo
```r
# Verificar convergência da estimação PPML
summary(m_leads_lags_full)$convergence
```

## Interpretação dos Resultados

### Critérios de Avaliação:
1. **Efeitos de antecipação**: Coeficientes de leads devem ser não significativos
2. **Efeito contemporâneo**: Deve ser positivo e significativo
3. **Persistência**: Lags podem ser significativos (evidência de persistência informacional)
4. **Testes de Wald**: Confirmar resultados dos coeficientes individuais

### Red Flags:
- Leads significativos (problema de causalidade reversa)
- Padrão não monotônico nos lags sem justificativa teórica
- Instabilidade através de especificações robustez

Este guia fornece implementação completa e rigorosa da análise de leads e lags para efeitos de lobbying usando R e fixest.