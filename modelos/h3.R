# =========================================================================
# H3) O efeito do lobby não empresarial é maior em temas mais salientes
# =========================================================================

# install.packages("fixest")
# install.packages("modelsummary")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("marginaleffects")
# install.packages("sandwich")

library(fixest)
library(modelsummary)
library(ggplot2)
library(dplyr)
library(marginaleffects)
library(boot)

# --- Diretórios ---
figures_dir <- file.path("Tese", "figures", "h3_test")
tables_dir <- file.path("Tese", "tables", "h3_test")
outputs_dir <- file.path("outputs", "h3_test")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)
if (!dir.exists(tables_dir)) dir.create(tables_dir, recursive = TRUE)
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)

# --- Carregar e Preparar Dados ---
df <- read.csv("./data/gold/df_long_v2.csv", stringsAsFactors = TRUE)

# 1) Efeitos Fixos e Clusters (consistente com H1 e H2)
df$fe_i <- df$member_id
df$fe_ct <- interaction(df$meps_country, df$Y.m, drop = TRUE)
df$fe_pt <- interaction(df$meps_party, df$Y.m, drop = TRUE)
df$fe_dt <- interaction(df$domain, df$Y.m, drop = TRUE)
df$cl_dt <- interaction(df$domain, df$Y.m, drop = TRUE)

# 2) Criar a variável de Saliência
# Proxy: log do total de reuniões no domínio x tempo.
# Usamos log(1 + x) para lidar com zeros e reduzir a assimetria.
df <- df %>%
  group_by(domain, Y.m) %>%
  mutate(salience_raw = sum(meetings, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    salience = log1p(salience_raw),
    # Padronizamos para facilitar a interpretação dos coeficientes
    salience_std = as.numeric(scale(salience))
  )

# --- Construir e Rodar o Modelo de Interação ---

# 1) Controles (consistente com H1 e H2)
controls <- c(
    "meps_POLITICAL_GROUP_5148.0", "meps_POLITICAL_GROUP_5151.0", "meps_POLITICAL_GROUP_5152.0",
    "meps_POLITICAL_GROUP_5153.0", "meps_POLITICAL_GROUP_5154.0", "meps_POLITICAL_GROUP_5155.0",
    "meps_POLITICAL_GROUP_5588.0", "meps_POLITICAL_GROUP_5704.0", "meps_POLITICAL_GROUP_6259.0",
    "meps_POLITICAL_GROUP_6561.0", "meps_POLITICAL_GROUP_7018.0", "meps_POLITICAL_GROUP_7028.0",
    "meps_POLITICAL_GROUP_7035.0", "meps_POLITICAL_GROUP_7036.0", "meps_POLITICAL_GROUP_7037.0",
    "meps_POLITICAL_GROUP_7038.0", "meps_POLITICAL_GROUP_7150.0", "meps_POLITICAL_GROUP_7151.0",
    "meps_COUNTRY_AUT", "meps_COUNTRY_BEL", "meps_COUNTRY_BGR", "meps_COUNTRY_CYP", "meps_COUNTRY_CZE",
    "meps_COUNTRY_DEU", "meps_COUNTRY_DNK", "meps_COUNTRY_ESP", "meps_COUNTRY_EST", "meps_COUNTRY_FIN",
    "meps_COUNTRY_FRA", "meps_COUNTRY_GBR", "meps_COUNTRY_GRC", "meps_COUNTRY_HRV", "meps_COUNTRY_HUN",
    "meps_COUNTRY_IRL", "meps_COUNTRY_ITA", "meps_COUNTRY_LTU", "meps_COUNTRY_LUX", "meps_COUNTRY_LVA",
    "meps_COUNTRY_MLT", "meps_COUNTRY_NLD", "meps_COUNTRY_POL", "meps_COUNTRY_PRT", "meps_COUNTRY_ROU",
    "meps_COUNTRY_SVK", "meps_COUNTRY_SVN", "meps_COUNTRY_SWE",
    "meps_COMMITTEE_PARLIAMENTARY_SPECIAL___CHAIR", "meps_COMMITTEE_PARLIAMENTARY_SPECIAL___MEMBER",
    "meps_COMMITTEE_PARLIAMENTARY_STANDING___CHAIR", "meps_COMMITTEE_PARLIAMENTARY_STANDING___MEMBER",
    "meps_COMMITTEE_PARLIAMENTARY_SUB___CHAIR", "meps_COMMITTEE_PARLIAMENTARY_SUB___MEMBER",
    "meps_DELEGATION_PARLIAMENTARY___CHAIR", "meps_DELEGATION_PARLIAMENTARY___MEMBER",
    "meps_EU_INSTITUTION___PRESIDENT", "meps_EU_INSTITUTION___QUAESTOR", "meps_EU_POLITICAL_GROUP___CHAIR",
    "meps_EU_POLITICAL_GROUP___MEMBER_BUREAU", "meps_EU_POLITICAL_GROUP___TREASURER", "meps_EU_POLITICAL_GROUP___TREASURER_CO",
    "meps_NATIONAL_CHAMBER___PRESIDENT_VICE", "meps_WORKING_GROUP___CHAIR", "meps_WORKING_GROUP___MEMBER",
    "meps_WORKING_GROUP___MEMBER_BUREAU"
)
controls_str <- paste(controls, collapse = " + ")

# 2) Fórmula do modelo com interações
# Interagimos cada tipo de lobby com a saliência padronizada
formula_h3_str <- paste0(
  "questions ~ l_category_Business*salience_std + l_category_NGOs*salience_std + l_category_Other*salience_std + ",
  controls_str,
  " | fe_ct + fe_pt + fe_dt"
)
formula_h3 <- as.formula(formula_h3_str)

# 3) Rodar o modelo PPML
m_h3_interaction <- fepois(
  formula_h3,
  data    = df,
  cluster = ~cl_dt
)

# --- Apresentar Resultados ---

# 1) Tabela de Regressão
msummary(
  list("PPML com Interação (H3)" = m_h3_interaction),
  gof_omit = "IC|Log|Adj|Pseudo|Within",
  coef_map = c(
    "l_category_Business" = "Empresa (base)",
    "l_category_NGOs" = "ONG (base)",
    "l_category_Other" = "Outros (base)",
    "salience_std" = "Saliência",
    "l_category_Business:salience_std" = "Empresa x Saliência",
    "salience_std:l_category_NGOs" = "ONG x Saliência",
    "salience_std:l_category_Other" = "Outros x Saliência"
  ),
  stars = TRUE,
  output = file.path(tables_dir, "tab_h3_interaction.tex")
)



#========= tentativa 2

# 2) Cálculo Manual dos Efeitos Marginais
# Extrair coeficientes do modelo
coefs <- coef(m_h3_interaction)
vcov_matrix <- vcov(m_h3_interaction)

# Criar grid de saliência
salience_grid <- seq(min(df$salience_std, na.rm = TRUE), 
                     max(df$salience_std, na.rm = TRUE), 
                     length.out = 50)

# Função para calcular efeito marginal de cada categoria
calc_marginal_effect <- function(salience_val, category) {
  if (category == "Empresa") {
    # Efeito marginal = coef_base + coef_interaction * salience
    base_coef <- coefs["l_category_Business"]
    interaction_coef <- coefs["l_category_Business:salience_std"]
  } else if (category == "ONG") {
    base_coef <- coefs["l_category_NGOs"]
    interaction_coef <- coefs["salience_std:l_category_NGOs"]
  } else if (category == "Outros") {
    base_coef <- coefs["l_category_Other"]
    interaction_coef <- coefs["salience_std:l_category_Other"]
  }
  
  return(base_coef + interaction_coef * salience_val)
}

# Calcular efeitos marginais para cada categoria
marginal_effects <- data.frame(
  salience_std = rep(salience_grid, 3),
  category = rep(c("Empresa", "ONG", "Outros"), each = length(salience_grid)),
  effect = c(
    sapply(salience_grid, function(x) calc_marginal_effect(x, "Empresa")),
    sapply(salience_grid, function(x) calc_marginal_effect(x, "ONG")),
    sapply(salience_grid, function(x) calc_marginal_effect(x, "Outros"))
  )
)

# 3) Bootstrap para calcular intervalos de confiança
# Função para bootstrap
boot_function <- function(data, indices) {
  # Amostrar com reposição
  boot_data <- data[indices, ]
  
  # Rodar modelo bootstrap
  boot_model <- fepois(
    formula_h3,
    data    = boot_data,
    cluster = ~cl_dt
  )
  
  # Extrair coeficientes
  boot_coefs <- coef(boot_model)
  
  # Calcular efeitos marginais para cada categoria
  effects <- c(
    sapply(salience_grid, function(x) {
      boot_coefs["l_category_Business"] + boot_coefs["l_category_Business:salience_std"] * x
    }),
    sapply(salience_grid, function(x) {
      boot_coefs["l_category_NGOs"] + boot_coefs["salience_std:l_category_NGOs"] * x
    }),
    sapply(salience_grid, function(x) {
      boot_coefs["l_category_Other"] + boot_coefs["salience_std:l_category_Other"] * x
    })
  )
  
  return(effects)
}

# Executar bootstrap (reduzido para velocidade)
set.seed(123)
boot_results <- boot(data = df, statistic = boot_function, R = 100)

# Calcular intervalos de confiança
boot_ci <- boot.ci(boot_results, type = "perc")

# Organizar resultados do bootstrap
boot_effects <- data.frame(
  salience_std = rep(salience_grid, 3),
  category = rep(c("Empresa", "ONG", "Outros"), each = length(salience_grid)),
  effect = boot_results$t0,
  ci_lower = NA,
  ci_upper = NA
)

# Para simplificar, vamos usar aproximação normal para os ICs
# (mais rápido que extrair todos os percentis do bootstrap)
for (i in 1:nrow(marginal_effects)) {
  # Calcular erro padrão do bootstrap
  boot_se <- sd(boot_results$t[, i], na.rm = TRUE)
  
  # Calcular ICs usando aproximação normal
  boot_effects$ci_lower[i] <- marginal_effects$effect[i] - 1.96 * boot_se
  boot_effects$ci_upper[i] <- marginal_effects$effect[i] + 1.96 * boot_se
}

# 4) Criar gráfico
p_manual <- ggplot(boot_effects, aes(x = salience_std, y = effect, color = category, fill = category)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, color = NA) +
  labs(
    x = "Nível de Saliência do Tema (padronizado)",
    y = "Efeito Marginal Esperado de uma Reunião",
    # title = "Efeito do Lobby Condicional à Saliência do Tema",
    # subtitle = "Linhas representam o efeito marginal de uma reunião sobre o n. de perguntas em diferentes níveis de saliência.",
    color = "Tipo de Ator",
    fill = "Tipo de Ator"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

ggsave(
  file.path(figures_dir, "fig_h3_marginal_effects_manual.pdf"),
  p_manual, width = 10, height = 7
)
ggsave(
  file.path(figures_dir, "fig_h3_marginal_effects_manual.png"),
  p_manual, width = 10, height = 7, dpi = 300
)

# Mensagens de conclusão
message("Script H3 (manual) concluído.")
message("Tabela salva em: ", file.path(tables_dir, "tab_h3_interaction.tex"))
message("Gráfico salvo em: ", file.path(figures_dir, "fig_h3_marginal_effects_manual.pdf"))
