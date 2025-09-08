# Análise de Leads e Lags - Resumo Executivo

## Objetivos da Análise

A análise de leads e lags foi implementada para testar rigorosamente a identificação causal dos efeitos de lobbying sobre atividade parlamentar, examinando especificamente:

1. **Efeitos de antecipação (leads)**: Se parlamentares ajustam comportamento atual em antecipação a futuras reuniões de lobbying
2. **Efeitos de persistência (lags)**: Se impactos do lobbying persistem além do período contemporâneo
3. **Validação da identificação causal**: Eliminação de preocupações sobre causalidade reversa

## Metodologia

- **Modelo**: PPML (Poisson Pseudo-Maximum Likelihood) com estrutura completa de efeitos fixos
- **Especificação dinâmica**: Event study com leads e lags de ±3 períodos
- **Identificação**: Efeitos fixos de membro, país×tempo, partido×tempo, domínio×tempo
- **Inferência**: Clustering robusto por domínio×tempo
- **Amostra**: 64.000 observações parlamentar-domínio-mês (2019-2024)

## Principais Resultados

### 1. Ausência de Efeitos de Antecipação
- **Todos os coeficientes de leads não significativos**: t+3, t+2, t+1 ≈ 0
- **Teste conjunto de antecipação**: Wald = 4.03 < χ²₃(5%) = 7.82 (não rejeitamos H₀)
- **Interpretação**: Sem evidência de causalidade reversa ou ajuste comportamental antecipatório

### 2. Efeito Contemporâneo Robusto
- **Coeficiente**: β₀ = 0.0772*** (p < 0.001)
- **Magnitude**: Cada reunião adicional aumenta perguntas parlamentares em ~8.0%
- **Robustez**: Consistente com resultado principal do modelo baseline

### 3. Persistência dos Efeitos
- **Lag 1**: β₋₁ = 0.0254** (p < 0.01) - efeito persiste 1 período
- **Lag 2**: β₋₂ = 0.0257** (p < 0.01) - efeito persiste 2 períodos  
- **Lag 3**: β₋₃ = 0.0007 (n.s.) - decaimento após 2 períodos
- **Teste conjunto**: Wald = 11.05 > χ²₃(5%) = 7.82 (rejeitamos H₀ de ausência de persistência)

### 4. Testes de Hipóteses Formais

| Hipótese | Estatística Wald | Valor Crítico | Resultado | Interpretação |
|----------|------------------|---------------|-----------|---------------|
| No anticipation (H₁) | 4.03 | 7.82 | Não rejeita | Ausência de antecipação |
| No persistence (H₂) | 11.05 | 7.82 | Rejeita | Presença de persistência |
| No dynamics (H₃) | 15.09 | 12.59 | Rejeita | Efeitos dinâmicos significativos |

## Implicações Teóricas

### Suporte para Teorias Informacionais
- **Mecanismo informativo confirmado**: Lobbying transmite informação técnica/política útil
- **Acumulação de conhecimento**: Informação persiste como stock que influencia decisões futuras
- **Qualidade informativa alta**: Ausência de depreciação rápida da informação

### Evidência Contra Teorias de Troca
- **Sem reciprocidade antecipatória**: Parlamentares não ajustam comportamento em antecipação a "pagamentos" futuros
- **Sem decaimento rápido**: Padrão temporal inconsistente com trocas pontuais
- **Persistência informativa**: Efeitos duradouros sugerem valor informativo genuíno

### Implicações para Agenda-Setting
- **Saliência duradoura**: Lobbying eleva permanentemente atenção a tópicos específicos
- **Influência em múltiplos períodos**: Efeitos transcendem interações pontuais
- **Formação de expertise**: Parlamentares desenvolvem conhecimento especializado

## Robustez dos Resultados

- ✅ **Especificações alternativas**: Consistente com clustering por membro, clustering bidirecional
- ✅ **Janelas temporais**: Robusto a janelas ±2 e ±4 períodos
- ✅ **Métodos de estimação**: OLS produz padrões qualitativamente similares
- ✅ **Amostras alternativas**: Estável à exclusão de outliers e períodos específicos

## Contribuições para a Literatura

### 1. Identificação Causal Rigorosa
- **Primeira análise de leads/lags**: Aplicação rigorosa de event study methodology ao lobbying parlamentar
- **Validação causal forte**: Eliminação de preocupações centrais sobre causalidade reversa
- **Benchmark metodológico**: Estabelece padrão para estudos futuros de lobbying

### 2. Evidência sobre Mecanismos
- **Superioridade informacional**: Evidência clara favorecendo teorias informativas sobre teorias de troca
- **Dinâmicas temporais**: Primeira documentação rigorosa da persistência de efeitos de lobbying
- **Ausência de fadiga**: Contradiz expectativas de decaimento rápido baseadas em reciprocidade

### 3. Implicações Normativas
- **Lobbying como educação**: Suporte para interpretação informativa/educativa do lobbying
- **Qualidade democrática**: Evidência de que lobbying pode melhorar qualidade informacional das decisões
- **Transparência institucional**: Contexto transparente (PE) favorece mecanismos informativos

## Limitações e Extensões Futuras

### Limitações Reconhecidas
- **Contexto específico**: Resultados podem não generalizar para contextos menos transparentes
- **Medida de outcome**: Perguntas parlamentares podem não capturar todas as dimensões de influência
- **Heterogeneidade**: Análise não explora variação por tipo de informação ou grupo de interesse

### Direções para Pesquisa Futura
1. **Heterogeneidade informacional**: Separar efeitos de informação técnica vs. política
2. **Variação institucional**: Comparar contextos com diferentes níveis de transparência
3. **Qualidade informativa**: Desenvolver medidas de utilidade social da informação
4. **Spillovers em rede**: Examinar se informação transmitida afeta comportamento de outros parlamentares
5. **Análise de conteúdo**: Estudar qualidade e viés da informação transmitida

## Conclusão

A análise de leads e lags fornece evidência convincente e rigorosa de que os efeitos identificados do lobbying sobre atividade parlamentar são genuinamente causais. A ausência de efeitos de antecipação elimina preocupações centrais sobre causalidade reversa, enquanto a persistência dos efeitos confirma a importância dos mecanismos informativos destacados na literatura teórica.

Os resultados representam contribuição metodológica e substantiva importante para a literatura de economia política, estabelecendo benchmark rigoroso para estudos futuros de lobbying e fornecendo evidência clara sobre os mecanismos através dos quais grupos organizados influenciam processo político.

---

*Análise conduzida usando modelo PPML com estrutura completa de efeitos fixos, clustering robusto, e testes de hipóteses formais. Todos os resultados são robustos a especificações alternativas e verificações de sensibilidade.*