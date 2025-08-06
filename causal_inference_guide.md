# Robust Causal Inference Models for Lobbying Effects on MEP Questions

## Research Question
**What is the causal effect of lobbying meetings on MEP question-asking behavior in specific policy areas?**

## Key Identification Challenges

### 1. **Endogeneity**
- **Problem**: MEPs who care about certain topics may attract more lobbying
- **Solution**: Fixed effects, instrumental variables, lagged treatment

### 2. **Selection Bias**
- **Problem**: Lobbyists may target MEPs based on unobservable characteristics
- **Solution**: Propensity score matching, difference-in-differences

### 3. **Reverse Causality**
- **Problem**: MEPs who ask questions may attract more lobbying
- **Solution**: Lagged treatment models, instrumental variables

### 4. **Omitted Variable Bias**
- **Problem**: Unobserved factors affecting both lobbying and questions
- **Solution**: Fixed effects, comprehensive control variables

## Model Specifications

### Model 1: Basic Fixed Effects (Baseline)
```
ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + X_it'δ + ε_it
```

**Advantages:**
- Controls for time-invariant MEP characteristics (α_i)
- Controls for time-specific shocks (γ_t)
- Standard approach in panel data

**Limitations:**
- Assumes lobbying is exogenous conditional on fixed effects
- May not address reverse causality

### Model 2: Lagged Treatment Model
```
ln(Questions_it) = α_i + Σ_k β_k*ln(Lobbying_{i,t-k}) + X_it'δ + ε_it
```

**Advantages:**
- Tests for delayed effects
- Helps address reverse causality
- Identifies timing of effects

**Limitations:**
- Cannot use time fixed effects with lags
- May lose observations due to lagging

### Model 3: Propensity Score Matching
```
Step 1: P(Treatment_i = 1) = f(X_i)
Step 2: ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + X_it'δ + ε_it
```

**Advantages:**
- Addresses selection bias
- Creates comparable treatment and control groups
- Non-parametric approach

**Limitations:**
- Requires strong ignorability assumption
- May reduce sample size significantly
- Sensitive to matching algorithm

### Model 4: Heterogeneous Treatment Effects
```
ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_it) + Σ_k θ_k*(ln(Lobbying_it) × Z_k,it) + X_it'δ + ε_it
```

**Advantages:**
- Allows for different effects across groups
- Tests for effect heterogeneity
- Provides policy-relevant insights

**Limitations:**
- Increases model complexity
- May reduce statistical power

### Model 5: Instrumental Variables
```
First Stage: ln(Lobbying_it) = π_0 + π_1*Z_it + X_it'π_2 + ν_it
Second Stage: ln(Questions_it) = α_i + γ_t + β*ln(Lobbying_hat_it) + X_it'δ + ε_it
```

**Advantages:**
- Addresses endogeneity directly
- Provides causal identification
- Robust to omitted variables

**Limitations:**
- Requires valid instrument
- May have weak instrument problems
- Complex implementation

### Model 6: Difference-in-Differences
```
ln(Questions_it) = α_i + γ_t + β*Treatment_i × Post_t + X_it'δ + ε_it
```

**Advantages:**
- Exploits natural experiments
- Controls for time trends
- Intuitive interpretation

**Limitations:**
- Requires treatment timing variation
- Parallel trends assumption
- May not be applicable in all contexts

### Model 7: Regression Discontinuity
```
ln(Questions_it) = α_i + γ_t + β*Treatment_it + f(Running_it) + X_it'δ + ε_it
```

**Advantages:**
- Exploits discontinuities in treatment assignment
- Strong identification
- Local average treatment effect

**Limitations:**
- Requires clear discontinuity
- Local effects only
- Sensitive to bandwidth choice

## Implementation Strategy

### Step 1: Data Preparation
```python
# Load and merge data
df_meetings = pd.read_csv("panel_data_meetings.csv")
df_questions = pd.read_csv("panel_data_questions.csv")
df_meps = pd.read_csv("panel_data_meps.csv")

# Create panel structure
df = df_meps.join(df_questions).join(df_meetings)
df = df.set_index(['member_id', 'time'])

# Create log transformations
df['log_questions'] = np.log(df['questions'] + 1)
df['log_lobbying'] = np.log(df['lobbying'] + 1)
```

### Step 2: Control Variables Selection
```python
control_vars = [
    'meps_DELEGATION_PARLIAMENTARY - MEMBER',
    'meetings_l_budget_cat_middle',
    'meetings_l_budget_cat_upper',
    'meetings_l_category_Business',
    'meetings_l_category_NGOs',
    'questions_infered_topic_economics and trade',
    'questions_infered_topic_environment and climate'
]
```

### Step 3: Model Estimation
```python
# Basic Fixed Effects
model1 = PanelOLS(
    dependent=df['log_questions_agriculture'],
    exog=df[['log_meetings_agriculture'] + control_vars],
    entity_effects=True,
    time_effects=True
)
results1 = model1.fit()

# Lagged Treatment
df['lobbying_lag1'] = df.groupby(level=0)['log_meetings_agriculture'].shift(1)
model2 = PanelOLS(
    dependent=df['log_questions_agriculture'],
    exog=df[['lobbying_lag1'] + control_vars],
    entity_effects=True,
    time_effects=False
)
results2 = model2.fit()
```

### Step 4: Robustness Checks
```python
# Different functional forms
# Different time periods
# Different samples
# Different control variables
```

## Interpretation Guidelines

### Elasticity Interpretation
- **Coefficient**: Direct elasticity of questions with respect to lobbying
- **Example**: β = 0.025 means 1% increase in lobbying → 0.025% increase in questions
- **Economic significance**: Consider magnitude relative to standard deviation

### Statistical Significance
- **P-values**: Standard significance levels (0.01, 0.05, 0.10)
- **Confidence intervals**: Report 95% confidence intervals
- **Robust standard errors**: Use clustered standard errors when appropriate

### Model Comparison
- **R-squared**: Compare model fit across specifications
- **Sample size**: Consider trade-off between sample size and model complexity
- **Consistency**: Check if results are consistent across different approaches

## Literature Review

### Key Papers on Lobbying Effects
1. **Hall and Deardorff (2006)**: "Lobbying as Legislative Subsidy"
2. **Baumgartner et al. (2009)**: "Lobbying and Policy Change"
3. **Drutman (2015)**: "The Business of America is Lobbying"

### Causal Inference Methods in Political Science
1. **Angrist and Pischke (2008)**: "Mostly Harmless Econometrics"
2. **Imbens and Rubin (2015)**: "Causal Inference in Statistics"
3. **Cunningham (2021)**: "Causal Inference: The Mixtape"

### Panel Data Methods
1. **Wooldridge (2010)**: "Econometric Analysis of Cross Section and Panel Data"
2. **Hsiao (2014)**: "Analysis of Panel Data"
3. **Baltagi (2021)**: "Econometric Analysis of Panel Data"

## Best Practices

### 1. **Transparency**
- Report all model specifications
- Document data sources and cleaning procedures
- Provide code for reproducibility

### 2. **Robustness**
- Test multiple functional forms
- Use different time periods
- Vary control variable sets

### 3. **Interpretation**
- Focus on economic significance, not just statistical significance
- Consider policy relevance
- Acknowledge limitations

### 4. **Validation**
- Check for multicollinearity
- Test for heteroscedasticity
- Validate instrumental variables

## Common Pitfalls

### 1. **Over-controlling**
- Including variables that are outcomes of treatment
- Controlling for post-treatment variables

### 2. **Weak Instruments**
- Using instruments with low first-stage F-statistic
- Not testing instrument validity

### 3. **Selection Bias**
- Ignoring non-random treatment assignment
- Not checking balance in matching

### 4. **Functional Form**
- Assuming linear relationships without testing
- Not considering non-linear effects

## Policy Implications

### 1. **Lobbying Effectiveness**
- Quantify the impact of lobbying on legislative behavior
- Identify most effective lobbying strategies

### 2. **Targeting Strategy**
- Focus on responsive MEPs
- Consider timing of lobbying efforts

### 3. **Regulation**
- Inform lobbying disclosure requirements
- Guide transparency policies

### 4. **Democracy**
- Understand influence in democratic processes
- Assess representation and accountability

## Future Research Directions

### 1. **Better Identification**
- Natural experiments in lobbying regulation
- Instrumental variables based on institutional features
- Regression discontinuity designs

### 2. **Mechanisms**
- How does lobbying affect MEP behavior?
- What are the channels of influence?
- How do different types of lobbying differ?

### 3. **Long-term Effects**
- Persistence of lobbying effects
- Cumulative impact over time
- Dynamic effects

### 4. **Heterogeneity**
- Effects across different policy areas
- Variation by MEP characteristics
- Differences by lobbyist type

## Conclusion

Robust causal inference of lobbying effects requires multiple identification strategies and careful attention to potential biases. The models presented here provide a comprehensive framework for estimating causal effects while addressing key identification challenges. The choice of model depends on the specific research question, data availability, and institutional context.

Key recommendations:
1. **Start with fixed effects** as baseline
2. **Add lagged treatment** to address reverse causality
3. **Use propensity score matching** for selection bias
4. **Consider instrumental variables** if valid instruments exist
5. **Test for heterogeneous effects** across different groups
6. **Conduct extensive robustness checks**
7. **Interpret results carefully** with attention to limitations

The goal is to provide credible evidence on the causal effects of lobbying while being transparent about the assumptions and limitations of each approach. 