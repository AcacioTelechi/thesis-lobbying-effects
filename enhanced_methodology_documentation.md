# Enhanced Lobbying Effects Methodology Documentation

## Overview

This document provides comprehensive documentation for the enhanced lobbying effects model (`lobbying_effects_model_v2.py`) that addresses critical methodological concerns raised in the review. The enhanced model implements state-of-the-art econometric methods to provide robust causal inference.

## Critical Concerns Addressed

### 1. Methodological Identification Assumptions

#### 1.1 Two-Way Fixed-Effects (TWFE) and Negative Weights

**Problem**: Traditional staggered DiD with TWFE can yield biased estimates when treatment timing varies and effects are heterogeneous, due to negative weights in the Goodman-Bacon decomposition.

**Solution**: `RobustDiDEstimator` class implements:
- **Goodman-Bacon Decomposition**: Identifies and quantifies negative weights
- **Callaway & Sant'Anna (2021) Estimator**: Provides robust ATT estimates
- **Sun & Abraham (2021) Approach**: Handles heterogeneous treatment effects

```python
# Example usage
robust_did = RobustDiDEstimator(df, treatment_col, outcome_col, entity_col, time_col)

# Goodman-Bacon decomposition
bacon_results = robust_did.goodman_bacon_decomposition()
print(f"Negative weight share: {bacon_results['negative_weight_share']:.3f}")

# Callaway & Sant'Anna estimator
cs_results = robust_did.callaway_santanna_estimator()
print(f"Overall ATT: {cs_results['overall_att']:.4f}")
```

#### 1.2 Enhanced Parallel Trends Testing

**Problem**: Limited statistical power with only three pre-periods and reliance on significance tests.

**Solution**: `EnhancedParallelTrends` class provides:
- **Placebo Tests**: 100+ placebo tests at arbitrary dates
- **Visual Trend Analysis**: Group-specific trend plots
- **Multiple Testing Approaches**: Beyond simple significance tests

```python
# Enhanced parallel trends testing
parallel_trends = EnhancedParallelTrends(df, treatment_col, outcome_col, entity_col, time_col)

# Placebo tests
placebo_results = parallel_trends.placebo_tests(n_placebos=100)
print(f"Placebo test p-value: {placebo_results['p_value']:.4f}")

# Visual trend analysis
trends_data = parallel_trends.plot_group_trends()
```

#### 1.3 Dynamic Treatment Effects and Functional Form

**Problem**: Limited post-period analysis and log-linear assumptions may not fit zero-inflated count data.

**Solution**: `ContinuousTreatmentEffects` class implements:
- **Dose-Response Functions**: Continuous treatment effects
- **Generalized Propensity Scores**: For continuous treatments
- **Non-linear Specifications**: Beyond log-linear models

```python
# Continuous treatment effects
continuous_treatment = ContinuousTreatmentEffects(df, treatment_col, outcome_col, entity_col, time_col)

# Dose-response function
dose_response = continuous_treatment.dose_response_function()
print(f"Linear effect: {dose_response['linear_effect']:.4f}")
print(f"Quadratic effect: {dose_response['quadratic_effect']:.4f}")

# Generalized propensity score
gps_results = continuous_treatment.generalized_propensity_score(covariates)
```

### 2. Measurement and Data Quality

#### 2.1 Lobbying Intensity Thresholds

**Problem**: Arbitrary percentile thresholds may be sensitive to outliers and obscure continuous relationships.

**Solution**: 
- **Continuous Treatment Effects**: Dose-response functions instead of binary thresholds
- **Multiple Threshold Sensitivity**: Test robustness across different thresholds
- **Generalized Propensity Scores**: Handle continuous treatment properly

#### 2.2 Classification Errors in Topics

**Problem**: Self-declared interests and thresholded inference may introduce classification errors.

**Solution**: `TopicClassificationValidator` class provides:
- **Classification Accuracy Assessment**: Compare with manual coding
- **Threshold Sensitivity Analysis**: Test robustness to threshold choice
- **Validation Metrics**: Precision, recall, F1-score

```python
# Topic classification validation
topic_validator = TopicClassificationValidator(df, topic_cols, manual_coding)

# Classification accuracy
accuracy_results = topic_validator.classification_accuracy(sample_size=100)
print(f"Overall accuracy: {accuracy_results['overall_accuracy']:.3f}")

# Threshold sensitivity
sensitivity_results = topic_validator.threshold_sensitivity()
```

#### 2.3 Network Measures as Covariates

**Problem**: Network centrality may be endogenous and post-treatment.

**Solution**: 
- **Pre-treatment Network Measures**: Restrict to pre-treatment periods
- **Instrumental Variables**: Use exogenous shocks for network measures
- **Network-Aware DiD**: Explicit modeling of network effects

### 3. Confounding and Endogeneity

#### 3.1 Reverse Causality

**Problem**: MEPs' proactive behavior may attract more lobbying.

**Solution**: `InstrumentalVariablesEnhanced` class implements:
- **Natural Experiments**: Transparency rule changes
- **Committee Reassignments**: Exogenous institutional changes
- **Multiple Instruments**: Robustness across different instruments

```python
# Enhanced IV analysis
iv_enhanced = InstrumentalVariablesEnhanced(df, treatment_col, outcome_col, entity_col, time_col)

# Natural experiment IV
natural_iv = iv_enhanced.natural_experiment_iv('transparency_rule_change')

# Committee reassignment IV
committee_iv = iv_enhanced.committee_reassignment_iv('committee_membership', reassignment_dates)
```

#### 3.2 Time-Varying Confounders

**Problem**: Shocks like COVID-19 may affect both lobbying and questions.

**Solution**:
- **Topic-Specific Time Trends**: Control for topic-specific shocks
- **Interaction Terms**: COVID dummy × policy area
- **Robustness Checks**: Different time periods

#### 3.3 Spillover and SUTVA Violations

**Problem**: Cross-topic spillovers may violate SUTVA.

**Solution**: `CrossTopicSpilloverModel` class implements:
- **Spatial DiD**: Model cross-topic spillovers
- **Network-Aware DiD**: Capture lobbying network effects
- **Spillover Estimation**: Direct vs. spillover effects

```python
# Cross-topic spillover modeling
spillover_model = CrossTopicSpilloverModel(df, topic_cols, outcome_col, entity_col, time_col)

# Spatial DiD
spatial_did = spillover_model.spatial_did_model()

# Network-aware DiD
network_did = spillover_model.network_aware_did()
```

### 4. Inferential Scope and External Validity

#### 4.1 Outcome Proxy

**Problem**: Question counts may not capture policy influence.

**Solution**: 
- **Multiple Outcomes**: Voting records, amendments, reports
- **Outcome Validation**: Correlation with policy outcomes
- **Robustness Checks**: Different outcome measures

#### 4.2 EU-Specific Context

**Problem**: Results may not generalize to other legislatures.

**Solution**:
- **Contextual Analysis**: Explicit discussion of EU-specific factors
- **Comparative Framework**: Suggest replication in other contexts
- **Institutional Details**: Account for EU-specific institutions

#### 4.3 Time Frame and Panel Balance

**Problem**: Single legislative term may limit inference.

**Solution**:
- **Extended Time Horizon**: Include prior terms where possible
- **Sub-period Analysis**: Test consistency across periods
- **Persistence Analysis**: Long-term effects

## Usage Guide

### Basic Usage

```python
from src.lobbying_effects_model_v2 import EnhancedLobbyingEffectsModel
from src.lobbying_effects_model import DataBase

# Load data
database = DataBase()
df_filtered, column_sets = database.prepare_data()

# Initialize enhanced model
model = EnhancedLobbyingEffectsModel(df_filtered, column_sets)
model.set_topic("agriculture")

# Run comprehensive enhanced analysis
results = model.run_enhanced_analysis()

# Generate summary report
model.create_enhanced_summary_report(results)
```

### Advanced Usage

```python
# Run individual components
robust_did = model.robust_did.goodman_bacon_decomposition()
parallel_trends = model.parallel_trends.placebo_tests()
continuous_treatment = model.continuous_treatment.dose_response_function()
topic_validation = model.topic_validator.classification_accuracy()
spillovers = model.spillover_model.spatial_did_model()
iv_analysis = model.iv_enhanced.natural_experiment_iv('instrument')
```

### Output Interpretation

#### Robust DiD Results
- **Negative Weight Share**: >0.1 suggests potential bias
- **Callaway-Sant'Anna ATT**: Robust treatment effect estimate
- **Timing Heterogeneity**: ATT by treatment timing

#### Parallel Trends Results
- **Placebo Test P-value**: <0.05 suggests violation
- **Visual Trends**: Check for pre-treatment divergence
- **Multiple Tests**: Robustness across different approaches

#### Continuous Treatment Results
- **Linear Effect**: Average treatment effect
- **Quadratic Effect**: Non-linear relationship
- **Dose-Response**: Treatment intensity effects

#### Topic Validation Results
- **Classification Accuracy**: >0.8 suggests good classification
- **Threshold Sensitivity**: Robustness to threshold choice
- **Topic-Specific Accuracy**: Performance by policy area

#### Spillover Results
- **Direct Effects**: Within-topic effects
- **Spillover Effects**: Cross-topic effects
- **Total Effects**: Combined direct and spillover

#### IV Results
- **First Stage F-stat**: >10 suggests strong instruments
- **IV Coefficient**: Instrumental variables estimate
- **Weak Instruments**: Check for weak instrument bias

## Methodological Advantages

### 1. Robustness
- **Multiple Approaches**: Different estimators for same effect
- **Sensitivity Analysis**: Robustness to assumptions
- **Validation**: Cross-validation across methods

### 2. Transparency
- **Explicit Assumptions**: Clear statement of assumptions
- **Validation Tests**: Formal tests of key assumptions
- **Documentation**: Comprehensive methodological documentation

### 3. Realism
- **Continuous Treatment**: More realistic than binary
- **Cross-Topic Spillovers**: Captures policy interdependence
- **Network Effects**: Accounts for lobbying networks

### 4. Identification
- **Natural Experiments**: Exogenous variation
- **Multiple Instruments**: Robustness across instruments
- **Placebo Tests**: Validation of identification strategy

## Limitations and Future Directions

### Current Limitations
1. **Computational Intensity**: Enhanced methods require more computation
2. **Data Requirements**: Some methods require additional data
3. **Complexity**: More complex interpretation required

### Future Improvements
1. **Machine Learning**: ML-based classification and prediction
2. **Bayesian Methods**: Bayesian inference for uncertainty
3. **Dynamic Models**: Explicit dynamic modeling
4. **Policy Outcomes**: Direct policy outcome analysis

## References

1. Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

2. Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.

3. Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

4. Imbens, G. W., & Rubin, D. B. (2015). *Causal inference in statistics, social, and biomedical sciences*. Cambridge University Press.

5. Angrist, J. D., & Pischke, J. S. (2008). *Mostly harmless econometrics: An empiricist's companion*. Princeton University Press.

## Conclusion

The enhanced lobbying effects model addresses critical methodological concerns through:

1. **Robust DiD estimators** that handle heterogeneous treatment effects
2. **Enhanced parallel trends testing** with multiple approaches
3. **Continuous treatment effects** beyond binary thresholds
4. **Topic classification validation** with accuracy assessment
5. **Cross-topic spillover modeling** to capture policy interdependence
6. **Enhanced instrumental variables** with multiple instruments

This comprehensive approach provides more robust and realistic causal inference while maintaining transparency about assumptions and limitations. 