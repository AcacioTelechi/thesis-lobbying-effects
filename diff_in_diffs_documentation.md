# Diff-in-Diffs Implementation for Lobbying Effects Analysis

## Overview

The diff-in-diffs (DiD) method has been implemented in the `LobbyingEffectsModel` class to provide a robust causal inference approach for analyzing lobbying effects on parliamentary questions. This implementation includes both **uniform timing** and **staggered timing** approaches, following the standard DiD framework while adapting it to the specific context of lobbying intensity and parliamentary activity.

## Method Implementation

### Model Specifications

#### 1. Uniform Timing DiD
```
ln(Questions_it) = α_i + γ_t + β*Treatment_i × Post_t + X_it'δ + ε_it
```

#### 2. Staggered Timing DiD (Event Study)
```
ln(Questions_it) = α_i + γ_t + Σ_k β_k*Treatment_i × Post_k,t + X_it'δ + ε_it
```

Where:
- `ln(Questions_it)`: Log of questions asked by MEP i at time t
- `α_i`: Individual fixed effects (MEP-specific)
- `γ_t`: Time fixed effects (period-specific)
- `Treatment_i`: Binary indicator for high lobbying intensity MEPs
- `Post_t`: Binary indicator for post-treatment period (uniform timing)
- `Post_k,t`: Binary indicator for k periods after treatment (staggered timing)
- `β`: DiD coefficient (treatment effect)
- `β_k`: Event study coefficients
- `X_it`: Control variables
- `ε_it`: Error term

### Treatment Assignment

#### Uniform Timing Approach
1. **Treatment Group**: MEPs with lobbying intensity above the threshold
2. **Control Group**: MEPs with lobbying intensity at or below the threshold
3. **Treatment Period**: Middle of the time series (median date)

#### Staggered Timing Approach
1. **Treatment Group**: MEPs who first receive high lobbying intensity
2. **Control Group**: MEPs who never receive high lobbying intensity
3. **Treatment Period**: Individual-specific first treatment date
4. **Event Study**: Tracks effects from 3 periods before to 3 periods after treatment

### Key Features

#### 1. Flexible Treatment Thresholds
- **Median**: 50th percentile of lobbying intensity
- **Mean**: Average lobbying intensity
- **75th Percentile**: 75th percentile of lobbying intensity
- **Custom**: User-defined numeric threshold

#### 2. Automatic Treatment Assignment
- Uses specified threshold for treatment assignment
- Creates binary treatment indicators
- Handles both uniform and staggered timing

#### 3. Event Study Analysis (Staggered Only)
- Tracks effects from 3 periods before to 3 periods after treatment
- Provides detailed coefficient for each period
- Enables dynamic effect analysis

#### 4. Parallel Trends Testing
- Tests the parallel trends assumption using pre-treatment data
- Creates treatment × time trend interaction
- Reports p-value for parallel trends test

#### 5. Comprehensive Results
- Manual DiD calculation for verification
- Regression-based DiD with controls
- Treatment group statistics (pre/post)
- Parallel trends test results
- Event study coefficients (staggered)

## Usage

### Basic Usage

```python
from lobbying_effects_model import DataBase, LobbyingEffectsModel

# Load data
database = DataBase()
df_filtered, column_sets = database.prepare_data()

# Initialize model
model = LobbyingEffectsModel(df_filtered, column_sets)
model.set_topic("agriculture")

# Run uniform timing diff-in-diffs
did_results = model.model_diff_in_diffs("uniform", "median")

# Run staggered timing diff-in-diffs
staggered_results = model.model_diff_in_diffs("staggered", "median")
```

### Advanced Usage

```python
# Compare uniform vs staggered approaches
comparison_results = model.run_diff_in_diffs_comparison("median")

# Run with different thresholds
results_median = model.model_diff_in_diffs("staggered", "median")
results_mean = model.model_diff_in_diffs("staggered", "mean")
results_75th = model.model_diff_in_diffs("staggered", "75th_percentile")

# Integration with full model suite
all_results = model.run_all_models(did_timing="staggered", treatment_threshold="median")
```

## Output and Interpretation

### Key Results

The methods return dictionaries containing:

#### Uniform Timing Results
- `elasticity`: DiD coefficient (β)
- `p_value`: Statistical significance
- `r_squared`: Model fit
- `n_obs`: Number of observations
- `manual_did`: Manual calculation for verification
- `treated_pre/post`: Treatment group means
- `control_pre/post`: Control group means
- `parallel_trends_p`: Parallel trends test p-value
- `treatment_threshold`: Threshold used for treatment assignment

#### Staggered Timing Results
- `elasticity`: Average treatment effect
- `p_value`: Statistical significance
- `r_squared`: Model fit
- `n_obs`: Number of observations
- `treated_meps`: Number of treated MEPs
- `total_meps`: Total number of MEPs
- `treatment_threshold`: Threshold used for treatment assignment
- `event_coefficients`: Dictionary of event study coefficients
- `parallel_trends_violated`: Boolean indicating parallel trends violation

### Interpretation Guidelines

#### 1. Statistical Significance
- **p < 0.05**: Statistically significant treatment effect
- **p ≥ 0.05**: No significant treatment effect

#### 2. Effect Direction
- **Positive coefficient**: High lobbying intensity MEPs ask more questions
- **Negative coefficient**: High lobbying intensity MEPs ask fewer questions

#### 3. Parallel Trends Assumption
- **p > 0.05**: Parallel trends assumption holds
- **p ≤ 0.05**: Parallel trends may be violated (interpret with caution)

#### 4. Event Study Interpretation (Staggered)
- **Pre-treatment coefficients**: Should be close to zero (parallel trends)
- **Post-treatment coefficients**: Show treatment effects over time
- **Dynamic effects**: Reveal immediate vs. delayed effects

## Methodological Considerations

### When to Use Each Approach

#### Uniform Timing DiD
- **Use when**: All MEPs receive treatment at the same time
- **Advantages**: Simpler interpretation, standard approach
- **Limitations**: May not reflect reality of staggered lobbying

#### Staggered Timing DiD
- **Use when**: MEPs receive treatment at different times
- **Advantages**: More realistic, richer analysis, event study
- **Limitations**: More complex, requires sufficient variation in timing

### Strengths

1. **Causal Inference**: Provides causal estimates under parallel trends
2. **Robustness**: Controls for time-invariant unobservables
3. **Flexibility**: Adapts to different topics and time periods
4. **Validation**: Includes parallel trends testing
5. **Realism**: Staggered approach reflects actual lobbying patterns

### Limitations

1. **Parallel Trends Assumption**: Requires parallel pre-treatment trends
2. **Treatment Timing**: Uniform approach uses arbitrary treatment date
3. **Binary Treatment**: Simplifies lobbying intensity to binary indicator
4. **No Anticipation**: Assumes no anticipation of treatment
5. **Heterogeneity**: May not capture heterogeneous treatment effects

### Assumptions

1. **Parallel Trends**: Treatment and control groups would follow parallel trends in absence of treatment
2. **No Anticipation**: MEPs don't anticipate the treatment
3. **Stable Unit Treatment Value**: Treatment effect is homogeneous
4. **No Spillovers**: Treatment doesn't affect control group

## Example Output

### Uniform Timing DiD
```
=== Model 6: Diff-in-Diff (agriculture) ===
Treatment threshold (median): 0.6931
Treatment group (pre): 2.1456
Treatment group (post): 2.2345
Control group (pre): 1.9876
Control group (post): 2.0123
Manual DiD estimate: 0.0890
Regression DiD coefficient: 0.0876
DiD P-value: 0.0234
R-squared: 0.4567
N observations: 12500
Parallel trends test p-value: 0.1234
✓ Parallel trends assumption appears to hold (p > 0.05)
```

### Staggered Timing DiD
```
=== Model 6b: Staggered Diff-in-Diff (agriculture) ===
Treatment threshold (median): 0.6931
Treated MEPs: 45 out of 100 (45.0%)

Staggered DiD coefficient: 0.0923
DiD P-value: 0.0187
R-squared: 0.4789
N observations: 12500

Event Study Coefficients:
  Period -3:   0.0123
  Period -2:  -0.0045
  Period -1:   0.0089
  Period  0:   0.0456 *
  Period  1:   0.0789 **
  Period  2:   0.0923 ***
  Period  3:   0.0876 ***

Parallel Trends Test:
✓ Pre-treatment coefficients are not significant - parallel trends assumption holds
```

## Integration with Other Methods

The diff-in-diffs methods complement other approaches in the model suite:

1. **Fixed Effects**: Provides baseline comparison
2. **Propensity Score Matching**: Alternative approach to selection bias
3. **Instrumental Variables**: Addresses different endogeneity concerns
4. **Heterogeneous Effects**: Explores treatment effect heterogeneity

## Best Practices

### 1. Data Requirements
- Sufficient pre-treatment periods for parallel trends testing
- Balanced panel data
- Clear treatment timing (especially for staggered approach)
- Adequate variation in treatment timing (staggered approach)

### 2. Model Specification
- Include relevant control variables
- Use fixed effects appropriately
- Test parallel trends assumption
- Consider treatment threshold sensitivity

### 3. Interpretation
- Consider economic significance alongside statistical significance
- Check parallel trends assumption
- Validate with manual calculations
- Examine event study patterns (staggered approach)

### 4. Robustness Checks
- Alternative treatment thresholds
- Different time periods
- Various control variable sets
- Sensitivity to treatment timing definition

## References

- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1), 249-275.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.
- Angrist, J. D., & Pischke, J. S. (2008). *Mostly harmless econometrics: An empiricist's companion*. Princeton University Press.
- Hall, R. L., & Deardorff, A. V. (2006). Lobbying as legislative subsidy. *American Political Science Review*, 100(1), 69-84.
- Baumgartner, F. R., Berry, J. M., Hojnacki, M., Kimball, D. C., & Leech, B. L. (2009). *Lobbying and policy change: Who wins, who loses, and why*. University of Chicago Press.