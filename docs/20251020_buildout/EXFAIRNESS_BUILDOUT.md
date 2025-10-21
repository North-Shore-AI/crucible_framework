# ExFairness Framework - Complete Implementation Buildout
## Date: 2025-10-20

---

## Executive Summary

**Mission**: Build ExFairness, a comprehensive fairness and bias detection library for Elixir AI/ML systems, providing rigorous fairness metrics, bias detection algorithms, and mitigation techniques to ensure equitable predictions across different demographic groups.

**Core Value Proposition**: The definitive fairness library for the Elixir ML ecosystem, bridging the gap between academic fairness research and production ML systems with mathematical rigor, transparency, and actionable mitigation strategies.

**Technical Foundation**: Tensor-first design using Nx for high-performance computations, integrating seamlessly with Axon, Scholar, and the broader North Shore AI ecosystem.

---

## Project Context

### Current State
- Project structure established with mix.exs
- Documentation framework in place (README.md, architecture.md, metrics.md, algorithms.md, roadmap.md)
- Skeleton main module: `/home/home/p/g/n/North-Shore-AI/ExFairness/lib/ex_fairness.ex`
- Dependencies: Nx (~> 0.7), ex_doc (~> 0.31)
- Target: Elixir ~> 1.14, OTP 25+

### Vision
ExFairness aims to be the standard fairness and bias detection library for Elixir ML, providing:
1. **Mathematical Rigor**: All metrics based on established fairness research
2. **Transparency**: Clear explanations of fairness definitions and trade-offs
3. **Actionability**: Concrete mitigation strategies with implementation guidance
4. **Flexibility**: Support for multiple fairness definitions and use cases
5. **Integration**: Seamless integration with Nx, Axon, Scholar, and other Elixir ML tools

---

## Complete System Architecture

### Module Structure

```
lib/ex_fairness/
├── ex_fairness.ex                    # Main API module
├── metrics/
│   ├── demographic_parity.ex         # Demographic parity metrics
│   ├── equalized_odds.ex             # Equalized odds metrics
│   ├── equal_opportunity.ex          # Equal opportunity metrics
│   ├── predictive_parity.ex          # Predictive parity metrics
│   ├── calibration.ex                # Calibration metrics
│   ├── individual_fairness.ex        # Individual fairness metrics
│   └── counterfactual.ex             # Counterfactual fairness
├── detection/
│   ├── disparate_impact.ex           # Disparate impact analysis
│   ├── statistical_parity.ex         # Statistical parity testing
│   ├── intersectional.ex             # Intersectional analysis
│   ├── temporal_drift.ex             # Temporal bias monitoring
│   ├── label_bias.ex                 # Label bias detection
│   └── representation.ex             # Representation bias
├── mitigation/
│   ├── reweighting.ex                # Reweighting techniques
│   ├── resampling.ex                 # Resampling techniques
│   ├── threshold_optimization.ex     # Threshold optimization
│   ├── adversarial_debiasing.ex      # Adversarial debiasing
│   ├── fair_representation.ex        # Fair representation learning
│   └── calibration.ex                # Calibration techniques
├── report.ex                         # Fairness reporting
└── utils.ex                          # Utility functions

test/ex_fairness/
├── metrics/
│   ├── demographic_parity_test.exs
│   ├── equalized_odds_test.exs
│   ├── equal_opportunity_test.exs
│   ├── predictive_parity_test.exs
│   ├── calibration_test.exs
│   ├── individual_fairness_test.exs
│   └── counterfactual_test.exs
├── detection/
│   ├── disparate_impact_test.exs
│   ├── statistical_parity_test.exs
│   ├── intersectional_test.exs
│   ├── temporal_drift_test.exs
│   ├── label_bias_test.exs
│   └── representation_test.exs
├── mitigation/
│   ├── reweighting_test.exs
│   ├── resampling_test.exs
│   ├── threshold_optimization_test.exs
│   ├── adversarial_debiasing_test.exs
│   ├── fair_representation_test.exs
│   └── calibration_test.exs
├── report_test.exs
├── utils_test.exs
└── test_helper.exs
```

---

## Fairness Metrics - Complete Specifications

### 1. Demographic Parity (Statistical Parity)

**Mathematical Definition**:
```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

Disparity Measure:
Δ_DP = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|
```

**When to Use**:
- Equal representation in positive outcomes required
- Advertising, content recommendation
- When base rates can differ between groups

**Implementation Requirements**:
- Compute positive prediction rates per group
- Calculate absolute disparity
- Statistical significance testing (Z-test)
- Bootstrap confidence intervals (1000 samples default)
- Threshold validation (default: 0.1)

**API**:
```elixir
result = ExFairness.demographic_parity(predictions, sensitive_attr, opts)
# => %{
#   group_a_rate: 0.50,
#   group_b_rate: 0.75,
#   disparity: 0.25,
#   passes: false,
#   threshold: 0.10,
#   confidence_interval: {0.18, 0.32},
#   p_value: 0.001
# }
```

### 2. Equalized Odds

**Mathematical Definition**:
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)  # Equal TPR
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)  # Equal FPR

Disparity Measures:
Δ_TPR = |TPR_{A=0} - TPR_{A=1}|
Δ_FPR = |FPR_{A=0} - FPR_{A=1}|
```

**When to Use**:
- Both false positives and false negatives matter
- Criminal justice (wrongful conviction + wrongful acquittal)
- Medical diagnosis

**Implementation Requirements**:
- Compute confusion matrices per group
- Calculate TPR and FPR for each group
- Test both TPR and FPR disparities
- Confidence intervals for both metrics
- Combined pass/fail based on threshold

**API**:
```elixir
result = ExFairness.equalized_odds(predictions, labels, sensitive_attr, opts)
# => %{
#   group_a_tpr: 0.67,
#   group_b_tpr: 0.67,
#   group_a_fpr: 0.50,
#   group_b_fpr: 0.00,
#   tpr_disparity: 0.00,
#   fpr_disparity: 0.50,
#   passes: false,
#   tpr_ci: {-0.05, 0.05},
#   fpr_ci: {0.35, 0.65}
# }
```

### 3. Equal Opportunity

**Mathematical Definition**:
```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)

Disparity Measure:
Δ_EO = |TPR_{A=0} - TPR_{A=1}|
```

**When to Use**:
- Cost of false negatives varies by group
- Hiring (missing qualified candidates)
- College admissions

**Implementation Requirements**:
- Compute TPR (recall) per group
- Calculate disparity
- Statistical testing
- Interpretation utilities

**API**:
```elixir
result = ExFairness.equal_opportunity(predictions, labels, sensitive_attr, opts)
# => %{
#   group_a_tpr: 0.67,
#   group_b_tpr: 0.67,
#   disparity: 0.00,
#   passes: true,
#   confidence_interval: {-0.08, 0.08},
#   interpretation: "Model provides equal opportunity across groups"
# }
```

### 4. Predictive Parity (Outcome Test)

**Mathematical Definition**:
```
P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)

Disparity Measure:
Δ_PP = |PPV_{A=0} - PPV_{A=1}|
```

**When to Use**:
- Meaning of positive prediction should be consistent
- Risk assessment tools
- Credit scoring

**Implementation Requirements**:
- Compute PPV (precision) per group
- Calculate disparity
- Handle cases with no positive predictions
- Confidence intervals

### 5. Calibration

**Mathematical Definition**:
```
P(Y = 1 | S(X) = s, A = 0) = P(Y = 1 | S(X) = s, A = 1)

For each bin b:
Δ_Cal(b) = |P(Y = 1 | S(X) ∈ bin_b, A = 0) - P(Y = 1 | S(X) ∈ bin_b, A = 1)|
```

**When to Use**:
- Probability estimates must be interpretable
- Medical risk prediction
- Any application where probabilities guide decisions

**Implementation Requirements**:
- Bin probabilities (default: 10 bins)
- Compute actual outcome rate per bin per group
- Calculate max disparity across bins
- Generate calibration curves
- Expected calibration error (ECE)

### 6. Individual Fairness (Lipschitz Continuity)

**Mathematical Definition**:
```
d(Ŷ(x₁), Ŷ(x₂)) ≤ L · d(x₁, x₂)

Measurement:
Fairness = (1/|P|) Σ_{(i,j) ∈ P} 𝟙[|f(xᵢ) - f(xⱼ)| ≤ ε]
```

**When to Use**:
- Individual treatment important
- Personalized recommendations
- Custom pricing

**Implementation Requirements**:
- Define similarity metric (Euclidean, cosine, custom)
- Find similar pairs (k-NN approach)
- Compute prediction consistency
- Lipschitz constant estimation

### 7. Counterfactual Fairness

**Mathematical Definition**:
```
P(Ŷ_{A←a}(U) = y | X = x, A = a) = P(Ŷ_{A←a'}(U) = y | X = x, A = a)
```

**When to Use**:
- Causal understanding important
- Legal compliance (disparate treatment)
- High-stakes decisions

**Implementation Requirements**:
- Causal graph specification
- Counterfactual generation
- Intervention operations
- Comparison of actual vs counterfactual predictions

---

## Bias Detection Algorithms

### 1. Statistical Parity Testing

**Purpose**: Detect violations of demographic parity using hypothesis tests

**Algorithm**:
```
1. Compute observed rates:
   rate_A = mean(predictions[sensitive_attr == 0])
   rate_B = mean(predictions[sensitive_attr == 1])

2. Under null hypothesis (no disparity):
   SE = sqrt(p(1-p)(1/n_A + 1/n_B))
   where p = (n_A * rate_A + n_B * rate_B) / (n_A + n_B)

3. Test statistic:
   z = (rate_A - rate_B) / SE

4. P-value:
   p_value = 2 * P(Z > |z|)  # Two-tailed test

5. Decision:
   reject_null = p_value < alpha
```

**Implementation Requirements**:
- Z-test implementation
- Chi-square test as alternative
- Permutation test for small samples
- Multiple testing correction (Bonferroni, FDR)
- Effect size calculation (Cohen's h)

### 2. Disparate Impact Analysis (80% Rule)

**Purpose**: Legal compliance testing (EEOC 4/5ths rule)

**Algorithm**:
```
Ratio = P(Ŷ = 1 | A = 1) / P(Ŷ = 1 | A = 0)

Passes 80% rule: Ratio ≥ 0.8
```

**Implementation Requirements**:
- Compute selection rates per group
- Calculate ratio
- Interpretation (legal context)
- Confidence interval for ratio

### 3. Intersectional Bias Detection

**Purpose**: Identify bias in combinations of sensitive attributes

**Algorithm**:
```
1. Create all attribute combinations:
   groups = cartesian_product(unique(attr1), unique(attr2), ...)

2. For each group g:
   a. Filter data: data_g = data[matches_group(g)]
   b. Compute metric: metric_g = compute_metric(data_g)
   c. Store: bias_map[g] = metric_g

3. Find reference group (typically majority or best performing)

4. Compute disparities:
   For each group g:
     disparity_g = |metric_g - metric_ref|

5. Identify most disadvantaged:
   most_disadvantaged = argmax(disparity_g)
```

**Implementation Requirements**:
- Cartesian product of attribute values
- Metric computation per subgroup
- Minimum sample size requirements
- Visualization support (heatmaps)
- Statistical power analysis

### 4. Temporal Bias Drift Detection

**Purpose**: Monitor fairness metrics over time

**Algorithm (CUSUM)**:
```
1. Initialize:
   S_pos = 0, S_neg = 0
   baseline = mean(metric_values[initial_period])

2. For each time t:
   deviation = metric_values[t] - baseline

   S_pos = max(0, S_pos + deviation - allowance)
   S_neg = max(0, S_neg - deviation - allowance)

   if S_pos > threshold or S_neg > threshold:
     return drift_detected = true, change_point = t

3. Return drift_detected = false
```

**Implementation Requirements**:
- CUSUM control chart
- EWMA (Exponentially Weighted Moving Average)
- Baseline period configuration
- Alert levels (warning, critical)
- Change point detection

### 5. Label Bias Detection

**Purpose**: Identify bias in training labels

**Algorithm**:
```
1. For each sensitive group:
   a. Find similar feature vectors across groups
   b. Compute label discrepancy for similar pairs

2. Statistical test:
   H0: No label bias
   H1: Label bias exists

   Compare discrepancy to random baseline
```

**Implementation Requirements**:
- Similarity search (k-NN, approximate methods)
- Discrepancy calculation
- t-test or Mann-Whitney U test
- Minimum pairs requirement
- Feature importance for bias source identification

### 6. Representation Bias Detection

**Purpose**: Measure data imbalance across groups

**Algorithm**:
```
1. Compute group distributions in dataset
2. Compare to expected/population distributions
3. Calculate representation ratios
4. Statistical testing (chi-square goodness of fit)
```

**Implementation Requirements**:
- Group count calculation
- Expected distribution specification
- Chi-square test
- Visualization of group distributions

---

## Bias Mitigation Techniques

### 1. Reweighting (Pre-processing)

**Purpose**: Adjust training sample weights for fairness

**Algorithm**:
```
1. Compute group and label combinations:
   groups = {(A=0, Y=0), (A=0, Y=1), (A=1, Y=0), (A=1, Y=1)}

2. For each combination (a, y):
   P_a_y = P(A=a, Y=y)
   P_a = P(A=a)
   P_y = P(Y=y)

3. For demographic parity:
   w(a, y) = P_y / P_a_y

4. Normalize weights:
   weights = weights / mean(weights)
```

**Implementation Requirements**:
- Support for demographic parity weights
- Support for equalized odds weights
- Weight normalization
- Integration with training pipelines
- Validation of weight distributions

### 2. Resampling (Pre-processing)

**Purpose**: Balance dataset through sampling

**Strategies**:
- Oversampling minority groups
- Undersampling majority groups
- SMOTE (Synthetic Minority Oversampling)
- Combined strategies

**Implementation Requirements**:
- Random oversampling
- Random undersampling
- Stratified sampling
- SMOTE implementation for continuous features
- Validation split preservation

### 3. Threshold Optimization (Post-processing)

**Purpose**: Find group-specific decision thresholds

**Algorithm**:
```
1. Define objective:
   Maximize accuracy subject to fairness constraints

2. Grid search over threshold pairs:
   For t_A in [0, 1]:
     For t_B in [0, 1]:
       predictions_A = (probs_A >= t_A)
       predictions_B = (probs_B >= t_B)

       if satisfies_fairness_constraint:
         accuracy = compute_accuracy()
         if accuracy > best_accuracy:
           best = (t_A, t_B, accuracy)

3. Return best thresholds
```

**Implementation Requirements**:
- Grid search with configurable resolution
- Gradient-based optimization for continuous case
- Support for equalized odds, equal opportunity targets
- Pareto frontier computation (accuracy-fairness tradeoff)
- Group-specific threshold application

### 4. Adversarial Debiasing (In-processing)

**Purpose**: Train model with fairness constraints

**Architecture**:
```
Model:
- Predictor: f(X) → Ŷ
- Adversary: g(f(X)) → Â

Loss:
L = L_prediction(Ŷ, Y) - λ * L_adversary(Â, A)

Training:
Alternate:
  1. Update predictor (minimize L)
  2. Update adversary (maximize L_adversary)
```

**Implementation Requirements**:
- Axon integration for neural networks
- Predictor network definition
- Adversary network definition
- Alternating training loop
- Adversary strength parameter (lambda)
- Convergence criteria

### 5. Fair Representation Learning (Pre-processing)

**Purpose**: Learn representations independent of sensitive attributes

**Algorithm (Variational Fair Autoencoder)**:
```
Loss:
L = L_reconstruction + L_KL + λ * L_independence

where:
- L_reconstruction = -E[log p(X | Z)]
- L_KL = KL(q(Z|X) || p(Z))
- L_independence = MMD(Z[A=0], Z[A=1])
```

**Implementation Requirements**:
- Encoder-decoder architecture (Axon)
- VAE implementation
- Maximum Mean Discrepancy (MMD) calculation
- Kernel selection (RBF, polynomial)
- Latent dimension configuration
- Independence weight parameter

### 6. Calibration Techniques (Post-processing)

**Purpose**: Achieve calibrated predictions per group

**Methods**:
- Platt scaling (logistic calibration) per group
- Isotonic regression per group
- Beta calibration

**Implementation Requirements**:
- Platt scaling implementation
- Isotonic regression implementation
- Separate calibration per group
- Validation of calibration quality
- ECE computation

---

## Comprehensive Reporting System

### Fairness Report Generation

**Purpose**: Generate comprehensive fairness assessments

**Components**:
1. **Metric Aggregation**: Compute multiple metrics
2. **Statistical Testing**: Significance and confidence intervals
3. **Interpretation**: Plain language explanations
4. **Recommendations**: Actionable mitigation suggestions
5. **Trade-off Analysis**: Accuracy-fairness Pareto frontier

**API**:
```elixir
report = ExFairness.fairness_report(
  predictions,
  labels,
  sensitive_attr,
  metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity],
  include_ci: true,
  bootstrap_samples: 1000
)

# => %{
#   demographic_parity: %{passes: false, disparity: 0.25, ci: {0.18, 0.32}},
#   equalized_odds: %{passes: false, tpr_disparity: 0.00, fpr_disparity: 0.50},
#   equal_opportunity: %{passes: true, disparity: 0.00},
#   predictive_parity: %{passes: true, disparity: 0.05},
#   overall_assessment: "2 of 4 fairness metrics passed",
#   recommendations: [
#     "Consider threshold optimization for demographic parity",
#     "Investigate false positive rate disparity in equalized odds",
#     "Review data collection for potential representation bias"
#   ],
#   impossibility_warnings: [
#     "Note: Demographic parity and equalized odds may be incompatible due to base rate differences"
#   ]
# }
```

**Export Formats**:
- Markdown (human-readable)
- JSON (machine-readable)
- HTML (interactive reports)
- LaTeX (academic papers, documentation)

**Report Sections**:
1. Executive Summary
2. Metric Results Table
3. Statistical Significance
4. Visual Analysis (when applicable)
5. Interpretation
6. Mitigation Recommendations
7. Impossibility Theorem Warnings
8. Reproducibility Information

---

## Testing Strategy - TDD Requirements

### Unit Testing

**Requirements for Each Module**:
1. **Basic Functionality**: Core computation correctness
2. **Edge Cases**:
   - Empty tensors
   - Single group data
   - Perfect fairness (all rates equal)
   - Perfect unfairness (maximum disparity)
   - All positive/all negative predictions
3. **Numerical Stability**:
   - Very small sample sizes
   - Extreme probability values (near 0, near 1)
   - Division by zero handling
4. **Input Validation**:
   - Mismatched tensor shapes
   - Invalid sensitive attribute values
   - Non-binary predictions (when required)

**Example Test Structure**:
```elixir
defmodule ExFairness.Metrics.DemographicParityTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Metrics.DemographicParity

  describe "demographic_parity/3" do
    test "computes correct disparity for balanced case" do
      predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0])
      sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

      result = ExFairness.demographic_parity(predictions, sensitive_attr)

      assert result.group_a_rate == 0.5
      assert result.group_b_rate == 0.5
      assert result.disparity == 0.0
      assert result.passes == true
    end

    test "detects disparity correctly" do
      predictions = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 0])
      sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

      result = ExFairness.demographic_parity(predictions, sensitive_attr)

      assert result.group_a_rate == 0.75
      assert result.group_b_rate == 0.0
      assert result.disparity == 0.75
      assert result.passes == false
    end

    test "handles edge case: all same group" do
      predictions = Nx.tensor([1, 0, 1, 0])
      sensitive_attr = Nx.tensor([0, 0, 0, 0])

      assert_raise ExFairness.Error, ~r/single group/i, fn ->
        ExFairness.demographic_parity(predictions, sensitive_attr)
      end
    end

    test "handles edge case: empty tensors" do
      predictions = Nx.tensor([])
      sensitive_attr = Nx.tensor([])

      assert_raise ExFairness.Error, ~r/empty/i, fn ->
        ExFairness.demographic_parity(predictions, sensitive_attr)
      end
    end

    test "provides confidence intervals when requested" do
      predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1] |> List.duplicate(100) |> List.flatten())
      sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1] |> List.duplicate(100) |> List.flatten())

      result = ExFairness.demographic_parity(predictions, sensitive_attr,
        include_ci: true,
        bootstrap_samples: 1000
      )

      assert Map.has_key?(result, :confidence_interval)
      {lower, upper} = result.confidence_interval
      assert lower <= result.disparity
      assert upper >= result.disparity
    end
  end
end
```

### Property-Based Testing

**Use StreamData for**:
1. **Symmetry Properties**: Fairness metrics unchanged when groups swapped
2. **Monotonicity**: Metric values increase with worse fairness
3. **Boundedness**: Metrics within expected ranges [0, 1] or [0, ∞)
4. **Invariants**: Certain transformations preserve fairness

**Example**:
```elixir
defmodule ExFairness.Properties.DemographicParityTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  property "demographic parity is symmetric" do
    check all predictions <- tensor_binary(100),
              sensitive <- tensor_binary(100) do

      result1 = ExFairness.demographic_parity(predictions, sensitive)
      result2 = ExFairness.demographic_parity(predictions, Nx.subtract(1, sensitive))

      assert_in_delta(result1.disparity, result2.disparity, 0.001)
    end
  end

  property "disparity is non-negative" do
    check all predictions <- tensor_binary(100),
              sensitive <- tensor_binary(100) do

      result = ExFairness.demographic_parity(predictions, sensitive)

      assert result.disparity >= 0
    end
  end

  property "disparity is at most 1.0" do
    check all predictions <- tensor_binary(100),
              sensitive <- tensor_binary(100) do

      result = ExFairness.demographic_parity(predictions, sensitive)

      assert result.disparity <= 1.0
    end
  end
end
```

### Integration Testing

**Test Complete Workflows**:
1. **Detection → Mitigation → Validation Pipeline**:
   ```elixir
   test "complete fairness improvement workflow" do
     # 1. Detect bias
     initial_report = ExFairness.fairness_report(predictions, labels, sensitive_attr)
     assert initial_report.demographic_parity.passes == false

     # 2. Apply mitigation
     {fair_data, weights} = ExFairness.Mitigation.reweight(
       training_data,
       sensitive_attr,
       target: :demographic_parity
     )

     # 3. Retrain (mock)
     new_predictions = retrain_with_weights(fair_data, weights)

     # 4. Validate improvement
     final_report = ExFairness.fairness_report(new_predictions, labels, sensitive_attr)
     assert final_report.demographic_parity.passes == true
     assert final_report.demographic_parity.disparity < initial_report.demographic_parity.disparity
   end
   ```

2. **Multi-Metric Analysis**:
   - Test that impossibility theorems are detected
   - Verify trade-off warnings are generated
   - Check that recommendations are context-appropriate

3. **Real-World Datasets**:
   - Adult Income dataset
   - COMPAS recidivism dataset
   - German Credit dataset
   - Synthetic datasets with known bias

### Benchmark Testing

**Performance Requirements**:
- 10,000 samples: < 100ms for basic metrics
- 100,000 samples: < 1s for basic metrics
- Bootstrap CI (1000 samples): < 5s
- Intersectional analysis (3 attributes): < 10s

**Example**:
```elixir
defmodule ExFairness.BenchmarkTest do
  use ExUnit.Case

  @tag :benchmark
  test "demographic parity scales linearly" do
    for n <- [1_000, 10_000, 100_000] do
      predictions = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)
      sensitive = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)

      {time, _result} = :timer.tc(fn ->
        ExFairness.demographic_parity(predictions, sensitive)
      end)

      time_ms = time / 1000
      IO.puts("n=#{n}: #{time_ms}ms")

      # Assert reasonable performance
      assert time_ms < n / 100  # Linear scaling assumption
    end
  end
end
```

---

## Quality Gates - Zero Tolerance

### Compiler Warnings

**Requirement**: ZERO warnings

**Configuration** (mix.exs):
```elixir
def project do
  [
    # ...
    elixirc_options: [warnings_as_errors: true],
    # ...
  ]
end
```

**Common Issues to Avoid**:
- Unused variables (prefix with `_` if intentionally unused)
- Undefined functions
- Deprecated function usage
- Pattern matching issues

### Dialyzer - Type Specifications

**Requirement**: ZERO dialyzer errors

**Setup**:
```elixir
# mix.exs
defp deps do
  [
    {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
  ]
end
```

**Type Specs Required For**:
- All public functions
- All callbacks
- Complex private functions

**Example**:
```elixir
@type prediction :: Nx.Tensor.t()
@type label :: Nx.Tensor.t()
@type sensitive_attribute :: Nx.Tensor.t()
@type opts :: keyword()

@type fairness_result :: %{
  group_a_rate: float(),
  group_b_rate: float(),
  disparity: float(),
  passes: boolean(),
  threshold: float()
}

@spec demographic_parity(prediction(), sensitive_attribute(), opts()) :: fairness_result()
def demographic_parity(predictions, sensitive_attr, opts \\ []) do
  # Implementation
end
```

### Test Coverage

**Requirement**: 90%+ coverage for all modules

**Setup**:
```elixir
# mix.exs
def project do
  [
    # ...
    test_coverage: [tool: ExCoveralls],
    preferred_cli_env: [
      coveralls: :test,
      "coveralls.detail": :test,
      "coveralls.html": :test
    ]
  ]
end

defp deps do
  [
    {:excoveralls, "~> 0.18", only: :test}
  ]
end
```

**Run**:
```bash
mix coveralls
mix coveralls.html  # Generate HTML report
```

### Documentation

**Requirement**: All public functions documented

**Style**:
```elixir
@doc """
Computes demographic parity disparity between groups.

Demographic parity (statistical parity) requires that the probability of a
positive prediction is equal across groups defined by the sensitive attribute.

## Mathematical Definition

    P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)

## Parameters

  * `predictions` - Binary predictions tensor (0 or 1)
  * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
  * `opts` - Options:
    * `:threshold` - Maximum acceptable disparity (default: 0.1)
    * `:include_ci` - Include bootstrap confidence interval (default: false)
    * `:bootstrap_samples` - Number of bootstrap samples (default: 1000)
    * `:confidence_level` - Confidence level for CI (default: 0.95)

## Returns

A map containing:
  * `:group_a_rate` - Positive prediction rate for group A
  * `:group_b_rate` - Positive prediction rate for group B
  * `:disparity` - Absolute difference between rates
  * `:passes` - Whether disparity is within threshold
  * `:threshold` - Threshold used
  * `:confidence_interval` - (optional) Bootstrap CI tuple {lower, upper}
  * `:p_value` - (optional) Statistical significance p-value

## Examples

    iex> predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
    iex> sensitive_attr = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    iex> ExFairness.demographic_parity(predictions, sensitive_attr)
    %{
      group_a_rate: 0.5,
      group_b_rate: 0.75,
      disparity: 0.25,
      passes: false,
      threshold: 0.1
    }

## When to Use

- When equal representation in positive outcomes is required
- Advertising and content recommendation systems
- When base rates can legitimately differ between groups

## Limitations

- Ignores base rate differences in actual outcomes
- May conflict with accuracy if base rates differ
- Can be satisfied by a random classifier

## References

- Dwork, C., et al. (2012). "Fairness through awareness." ITCS.
- Feldman, M., et al. (2015). "Certifying and removing disparate impact." KDD.
"""
@spec demographic_parity(prediction(), sensitive_attribute(), opts()) :: fairness_result()
def demographic_parity(predictions, sensitive_attr, opts \\ []) do
  # Implementation
end
```

### Continuous Integration

**GitHub Actions Workflow** (.github/workflows/ci.yml):
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        elixir: ['1.14', '1.15', '1.16']
        otp: ['25', '26']
    steps:
      - uses: actions/checkout@v3
      - uses: erlef/setup-beam@v1
        with:
          elixir-version: ${{ matrix.elixir }}
          otp-version: ${{ matrix.otp }}
      - name: Install dependencies
        run: mix deps.get
      - name: Compile (warnings as errors)
        run: mix compile --warnings-as-errors
      - name: Run tests
        run: mix test
      - name: Check coverage
        run: mix coveralls
      - name: Run dialyzer
        run: mix dialyzer
      - name: Check formatting
        run: mix format --check-formatted
      - name: Run credo
        run: mix credo --strict

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: erlef/setup-beam@v1
      - name: Install dependencies
        run: mix deps.get
      - name: Generate docs
        run: mix docs
```

---

## Implementation Phases - TDD Red-Green-Refactor

### Phase 1: Core Infrastructure (Week 1)

**Goal**: Establish foundation with utils and basic validation

#### Red Phase (Write Failing Tests First)
```elixir
# test/ex_fairness/utils_test.exs
defmodule ExFairness.UtilsTest do
  use ExUnit.Case

  test "validates binary tensors" do
    assert ExFairness.Utils.validate_binary_tensor!(Nx.tensor([0, 1, 1, 0]))

    assert_raise ExFairness.Error, fn ->
      ExFairness.Utils.validate_binary_tensor!(Nx.tensor([0, 1, 2]))
    end
  end

  test "validates tensor shapes match" do
    t1 = Nx.tensor([1, 2, 3])
    t2 = Nx.tensor([4, 5, 6])
    assert ExFairness.Utils.validate_shapes_match!(t1, t2)

    t3 = Nx.tensor([7, 8])
    assert_raise ExFairness.Error, fn ->
      ExFairness.Utils.validate_shapes_match!(t1, t3)
    end
  end

  test "computes positive rate with mask" do
    predictions = Nx.tensor([1, 0, 1, 1, 0, 1, 0, 1])
    mask = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0])

    rate = ExFairness.Utils.positive_rate(predictions, mask)
    assert_in_delta(rate, 0.5, 0.001)
  end
end
```

#### Green Phase (Implement to Pass)
```elixir
# lib/ex_fairness/utils.ex
defmodule ExFairness.Utils do
  import Nx.Defn

  @doc "Validates tensor contains only 0 and 1"
  def validate_binary_tensor!(tensor) do
    {min, max} = {Nx.reduce_min(tensor), Nx.reduce_max(tensor)}

    if Nx.to_number(min) < 0 or Nx.to_number(max) > 1 do
      raise ExFairness.Error, "Tensor must be binary (0 or 1)"
    end

    tensor
  end

  @doc "Validates two tensors have same shape"
  def validate_shapes_match!(t1, t2) do
    if Nx.shape(t1) != Nx.shape(t2) do
      raise ExFairness.Error, "Tensor shapes must match"
    end

    {t1, t2}
  end

  @doc "Computes positive rate for masked subset"
  defn positive_rate(predictions, mask) do
    masked_preds = Nx.select(mask, predictions, 0)
    count = Nx.sum(mask)

    Nx.sum(masked_preds) / count
  end
end
```

#### Refactor Phase
- Extract common validation patterns
- Add comprehensive error messages
- Add type specs
- Document edge cases

### Phase 2: Demographic Parity (Week 1)

#### Red Phase
```elixir
# test/ex_fairness/metrics/demographic_parity_test.exs
defmodule ExFairness.Metrics.DemographicParityTest do
  use ExUnit.Case

  test "computes perfect parity" do
    predictions = Nx.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    result = ExFairness.Metrics.DemographicParity.compute(predictions, sensitive)

    assert result.disparity == 0.0
    assert result.passes == true
  end

  test "detects disparity" do
    predictions = Nx.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    sensitive = Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    result = ExFairness.Metrics.DemographicParity.compute(predictions, sensitive)

    assert result.disparity == 1.0
    assert result.passes == false
  end
end
```

#### Green Phase
```elixir
# lib/ex_fairness/metrics/demographic_parity.ex
defmodule ExFairness.Metrics.DemographicParity do
  alias ExFairness.Utils

  @default_threshold 0.1

  def compute(predictions, sensitive_attr, opts \\ []) do
    # Validate
    Utils.validate_binary_tensor!(predictions)
    Utils.validate_binary_tensor!(sensitive_attr)
    Utils.validate_shapes_match!(predictions, sensitive_attr)

    threshold = Keyword.get(opts, :threshold, @default_threshold)

    # Compute masks
    group_a_mask = Nx.equal(sensitive_attr, 0)
    group_b_mask = Nx.equal(sensitive_attr, 1)

    # Compute rates
    rate_a = Utils.positive_rate(predictions, group_a_mask) |> Nx.to_number()
    rate_b = Utils.positive_rate(predictions, group_b_mask) |> Nx.to_number()

    # Compute disparity
    disparity = abs(rate_a - rate_b)

    %{
      group_a_rate: rate_a,
      group_b_rate: rate_b,
      disparity: disparity,
      passes: disparity <= threshold,
      threshold: threshold
    }
  end
end
```

#### Refactor Phase
- Add statistical testing
- Add confidence intervals
- Add interpretation
- Optimize tensor operations

### Phase 3: Remaining Metrics (Weeks 2-3)

**Repeat TDD cycle for**:
1. Equalized Odds
2. Equal Opportunity
3. Predictive Parity
4. Calibration
5. Individual Fairness
6. Counterfactual Fairness

### Phase 4: Detection Algorithms (Week 4)

**Repeat TDD cycle for**:
1. Statistical Parity Testing
2. Disparate Impact
3. Intersectional Analysis
4. Temporal Drift
5. Label Bias
6. Representation Bias

### Phase 5: Mitigation Techniques (Weeks 5-6)

**Repeat TDD cycle for**:
1. Reweighting
2. Resampling
3. Threshold Optimization
4. Adversarial Debiasing
5. Fair Representation Learning
6. Calibration

### Phase 6: Reporting and Integration (Week 7)

**Implement**:
1. Fairness report generation
2. Export formats (Markdown, JSON, HTML)
3. Main API consolidation
4. Integration examples

### Phase 7: Documentation and Polish (Week 8)

**Complete**:
1. Comprehensive API docs
2. Usage guides
3. Tutorial notebooks
4. Example applications
5. Performance optimization

---

## Key Algorithms - Implementation Details

### Bootstrap Confidence Intervals

```elixir
defmodule ExFairness.Utils.Bootstrap do
  @doc """
  Computes bootstrap confidence interval for a statistic.

  ## Parameters

    * `data` - Input data tensors (list of tensors)
    * `statistic_fn` - Function to compute statistic from data
    * `n_samples` - Number of bootstrap samples (default: 1000)
    * `confidence_level` - Confidence level (default: 0.95)

  ## Returns

  Tuple {lower, upper} representing confidence interval
  """
  def confidence_interval(data, statistic_fn, opts \\ []) do
    n_samples = Keyword.get(opts, :n_samples, 1000)
    confidence_level = Keyword.get(opts, :confidence_level, 0.95)

    # Get sample size
    n = elem(Nx.shape(hd(data)), 0)

    # Generate bootstrap samples
    bootstrap_statistics =
      for _ <- 1..n_samples do
        # Sample with replacement
        indices = Enum.map(1..n, fn _ -> :rand.uniform(n) - 1 end) |> Nx.tensor()

        # Create bootstrap sample
        bootstrap_data = Enum.map(data, fn tensor ->
          Nx.take(tensor, indices)
        end)

        # Compute statistic
        statistic_fn.(bootstrap_data)
      end
      |> Enum.sort()

    # Compute percentiles
    alpha = 1 - confidence_level
    lower_idx = floor(n_samples * alpha / 2)
    upper_idx = ceil(n_samples * (1 - alpha / 2))

    lower = Enum.at(bootstrap_statistics, lower_idx)
    upper = Enum.at(bootstrap_statistics, upper_idx)

    {lower, upper}
  end
end
```

### Confusion Matrix Computation

```elixir
defmodule ExFairness.Utils.Metrics do
  import Nx.Defn

  @doc "Computes confusion matrix for masked subset"
  defn confusion_matrix(predictions, labels, mask) do
    # Apply mask
    preds = Nx.select(mask, predictions, -1)
    labs = Nx.select(mask, labels, -1)

    # Compute counts (only for masked elements)
    tp = Nx.sum(Nx.logical_and(Nx.equal(preds, 1), Nx.equal(labs, 1)))
    fp = Nx.sum(Nx.logical_and(Nx.equal(preds, 1), Nx.equal(labs, 0)))
    tn = Nx.sum(Nx.logical_and(Nx.equal(preds, 0), Nx.equal(labs, 0)))
    fn_ = Nx.sum(Nx.logical_and(Nx.equal(preds, 0), Nx.equal(labs, 1)))

    %{tp: tp, fp: fp, tn: tn, fn: fn_}
  end

  defn true_positive_rate(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    cm.tp / (cm.tp + cm.fn)
  end

  defn false_positive_rate(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    cm.fp / (cm.fp + cm.tn)
  end

  defn positive_predictive_value(predictions, labels, mask) do
    cm = confusion_matrix(predictions, labels, mask)
    cm.tp / (cm.tp + cm.fp)
  end
end
```

### Maximum Mean Discrepancy (MMD)

```elixir
defmodule ExFairness.Utils.MMD do
  import Nx.Defn

  @doc """
  Computes Maximum Mean Discrepancy between two distributions.

  Uses RBF (Gaussian) kernel by default.
  """
  defn maximum_mean_discrepancy(x, y, opts \\ []) do
    kernel_type = opts[:kernel] || :rbf
    bandwidth = opts[:bandwidth] || 1.0

    # Compute kernel matrices
    k_xx = kernel_matrix(x, x, kernel_type, bandwidth)
    k_yy = kernel_matrix(y, y, kernel_type, bandwidth)
    k_xy = kernel_matrix(x, y, kernel_type, bandwidth)

    # MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    mmd_squared = Nx.mean(k_xx) + Nx.mean(k_yy) - 2 * Nx.mean(k_xy)

    # Return MMD (take sqrt of max(0, mmd²))
    Nx.sqrt(Nx.max(mmd_squared, 0))
  end

  defnp kernel_matrix(x, y, :rbf, bandwidth) do
    # Compute pairwise squared distances
    x_norm = Nx.sum(x * x, axes: [1], keep_axes: true)
    y_norm = Nx.sum(y * y, axes: [1], keep_axes: true)

    distances_squared =
      x_norm
      |> Nx.add(Nx.transpose(y_norm))
      |> Nx.subtract(2 * Nx.dot(x, [1], y, [1]))

    # RBF kernel: exp(-||x-y||²/(2σ²))
    Nx.exp(Nx.divide(distances_squared, -2 * bandwidth * bandwidth))
  end
end
```

---

## Integration Examples

### With Axon (Neural Network Training)

```elixir
defmodule MyApp.FairModel do
  def train_with_fairness(train_data, train_labels, sensitive_attr) do
    # Define model
    model =
      Axon.input("features")
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(1, activation: :sigmoid)

    # Train with reweighted samples
    weights = ExFairness.Mitigation.Reweighting.compute_weights(
      train_labels,
      sensitive_attr,
      target: :demographic_parity
    )

    # Training loop with sample weights
    trained_model =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(
        Stream.zip(train_data, train_labels, weights),
        %{},
        epochs: 50,
        compiler: EXLA
      )

    trained_model
  end

  def validate_fairness(model, test_data, test_labels, sensitive_attr) do
    # Get predictions
    predictions = Axon.predict(model, test_data)
    binary_predictions = Nx.greater(predictions, 0.5)

    # Generate fairness report
    report = ExFairness.fairness_report(
      binary_predictions,
      test_labels,
      sensitive_attr,
      metrics: [:demographic_parity, :equalized_odds, :equal_opportunity]
    )

    report
  end
end
```

### With Scholar (Classical ML)

```elixir
defmodule MyApp.FairClassifier do
  def train_fair_logistic_regression(features, labels, sensitive_attr) do
    # Apply reweighting
    weights = ExFairness.Mitigation.Reweighting.compute_weights(
      labels,
      sensitive_attr,
      target: :demographic_parity
    )

    # Train with Scholar
    model = Scholar.Linear.LogisticRegression.fit(
      features,
      labels,
      sample_weights: weights
    )

    model
  end

  def optimize_thresholds(model, features, labels, sensitive_attr) do
    # Get probability predictions
    probabilities = Scholar.Linear.LogisticRegression.predict_probability(model, features)

    # Find optimal thresholds
    thresholds = ExFairness.Mitigation.ThresholdOptimization.optimize(
      probabilities,
      labels,
      sensitive_attr,
      target_metric: :equalized_odds,
      epsilon: 0.05
    )

    # Apply thresholds to get final predictions
    predictions = ExFairness.Mitigation.ThresholdOptimization.apply(
      probabilities,
      sensitive_attr,
      thresholds
    )

    {predictions, thresholds}
  end
end
```

### Production Monitoring

```elixir
defmodule MyApp.FairnessMonitor do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    # Initialize with baseline metrics
    baseline = compute_baseline_metrics()

    # Schedule periodic checks
    schedule_check()

    {:ok, %{
      baseline: baseline,
      history: [],
      alert_threshold: opts[:alert_threshold] || 0.05
    }}
  end

  def handle_info(:check_fairness, state) do
    # Get recent predictions
    {predictions, labels, sensitive_attrs} = fetch_recent_data()

    # Compute current metrics
    current_metrics = ExFairness.fairness_report(
      predictions,
      labels,
      sensitive_attrs,
      metrics: [:demographic_parity, :equalized_odds]
    )

    # Check for drift
    drift_result = ExFairness.Detection.TemporalDrift.detect(
      state.history ++ [{DateTime.utc_now(), current_metrics}],
      baseline: state.baseline,
      threshold: state.alert_threshold
    )

    # Alert if drift detected
    if drift_result.drift_detected do
      send_alert(drift_result)
    end

    # Schedule next check
    schedule_check()

    {:noreply, %{state | history: state.history ++ [{DateTime.utc_now(), current_metrics}]}}
  end

  defp schedule_check do
    # Check every hour
    Process.send_after(self(), :check_fairness, :timer.hours(1))
  end
end
```

---

## Testing Datasets

### Required Test Datasets

1. **Synthetic Balanced Dataset**
   - Perfect fairness (all metrics pass)
   - Use for baseline correctness tests

2. **Synthetic Biased Dataset**
   - Known bias magnitude
   - Use for detection sensitivity tests

3. **Adult Income Dataset**
   - Real-world census data
   - Binary classification (income >50K)
   - Sensitive attributes: gender, race
   - Use for integration tests

4. **COMPAS Recidivism Dataset**
   - Criminal justice risk assessment
   - Known fairness issues
   - Use for calibration and equalized odds tests

5. **German Credit Dataset**
   - Credit approval
   - Multiple sensitive attributes
   - Use for intersectional fairness tests

### Dataset Loading Utilities

```elixir
defmodule ExFairness.TestDatasets do
  @moduledoc """
  Utilities for loading standard fairness testing datasets.
  """

  def synthetic_balanced(n \\ 1000) do
    # Generate perfectly balanced dataset
    predictions = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)
    labels = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)

    # Sensitive attribute (50/50 split)
    sensitive = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)

    {predictions, labels, sensitive}
  end

  def synthetic_biased(n \\ 1000, bias_magnitude \\ 0.3) do
    # Group A: 50% positive rate
    # Group B: (50% - bias_magnitude) positive rate

    sensitive = Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)

    predictions =
      Nx.select(
        sensitive,
        # Group B (1): lower positive rate
        Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5 + bias_magnitude),
        # Group A (0): baseline positive rate
        Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)
      )

    # Labels somewhat correlated with predictions
    labels =
      Nx.select(
        Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.2),
        predictions,
        Nx.random_uniform({n}, 0, 1) |> Nx.greater(0.5)
      )

    {predictions, labels, sensitive}
  end

  def load_adult_income do
    # Load from CSV or cached dataset
    # Implementation depends on data storage approach
    # Return {features, labels, sensitive_attrs}
  end
end
```

---

## Error Handling and Validation

### Custom Error Module

```elixir
defmodule ExFairness.Error do
  @moduledoc """
  Custom error for ExFairness operations.
  """
  defexception [:message]

  def exception(message) when is_binary(message) do
    %__MODULE__{message: message}
  end
end
```

### Comprehensive Validation

```elixir
defmodule ExFairness.Validation do
  @moduledoc """
  Input validation utilities.
  """

  @doc "Validates predictions tensor"
  def validate_predictions!(predictions) do
    validate_tensor!(predictions, "predictions")
    validate_binary!(predictions, "predictions")
    validate_non_empty!(predictions, "predictions")

    predictions
  end

  @doc "Validates labels tensor"
  def validate_labels!(labels) do
    validate_tensor!(labels, "labels")
    validate_binary!(labels, "labels")
    validate_non_empty!(labels, "labels")

    labels
  end

  @doc "Validates sensitive attribute tensor"
  def validate_sensitive_attr!(sensitive_attr) do
    validate_tensor!(sensitive_attr, "sensitive_attr")
    validate_binary!(sensitive_attr, "sensitive_attr")
    validate_non_empty!(sensitive_attr, "sensitive_attr")
    validate_multiple_groups!(sensitive_attr)
    validate_sufficient_samples!(sensitive_attr)

    sensitive_attr
  end

  @doc "Validates tensors have matching shapes"
  def validate_matching_shapes!(tensors, names) do
    shapes = Enum.map(tensors, &Nx.shape/1)

    unless Enum.all?(shapes, fn s -> s == hd(shapes) end) do
      raise ExFairness.Error, """
      Tensor shape mismatch.

      Expected all tensors to have the same shape, but got:
      #{Enum.zip(names, shapes) |> Enum.map(fn {n, s} -> "  #{n}: #{inspect(s)}" end) |> Enum.join("\n")}
      """
    end

    tensors
  end

  defp validate_tensor!(value, name) do
    unless match?(%Nx.Tensor{}, value) do
      raise ExFairness.Error, "#{name} must be an Nx.Tensor, got: #{inspect(value)}"
    end
  end

  defp validate_binary!(tensor, name) do
    min_val = Nx.reduce_min(tensor) |> Nx.to_number()
    max_val = Nx.reduce_max(tensor) |> Nx.to_number()

    unless min_val >= 0 and max_val <= 1 do
      raise ExFairness.Error, """
      #{name} must be binary (containing only 0 and 1).

      Found values in range [#{min_val}, #{max_val}].
      """
    end
  end

  defp validate_non_empty!(tensor, name) do
    size = Nx.size(tensor)

    if size == 0 do
      raise ExFairness.Error, "#{name} cannot be empty"
    end
  end

  defp validate_multiple_groups!(sensitive_attr) do
    unique_count =
      sensitive_attr
      |> Nx.to_flat_list()
      |> Enum.uniq()
      |> length()

    if unique_count < 2 do
      raise ExFairness.Error, """
      sensitive_attr must contain at least 2 different groups.

      Found only #{unique_count} unique value(s).
      Fairness metrics require comparing multiple groups.
      """
    end
  end

  defp validate_sufficient_samples!(sensitive_attr, min_per_group \\ 10) do
    group_0_count = Nx.sum(Nx.equal(sensitive_attr, 0)) |> Nx.to_number()
    group_1_count = Nx.sum(Nx.equal(sensitive_attr, 1)) |> Nx.to_number()

    if group_0_count < min_per_group or group_1_count < min_per_group do
      raise ExFairness.Error, """
      Insufficient samples per group for reliable fairness metrics.

      Found:
        Group 0: #{group_0_count} samples
        Group 1: #{group_1_count} samples

      Recommended minimum: #{min_per_group} samples per group.

      Consider:
      - Collecting more data
      - Using bootstrap methods with caution
      - Aggregating smaller groups if appropriate
      """
    end
  end
end
```

---

## Performance Optimization

### Nx Defn for GPU Acceleration

```elixir
defmodule ExFairness.Metrics.DemographicParityOptimized do
  import Nx.Defn

  @doc "GPU-accelerated demographic parity computation"
  defn compute_disparity(predictions, sensitive_attr) do
    # All operations in defn are JIT-compiled for GPU
    group_a_mask = Nx.equal(sensitive_attr, 0)
    group_b_mask = Nx.equal(sensitive_attr, 1)

    # Vectorized operations
    rate_a = Nx.sum(Nx.select(group_a_mask, predictions, 0)) / Nx.sum(group_a_mask)
    rate_b = Nx.sum(Nx.select(group_b_mask, predictions, 0)) / Nx.sum(group_b_mask)

    disparity = Nx.abs(rate_a - rate_b)

    {rate_a, rate_b, disparity}
  end

  def compute(predictions, sensitive_attr, opts \\ []) do
    # Validation in regular Elixir
    ExFairness.Validation.validate_predictions!(predictions)
    ExFairness.Validation.validate_sensitive_attr!(sensitive_attr)

    # GPU-accelerated computation
    {rate_a, rate_b, disparity} = compute_disparity(predictions, sensitive_attr)

    # Post-processing in Elixir
    threshold = Keyword.get(opts, :threshold, 0.1)

    %{
      group_a_rate: Nx.to_number(rate_a),
      group_b_rate: Nx.to_number(rate_b),
      disparity: Nx.to_number(disparity),
      passes: Nx.to_number(disparity) <= threshold,
      threshold: threshold
    }
  end
end
```

### Caching for Repeated Computations

```elixir
defmodule ExFairness.Cache do
  @moduledoc """
  Caches expensive intermediate computations.
  """

  def cached_confusion_matrix(predictions, labels, sensitive_attr) do
    cache_key = {
      Nx.to_binary(predictions),
      Nx.to_binary(labels),
      Nx.to_binary(sensitive_attr)
    }

    case :persistent_term.get(cache_key, nil) do
      nil ->
        result = compute_all_confusion_matrices(predictions, labels, sensitive_attr)
        :persistent_term.put(cache_key, result)
        result

      cached ->
        cached
    end
  end

  defp compute_all_confusion_matrices(predictions, labels, sensitive_attr) do
    group_a_mask = Nx.equal(sensitive_attr, 0)
    group_b_mask = Nx.equal(sensitive_attr, 1)

    %{
      group_a: ExFairness.Utils.Metrics.confusion_matrix(predictions, labels, group_a_mask),
      group_b: ExFairness.Utils.Metrics.confusion_matrix(predictions, labels, group_b_mask)
    }
  end
end
```

---

## Documentation Requirements

### README.md Sections

1. **Overview**: What is ExFairness?
2. **Features**: Comprehensive list with examples
3. **Installation**: Hex and GitHub installation
4. **Quick Start**: 5-minute tutorial
5. **Metrics Guide**: When to use each metric
6. **API Reference**: Link to HexDocs
7. **Use Cases**: Real-world applications
8. **Best Practices**: Dos and don'ts
9. **Limitations**: Impossibility theorems, caveats
10. **Research**: Citations and foundations
11. **Contributing**: How to contribute
12. **License**: MIT

### Module Documentation

**Every module must have**:
- `@moduledoc` with overview
- Examples in module doc
- `@doc` for every public function
- Doctest examples that actually run
- Type specs with `@spec`
- Links to relevant research papers

### Guides (in docs/ or guides/)

1. **Getting Started**: Installation and first steps
2. **Fairness Metrics Guide**: Deep dive into each metric
3. **Choosing Metrics**: Decision tree for metric selection
4. **Mitigation Strategies**: When and how to use each
5. **Integration Guide**: Axon, Scholar, Bumblebee examples
6. **Production Monitoring**: Building fairness monitoring systems
7. **Case Studies**: Real-world examples
8. **API Reference**: Generated by ExDoc

---

## Success Criteria

### Functional Requirements
- [ ] All 7 fairness metrics implemented
- [ ] All 6 detection algorithms implemented
- [ ] All 6 mitigation techniques implemented
- [ ] Comprehensive reporting system
- [ ] Bootstrap confidence intervals
- [ ] Statistical significance testing
- [ ] Multi-attribute intersectional analysis
- [ ] Temporal drift monitoring

### Quality Requirements
- [ ] Zero compiler warnings
- [ ] Zero dialyzer errors
- [ ] 90%+ test coverage
- [ ] All public functions documented
- [ ] All doctests pass
- [ ] CI/CD pipeline green
- [ ] Performance benchmarks met

### Integration Requirements
- [ ] Works with Nx tensors
- [ ] Integrates with Axon training
- [ ] Integrates with Scholar models
- [ ] Compatible with EXLA backend
- [ ] Examples for common workflows

### Documentation Requirements
- [ ] Comprehensive README
- [ ] API docs on HexDocs
- [ ] Usage guides
- [ ] Tutorial notebooks
- [ ] Case studies
- [ ] Research citations

---

## Timeline and Milestones

### Week 1: Foundation
- Core utils and validation
- Demographic parity metric
- Basic testing infrastructure
- CI/CD setup

### Week 2: Group Fairness Metrics
- Equalized odds
- Equal opportunity
- Predictive parity
- Calibration

### Week 3: Advanced Metrics
- Individual fairness
- Counterfactual fairness
- Bootstrap CI implementation
- Statistical testing

### Week 4: Detection Algorithms
- Statistical parity testing
- Disparate impact
- Intersectional analysis
- Temporal drift
- Label bias

### Week 5: Pre/Post Processing Mitigation
- Reweighting
- Resampling
- Threshold optimization
- Calibration

### Week 6: In-Processing Mitigation
- Adversarial debiasing (Axon)
- Fair representation learning
- Integration examples

### Week 7: Reporting and Integration
- Fairness report generation
- Export formats (MD, JSON, HTML)
- Main API consolidation
- Integration guides

### Week 8: Polish and Release
- Documentation completion
- Performance optimization
- Final testing
- v0.1.0 release to Hex

---

## Development Environment Setup

### Required Tools
```bash
# Elixir and Erlang
asdf install elixir 1.16.0
asdf install erlang 26.2.1

# Dependencies
cd /home/home/p/g/n/North-Shore-AI/ExFairness
mix deps.get
mix deps.compile

# Development tools
mix archive.install hex ex_doc
mix local.hex --force
mix local.rebar --force
```

### Development Workflow
```bash
# Run tests
mix test

# Run tests with coverage
mix coveralls
mix coveralls.html

# Type checking
mix dialyzer

# Format code
mix format

# Linting
mix credo --strict

# Generate docs
mix docs

# Full CI check
mix format --check-formatted && \
mix compile --warnings-as-errors && \
mix test && \
mix coveralls && \
mix dialyzer && \
mix credo --strict
```

---

## References and Research Foundations

### Key Papers

1. **Hardt, M., Price, E., & Srebro, N. (2016)**
   "Equality of Opportunity in Supervised Learning"
   *NeurIPS*
   - Foundation for equalized odds and equal opportunity

2. **Chouldechova, A. (2017)**
   "Fair prediction with disparate impact"
   *Big Data*, 5(2), 153-163
   - Impossibility theorem for fairness metrics

3. **Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016)**
   "Inherent trade-offs in the fair determination of risk scores"
   *ITCS*
   - Calibration vs balance trade-offs

4. **Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012)**
   "Fairness through awareness"
   *ITCS*
   - Individual fairness definition

5. **Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017)**
   "Counterfactual fairness"
   *NeurIPS*
   - Causal approach to fairness

6. **Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015)**
   "Certifying and removing disparate impact"
   *KDD*
   - Disparate impact measurement and mitigation

7. **Kamiran, F., & Calders, T. (2012)**
   "Data preprocessing techniques for classification without discrimination"
   *KAIS*
   - Reweighting and resampling techniques

8. **Zhang, B. H., Lemoine, B., & Mitchell, M. (2018)**
   "Mitigating unwanted biases with adversarial learning"
   *AIES*
   - Adversarial debiasing approach

9. **Louizos, C., Swersky, K., Li, Y., Welling, M., & Zemel, R. (2016)**
   "The variational fair autoencoder"
   *ICLR*
   - Fair representation learning

### Related Frameworks

- **IBM AI Fairness 360** (Python): Comprehensive toolkit, inspiration for API design
- **Google Fairness Indicators**: Production monitoring approach
- **Microsoft Fairlearn** (Python): Threshold optimization and mitigation
- **Aequitas**: Bias audit toolkit

---

## Appendix: Complete File Checklist

### Core Implementation Files
- [ ] lib/ex_fairness.ex (main API)
- [ ] lib/ex_fairness/utils.ex
- [ ] lib/ex_fairness/validation.ex
- [ ] lib/ex_fairness/metrics/demographic_parity.ex
- [ ] lib/ex_fairness/metrics/equalized_odds.ex
- [ ] lib/ex_fairness/metrics/equal_opportunity.ex
- [ ] lib/ex_fairness/metrics/predictive_parity.ex
- [ ] lib/ex_fairness/metrics/calibration.ex
- [ ] lib/ex_fairness/metrics/individual_fairness.ex
- [ ] lib/ex_fairness/metrics/counterfactual.ex
- [ ] lib/ex_fairness/detection/disparate_impact.ex
- [ ] lib/ex_fairness/detection/statistical_parity.ex
- [ ] lib/ex_fairness/detection/intersectional.ex
- [ ] lib/ex_fairness/detection/temporal_drift.ex
- [ ] lib/ex_fairness/detection/label_bias.ex
- [ ] lib/ex_fairness/detection/representation.ex
- [ ] lib/ex_fairness/mitigation/reweighting.ex
- [ ] lib/ex_fairness/mitigation/resampling.ex
- [ ] lib/ex_fairness/mitigation/threshold_optimization.ex
- [ ] lib/ex_fairness/mitigation/adversarial_debiasing.ex
- [ ] lib/ex_fairness/mitigation/fair_representation.ex
- [ ] lib/ex_fairness/mitigation/calibration.ex
- [ ] lib/ex_fairness/report.ex

### Test Files
- [ ] test/ex_fairness_test.exs
- [ ] test/ex_fairness/utils_test.exs
- [ ] test/ex_fairness/validation_test.exs
- [ ] test/ex_fairness/metrics/demographic_parity_test.exs
- [ ] test/ex_fairness/metrics/equalized_odds_test.exs
- [ ] test/ex_fairness/metrics/equal_opportunity_test.exs
- [ ] test/ex_fairness/metrics/predictive_parity_test.exs
- [ ] test/ex_fairness/metrics/calibration_test.exs
- [ ] test/ex_fairness/metrics/individual_fairness_test.exs
- [ ] test/ex_fairness/metrics/counterfactual_test.exs
- [ ] test/ex_fairness/detection/disparate_impact_test.exs
- [ ] test/ex_fairness/detection/statistical_parity_test.exs
- [ ] test/ex_fairness/detection/intersectional_test.exs
- [ ] test/ex_fairness/detection/temporal_drift_test.exs
- [ ] test/ex_fairness/detection/label_bias_test.exs
- [ ] test/ex_fairness/detection/representation_test.exs
- [ ] test/ex_fairness/mitigation/reweighting_test.exs
- [ ] test/ex_fairness/mitigation/resampling_test.exs
- [ ] test/ex_fairness/mitigation/threshold_optimization_test.exs
- [ ] test/ex_fairness/mitigation/adversarial_debiasing_test.exs
- [ ] test/ex_fairness/mitigation/fair_representation_test.exs
- [ ] test/ex_fairness/mitigation/calibration_test.exs
- [ ] test/ex_fairness/report_test.exs

### Documentation Files
- [x] README.md
- [x] docs/architecture.md
- [x] docs/metrics.md
- [x] docs/algorithms.md
- [x] docs/roadmap.md
- [ ] CHANGELOG.md
- [ ] CONTRIBUTING.md
- [ ] LICENSE
- [ ] guides/getting_started.md
- [ ] guides/choosing_metrics.md
- [ ] guides/mitigation_strategies.md
- [ ] guides/integration.md
- [ ] guides/case_studies.md

### Configuration Files
- [x] mix.exs
- [ ] .formatter.exs
- [ ] .dialyzer_ignore.exs
- [ ] .github/workflows/ci.yml
- [ ] .gitignore

---

## Final Notes

This buildout prompt provides a complete specification for implementing the ExFairness framework. The implementation should follow **strict TDD** (Red-Green-Refactor), maintain **zero warnings and dialyzer errors**, achieve **90%+ test coverage**, and provide **comprehensive documentation**.

The framework bridges academic fairness research and production ML systems in the Elixir ecosystem, providing mathematical rigor, transparency, and actionable mitigation strategies.

**Key Principles**:
1. **Test-Driven Development**: Write tests first, always
2. **Mathematical Correctness**: Implement metrics per research papers
3. **Performance**: Leverage Nx for GPU acceleration
4. **Usability**: Clear API, helpful error messages, comprehensive docs
5. **Integration**: Seamless integration with Elixir ML ecosystem

**Success Metrics**:
- All quality gates pass (warnings, dialyzer, coverage, tests)
- All fairness metrics implemented correctly
- Comprehensive test suite with real-world datasets
- Production-ready documentation
- v0.1.0 released to Hex.pm

---

**Buildout Prompt Complete**
**Ready for Implementation**
**Date: 2025-10-20**
