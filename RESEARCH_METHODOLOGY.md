# Research Methodology: LLM Reliability and Performance Optimization

**Version:** 0.1.0
**Last Updated:** 2025-10-08
**Authors:** Research Infrastructure Team
**Ethics Approval:** Not required (no human subjects, public datasets only)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Questions](#research-questions)
3. [Six Core Hypotheses](#six-core-hypotheses)
4. [Experimental Design](#experimental-design)
5. [Statistical Methods](#statistical-methods)
6. [Datasets and Benchmarks](#datasets-and-benchmarks)
7. [Metrics and Measurements](#metrics-and-measurements)
8. [Reproducibility Protocol](#reproducibility-protocol)
9. [Ethical Considerations](#ethical-considerations)
10. [Publication Guidelines](#publication-guidelines)

---

## Executive Summary

This document defines the scientific methodology for conducting rigorous experiments on large language model (LLM) reliability, performance, and cost optimization using the Elixir AI Research Framework.

**Research Goal:**

Systematically evaluate techniques for improving LLM system reliability while maintaining acceptable performance and cost characteristics, with focus on production-ready strategies that can be deployed at scale.

**Core Research Questions:**

1. Can multi-model ensemble voting achieve >99% reliability?
2. Does request hedging reduce tail latencies without prohibitive cost?
3. How do ensemble size, voting strategy, and model selection affect accuracy-cost trade-offs?
4. Can adaptive strategies learn optimal configurations over time?
5. Does transparency (causal tracing) affect user trust and debugging efficiency?
6. Which statistical tests are most appropriate for LLM evaluation?

**Methodology Highlights:**

- **Randomized controlled experiments** with multiple conditions
- **Repeated measures designs** for within-subjects comparisons
- **Factorial designs** to study interaction effects
- **Power analysis** to ensure adequate sample sizes
- **Multiple comparison correction** to control false discovery rate
- **Effect size reporting** beyond mere significance testing
- **Complete reproducibility** via deterministic seeding

**Expected Impact:**

This research aims to provide **evidence-based guidelines** for building reliable LLM systems, moving beyond anecdotal "best practices" to statistically-validated strategies with quantified trade-offs.

---

## Research Questions

### Primary Research Questions

**RQ1: Reliability**

> Can multi-model ensemble methods achieve significantly higher reliability (accuracy, consistency) than single-model baselines?

- **Null Hypothesis (H₀):** Ensemble accuracy = Single model accuracy
- **Alternative Hypothesis (H₁):** Ensemble accuracy > Single model accuracy
- **Significance Level:** α = 0.05
- **Minimum Detectable Effect:** δ = 0.05 (5 percentage points)
- **Desired Power:** 1-β = 0.80

**RQ2: Latency**

> Does request hedging significantly reduce tail latencies (P95, P99) compared to non-hedged requests?

- **Null Hypothesis (H₀):** P99_hedged = P99_baseline
- **Alternative Hypothesis (H₁):** P99_hedged < P99_baseline
- **Significance Level:** α = 0.05
- **Minimum Detectable Effect:** δ = 25% reduction
- **Desired Power:** 1-β = 0.80

**RQ3: Cost-Accuracy Trade-offs**

> What is the Pareto frontier of accuracy vs cost for different ensemble configurations?

- **Analysis:** Multi-objective optimization
- **Metrics:** Accuracy (%), Cost (USD per query)
- **Configurations:** 2^n factorial design (size, strategy, models)

### Secondary Research Questions

**RQ4: Adaptation**

> Do adaptive strategies (learning from history) outperform static strategies over time?

- **Design:** Time series analysis with repeated measures
- **Metrics:** Accuracy, latency, cost over time
- **Statistical Test:** Growth curve modeling or ARIMA

**RQ5: Transparency**

> Does providing causal decision traces improve user trust and debugging efficiency?

- **Design:** User study with control/treatment groups
- **Metrics:** Trust scores (Likert scale), debugging time, success rate
- **Statistical Test:** Mann-Whitney U (non-parametric)

**RQ6: Statistical Methods**

> Which statistical tests have highest power for detecting LLM performance differences?

- **Design:** Simulation study with known ground truth
- **Metrics:** Type I error rate, power, robustness
- **Statistical Test:** Meta-analysis of test characteristics

---

## Six Core Hypotheses

### H1: Ensemble Reliability (Primary)

**Hypothesis:**

> A 5-model ensemble with majority voting achieves ≥99% accuracy on MMLU-STEM, significantly higher than the best single model (≤92%).

**Type:** Superiority trial

**Design:** Randomized controlled experiment
- **Control:** Best single model (GPT-4)
- **Treatment:** 5-model ensemble (GPT-4, Claude Opus, Gemini Pro, GPT-3.5, Claude Sonnet)
- **Voting:** Majority vote
- **Dataset:** MMLU-STEM, n=200 questions (stratified sampling)
- **Repetitions:** 3 independent runs with different random seeds

**Statistical Analysis:**

```
Primary Analysis: Independent samples t-test
- DV: Accuracy (proportion correct)
- IV: Condition (control vs ensemble)
- Assumptions: Check normality (Shapiro-Wilk), equal variance (Levene's test)
- Fallback: Mann-Whitney U if assumptions violated
- Effect size: Cohen's d
- Confidence interval: 95% CI for mean difference

Power Analysis:
- Expected effect: d = 1.5 (large effect)
- α = 0.05, 1-β = 0.80
- Required n per group: 12 subjects
- Actual n = 200 (highly powered)

Multiple Comparisons:
- If testing multiple ensemble sizes: Bonferroni correction
- Family-wise error rate: α_family = 0.05
```

**Success Criteria:**

1. p < 0.05 (statistically significant)
2. Ensemble accuracy ≥ 0.99
3. Effect size d > 0.8 (large effect)
4. Lower bound of 95% CI for ensemble accuracy > 0.95

**Expected Results:**

Based on independence assumptions:
```
Single model error rate: 8% (0.92 accuracy)
Ensemble error (5 independent models): 0.08^5 = 0.00003 (99.997% accuracy)

In practice, errors are correlated, so expect:
Ensemble accuracy: 96-99%
Improvement: 4-7 percentage points
Cost: 5× single model
```

**Failure Modes:**

1. **High error correlation:** If models make same mistakes, no improvement
2. **Voting failures:** If 3+ models wrong, ensemble fails
3. **Dataset bias:** If MMLU-STEM too easy/hard, ceiling/floor effects

**Contingency Plans:**

- If H1 fails: Test with harder dataset (MMLU-Full), different models
- If correlation too high: Use more diverse models
- If voting suboptimal: Test weighted voting strategies

### H2: Hedging Latency Reduction (Primary)

**Hypothesis:**

> Request hedging with P95 delay reduces P99 latency by ≥50% with <15% cost increase.

**Type:** Superiority trial with cost constraint

**Design:** Paired comparison (within-subjects)
- **Condition 1:** Baseline (no hedging)
- **Condition 2:** Hedging (P95 delay, max 1 backup)
- **Requests:** n=1000 API calls to GPT-4
- **Repetitions:** 3 independent experiments

**Statistical Analysis:**

```
Primary Analysis: Paired t-test on log-transformed latencies
- DV: P99 latency (ms)
- IV: Hedging condition (baseline vs hedged)
- Transformation: log(latency) for normality
- Effect size: Cohen's d_z (within-subjects)
- Confidence interval: 95% CI for difference

Secondary Analysis: Cost overhead
- DV: Cost per query (USD)
- Statistical test: Paired t-test
- Constraint: Mean cost ratio < 1.15

Power Analysis:
- Expected effect: d_z = 1.0 (large within-subjects effect)
- α = 0.05, 1-β = 0.80
- Required n: 11 pairs
- Actual n = 1000 (very highly powered)
```

**Success Criteria:**

1. p < 0.05 for P99 latency reduction
2. P99 latency reduced by ≥50%
3. Mean cost increase <15%
4. Lower bound of 95% CI for P99 reduction > 25%

**Expected Results:**

Based on Google's research (Dean & Barroso, 2013):
```
Baseline P99: 2000ms
Hedged P99: 800ms (60% reduction)
Cost increase: 10% (hedge fires ~10% of time)
```

**Failure Modes:**

1. **Correlated latencies:** If primary and backup both slow, no improvement
2. **Cost explosion:** If hedge fires too often (>50%), cost prohibitive
3. **Cancellation failures:** If can't cancel primary, cost doubles

**Contingency Plans:**

- If latency not reduced: Try lower percentile delay (P90, P75)
- If cost too high: Reduce hedge firing rate via higher percentile
- If still fails: Test multi-tier hedging (cheap → expensive models)

### H3: Ensemble Size vs Accuracy (Secondary)

**Hypothesis:**

> Ensemble accuracy follows a power law: accuracy ∝ size^α, with diminishing returns above 5 models.

**Type:** Dose-response relationship

**Design:** 2^k factorial design
- **Factor 1:** Ensemble size (1, 2, 3, 5, 7 models)
- **Factor 2:** Voting strategy (majority, weighted, unanimous)
- **Dataset:** MMLU-STEM, n=100 per cell
- **Total cells:** 5 sizes × 3 strategies = 15 cells
- **Repetitions:** 3 runs per cell

**Statistical Analysis:**

```
Primary Analysis: Repeated measures ANOVA
- DV: Accuracy
- IV1: Ensemble size (within-subjects)
- IV2: Voting strategy (within-subjects)
- Interaction: Size × Strategy
- Assumptions: Sphericity (Mauchly's test), normality
- Correction: Greenhouse-Geisser if sphericity violated
- Post-hoc: Tukey HSD with Bonferroni correction

Secondary Analysis: Curve fitting
- Model: accuracy = a + b * size^α
- Estimation: Nonlinear least squares
- Goodness of fit: R², RMSE

Multiple Comparisons:
- 15 pairwise comparisons
- Correction: Benjamini-Hochberg FDR control
- α_FDR = 0.05
```

**Success Criteria:**

1. Main effect of size: F-test p < 0.05
2. Power law fits better than linear: AIC comparison
3. Diminishing returns evident: α < 1
4. Optimal size identified: Balance accuracy vs cost

**Expected Results:**

```
Size 1: 89% accuracy, $0.01 cost
Size 2: 94% accuracy, $0.02 cost (+5% acc, +$0.01)
Size 3: 96% accuracy, $0.03 cost (+2% acc, +$0.01)
Size 5: 97.5% accuracy, $0.05 cost (+1.5% acc, +$0.02)
Size 7: 98% accuracy, $0.07 cost (+0.5% acc, +$0.02)

Diminishing returns clear after size 5.
```

**Analysis:**

```elixir
# Fit power law
data = [
  {1, 0.89}, {2, 0.94}, {3, 0.96}, {5, 0.975}, {7, 0.98}
]

# Model: accuracy = a + b * size^α
# Expected: α ≈ 0.3-0.4 (diminishing returns)

# Calculate cost-effectiveness ratio
cost_per_accuracy = (cost / accuracy) per additional model
# Expect: ratio increases with size (diminishing returns)
```

### H4: Adaptive vs Static Strategies (Secondary)

**Hypothesis:**

> Adaptive hedging strategies learn to reduce latency by an additional 10-15% compared to static P95 strategies after 1000 queries.

**Type:** Learning curve analysis

**Design:** Time series with repeated measures
- **Condition 1:** Static P95 hedging
- **Condition 2:** Adaptive hedging (learns delay)
- **Queries:** n=2000, measured every 100 queries
- **Measurements:** 20 time points
- **Repetitions:** 3 independent runs

**Statistical Analysis:**

```
Primary Analysis: Growth Curve Modeling (GCM)
- DV: P99 latency over time
- IV: Strategy (static vs adaptive)
- Time: Linear and quadratic effects
- Model: Mixed effects with random intercepts per run
- Comparison: ΔAIC between models

Alternative: Time series ANOVA
- Repeated measures at each time point
- Correction: Greenhouse-Geisser for sphericity
- Post-hoc: Contrasts at specific time points

Effect Size:
- Partial η² for strategy effect
- Cohen's f² for model comparison
```

**Success Criteria:**

1. Significant interaction: Strategy × Time (p < 0.05)
2. Adaptive strategy shows steeper learning curve
3. Final latency difference ≥10% (practical significance)
4. Adaptive better after ≤500 queries (reasonable burn-in)

**Expected Results:**

```
Time 0 (cold start):
- Static: 2000ms P99 (optimal from start)
- Adaptive: 2500ms P99 (random initialization)

Time 500:
- Static: 2000ms P99 (no learning)
- Adaptive: 1900ms P99 (learned better delay)

Time 1000+:
- Static: 2000ms P99
- Adaptive: 1700ms P99 (15% improvement)
```

**Learning Algorithm:**

```elixir
# Adaptive strategy uses exponential weighted moving average
# to predict P95 latency dynamically

defmodule AdaptiveHedging do
  def update_delay(current_delay, observed_latency, α \\ 0.1) do
    # α = learning rate (0.1 = slow, stable learning)
    new_estimate = α * observed_latency + (1 - α) * current_delay
    new_estimate
  end
end

# Expected: Converges to optimal delay after ~500 observations
# Formula: N = -ln(ε) / ln(1 - α)  where ε = tolerance
# For α=0.1, ε=0.05: N ≈ 500 queries
```

### H5: Voting Strategy Comparison (Secondary)

**Hypothesis:**

> Weighted voting outperforms majority voting when confidence scores are well-calibrated (calibration error <10%).

**Type:** Conditional superiority trial

**Design:** 2×2 factorial with moderator
- **Factor 1:** Voting strategy (majority vs weighted)
- **Factor 2:** Model set (well-calibrated vs poorly-calibrated)
- **Moderator:** Calibration error (continuous)
- **Dataset:** MMLU-STEM, n=200 per cell
- **Total cells:** 2 strategies × 2 model sets = 4 cells

**Statistical Analysis:**

```
Primary Analysis: Two-way ANOVA with interaction
- DV: Accuracy
- IV1: Voting strategy
- IV2: Model calibration quality
- Interaction: Strategy × Calibration
- Hypothesis: Significant interaction (weighted benefits from good calibration)

Secondary Analysis: Moderation analysis
- Predictor: Voting strategy
- Outcome: Accuracy
- Moderator: Calibration error (continuous)
- Method: Regression with interaction term
- Test: Δaccuracy = β₀ + β₁·strategy + β₂·calibration + β₃·strategy×calibration

Calibration Measurement:
- Expected Calibration Error (ECE)
- ECE = Σ |confidence - accuracy| over bins
- Well-calibrated: ECE < 0.10
- Poorly-calibrated: ECE > 0.20
```

**Success Criteria:**

1. Significant interaction: Strategy × Calibration (p < 0.05)
2. Weighted > Majority when ECE < 0.10
3. Weighted ≤ Majority when ECE > 0.20
4. Effect size η² ≥ 0.06 (medium effect)

**Expected Results:**

```
Well-Calibrated Models (ECE < 0.10):
- Majority voting: 94% accuracy
- Weighted voting: 97% accuracy (+3%)

Poorly-Calibrated Models (ECE > 0.20):
- Majority voting: 94% accuracy
- Weighted voting: 92% accuracy (-2%, overconfident models dominate)

Interaction: Weighted benefits depend on calibration quality
```

**Model Calibration:**

```elixir
# Measure Expected Calibration Error
defmodule Calibration do
  def calculate_ece(predictions, n_bins \\ 10) do
    # predictions: [{confidence, correct?}, ...]

    # Bin predictions by confidence
    bins = bin_by_confidence(predictions, n_bins)

    # For each bin: |avg_confidence - accuracy|
    errors =
      Enum.map(bins, fn bin ->
        avg_conf = Enum.map(bin, & &1.confidence) |> mean()
        accuracy = Enum.count(bin, & &1.correct) / length(bin)
        bin_weight = length(bin) / length(predictions)
        abs(avg_conf - accuracy) * bin_weight
      end)

    Enum.sum(errors)
  end
end

# Well-calibrated: ECE ≈ 0.05
# Poorly-calibrated: ECE ≈ 0.30
```

### H6: Transparency and Trust (Exploratory)

**Hypothesis:**

> Providing causal decision traces increases user trust scores by ≥1.0 points (7-point Likert scale) and reduces debugging time by ≥30%.

**Type:** Exploratory user study

**Design:** Between-subjects randomized controlled trial
- **Control:** LLM output only (no trace)
- **Treatment:** LLM output + causal trace
- **Participants:** n=40 (20 per group)
- **Tasks:** 5 debugging scenarios, 5 trust assessment scenarios
- **Measurements:** Trust scores (Likert), debugging time (seconds), success rate

**Statistical Analysis:**

```
Primary Analysis: Independent samples t-test
- DV1: Trust score (7-point Likert)
- DV2: Debugging time (log-transformed)
- IV: Condition (control vs treatment)
- Effect size: Cohen's d

Secondary Analysis: Success rate
- DV: Binary (success/failure)
- Statistical test: Chi-square test or Fisher's exact
- Effect size: Odds ratio

Covariates:
- Participant programming experience
- Prior LLM usage
- Analysis: ANCOVA if covariates significant

Non-parametric Alternatives:
- Trust scores may not be interval scale
- Fallback: Mann-Whitney U test
- Effect size: r = Z / √N
```

**Success Criteria:**

1. Trust score difference ≥1.0 (p < 0.05)
2. Debugging time reduction ≥30% (p < 0.05)
3. Success rate ≥10 percentage points higher
4. Medium-to-large effect sizes (d ≥ 0.5)

**Expected Results:**

```
Control Group (no trace):
- Trust score: 4.2 ± 1.5 (moderate trust)
- Debugging time: 180 ± 60 seconds
- Success rate: 65%

Treatment Group (with trace):
- Trust score: 5.5 ± 1.2 (higher trust) [+1.3 points]
- Debugging time: 110 ± 40 seconds [39% faster]
- Success rate: 80% [+15 percentage points]
```

**Trust Scale (7-point Likert):**

```
1. Strongly distrust
2. Distrust
3. Somewhat distrust
4. Neither trust nor distrust
5. Somewhat trust
6. Trust
7. Strongly trust

Questions:
- "I trust the AI's decision-making process"
- "I understand why the AI made this choice"
- "I would rely on this AI for important tasks"
- "The AI's reasoning seems reliable"
(Average of 4 items)
```

---

## Experimental Design

### Design Principles

**1. Randomization**

All experiments use proper randomization to prevent bias:

```elixir
# Randomize query order
defmodule Experiment do
  def randomize_queries(queries, seed) do
    :rand.seed(:exsplus, {seed, seed * 2, seed * 3})
    Enum.shuffle(queries)
  end
end

# Randomize condition order (counterbalancing)
defmodule Experiment do
  def counterbalance_conditions(conditions, participant_id) do
    # Use participant ID to determine order
    # Ensures balanced order across participants
    case rem(participant_id, 2) do
      0 -> conditions
      1 -> Enum.reverse(conditions)
    end
  end
end
```

**2. Blocking**

Use blocking to reduce variance:

```elixir
# Block by dataset difficulty
defmodule Experiment do
  def stratified_assignment(queries) do
    queries
    |> Enum.group_by(& &1.difficulty)  # Easy, Medium, Hard
    |> Enum.flat_map(fn {_difficulty, group} ->
      # Randomize within block
      Enum.shuffle(group)
    end)
  end
end
```

**3. Blinding**

Where possible, use single or double blinding:

- **Single blind:** Participants don't know condition (user studies)
- **Double blind:** Neither participants nor experimenters know (not applicable to automated experiments)
- **Automated experiments:** Blinding not needed (objective measurements)

**4. Replication**

All experiments include multiple repetitions:

```elixir
defmodule Experiment do
  use ResearchHarness.Experiment

  # Independent repetitions
  repeat 3

  # Each repetition uses different seed
  # seed: 42, 43, 44

  # Ensures results not due to random chance
end
```

### Common Design Patterns

#### Pattern 1: Independent Groups Design

**Use:** Compare different systems/models

```
Group A: Single model
Group B: 3-model ensemble
Group C: 5-model ensemble

Analysis: One-way ANOVA or Kruskal-Wallis
```

**Advantages:**
- No carryover effects
- Simple analysis

**Disadvantages:**
- Requires more samples
- Between-subjects variance

#### Pattern 2: Repeated Measures Design

**Use:** Compare strategies on same queries

```
Each query processed by:
- Baseline (no hedging)
- Hedging (P95 delay)
- Hedging (P90 delay)

Analysis: Repeated measures ANOVA or Friedman test
```

**Advantages:**
- Higher power (within-subjects)
- Controls for query difficulty

**Disadvantages:**
- Order effects (use counterbalancing)
- Sphericity assumptions

#### Pattern 3: Factorial Design

**Use:** Study multiple factors and interactions

```
2×2×2 Design:
- Factor A: Ensemble size (3 vs 5)
- Factor B: Voting (majority vs weighted)
- Factor C: Model tier (cheap vs expensive)

8 cells total, n=50 per cell = 400 queries
```

**Analysis:**
```
Three-way ANOVA:
- Main effects: A, B, C
- Two-way interactions: A×B, A×C, B×C
- Three-way interaction: A×B×C

Effect size: Partial η² for each effect
```

**Advantages:**
- Study interactions
- Efficient (one experiment, multiple questions)

**Disadvantages:**
- Requires large sample sizes
- Complex interpretation

#### Pattern 4: Dose-Response Design

**Use:** Find optimal parameter values

```
Test ensemble sizes: 1, 2, 3, 5, 7, 10
Measure: Accuracy, Cost, Latency

Analysis: Polynomial regression or curve fitting
```

**Advantages:**
- Identifies optimal value
- Tests nonlinear relationships

**Disadvantages:**
- Requires many levels
- Assumption of monotonic relationship

#### Pattern 5: Time Series Design

**Use:** Study adaptation/learning over time

```
Measure P99 latency every 100 queries for 2000 queries
Compare: Static vs Adaptive strategy

Analysis: Growth curve modeling or ARIMA
```

**Advantages:**
- Studies temporal dynamics
- Identifies learning curves

**Disadvantages:**
- Autocorrelation in data
- Complex statistical models

### Sample Size Determination

Use power analysis to determine required sample size:

```elixir
defmodule PowerAnalysis do
  @doc """
  Calculate required sample size for t-test.

  ## Parameters
  - effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
  - alpha: Significance level (typically 0.05)
  - power: Desired power (typically 0.80)
  - alternative: :two_sided or :one_sided

  ## Examples

      iex> PowerAnalysis.t_test_n(effect_size: 0.5, alpha: 0.05, power: 0.80)
      {:ok, %{n_per_group: 64}}

      iex> PowerAnalysis.t_test_n(effect_size: 0.8, alpha: 0.05, power: 0.80)
      {:ok, %{n_per_group: 26}}
  """
  def t_test_n(opts) do
    effect_size = Keyword.fetch!(opts, :effect_size)
    alpha = Keyword.get(opts, :alpha, 0.05)
    power = Keyword.get(opts, :power, 0.80)

    # Approximate formula for two-sample t-test
    # n ≈ 2 * (z_α + z_β)² / d²
    z_alpha = Statistics.Distributions.Normal.quantile(1 - alpha / 2, 0, 1)
    z_beta = Statistics.Distributions.Normal.quantile(power, 0, 1)

    n = 2 * :math.pow(z_alpha + z_beta, 2) / :math.pow(effect_size, 2)
    n_rounded = ceil(n)

    {:ok, %{n_per_group: n_rounded}}
  end
end
```

**Sample Size Table:**

| Effect Size (d) | α = 0.05, 1-β = 0.80 | α = 0.01, 1-β = 0.90 |
|-----------------|----------------------|----------------------|
| 0.2 (small)     | 393 per group        | 703 per group        |
| 0.5 (medium)    | 64 per group         | 105 per group        |
| 0.8 (large)     | 26 per group         | 40 per group         |
| 1.0 (very large)| 17 per group         | 26 per group         |

**Practical Guidelines:**

- **Pilot studies:** n=20-30 per group (detect large effects)
- **Confirmatory studies:** n=50-100 per group (detect medium effects)
- **High-stakes studies:** n=200+ per group (detect small effects)

---

## Statistical Methods

### Parametric Tests

#### Student's t-test

**Use:** Compare means of two independent groups

**Assumptions:**
1. Independence of observations
2. Normality (Shapiro-Wilk test)
3. Equal variances (Levene's test)

**Formula:**
```
t = (M₁ - M₂) / SE
SE = √(s²₁/n₁ + s²₂/n₂)  [pooled variance version]

df = n₁ + n₂ - 2
```

**Example:**
```elixir
control = [0.89, 0.87, 0.90, 0.88, 0.91]
treatment = [0.95, 0.97, 0.94, 0.96, 0.98]

result = Bench.Stats.TTest.independent(control, treatment)
# => %{
#   t_statistic: 8.45,
#   df: 8,
#   p_value: 0.00001,
#   mean_diff: 0.07,
#   ci: {0.048, 0.092}
# }
```

#### Welch's t-test

**Use:** Like Student's t-test but doesn't assume equal variances

**More robust:** Use by default unless you know variances are equal

**Formula:**
```
t = (M₁ - M₂) / SE
SE = √(s²₁/n₁ + s²₂/n₂)  [no pooling]

df = (s²₁/n₁ + s²₂/n₂)² / [(s²₁/n₁)²/(n₁-1) + (s²₂/n₂)²/(n₂-1)]  [Welch-Satterthwaite]
```

#### Paired t-test

**Use:** Compare two related measurements (same subjects)

**Assumptions:**
1. Paired observations
2. Differences normally distributed

**Formula:**
```
t = d̄ / (sd / √n)
where d̄ = mean of differences, sd = standard deviation of differences

df = n - 1
```

**Example:**
```elixir
before = [0.72, 0.68, 0.75, 0.71, 0.69]
after = [0.78, 0.73, 0.81, 0.76, 0.74]

result = Bench.Stats.PairedTTest.test(before, after)
# => %{
#   t_statistic: 3.24,
#   df: 4,
#   p_value: 0.031,
#   mean_diff: 0.06,
#   ci: {0.008, 0.112}
# }
```

#### One-way ANOVA

**Use:** Compare means of 3+ independent groups

**Assumptions:**
1. Independence
2. Normality in each group
3. Homogeneity of variances (Levene's test)

**Formula:**
```
F = MSbetween / MSwithin
MSbetween = SSbetween / (k - 1)
MSwithin = SSwithin / (N - k)

df1 = k - 1 (between groups)
df2 = N - k (within groups)
```

**Post-hoc:** Tukey HSD for pairwise comparisons

**Example:**
```elixir
group1 = [0.89, 0.87, 0.90]  # Single model
group2 = [0.94, 0.92, 0.95]  # 3-model ensemble
group3 = [0.97, 0.96, 0.98]  # 5-model ensemble

result = Bench.Stats.ANOVA.one_way([group1, group2, group3])
# => %{
#   f_statistic: 12.45,
#   df_between: 2,
#   df_within: 6,
#   p_value: 0.008,
#   eta_squared: 0.806  [large effect]
# }
```

### Non-Parametric Tests

**Use when:** Assumptions of parametric tests violated

#### Mann-Whitney U Test

**Use:** Non-parametric alternative to independent t-test

**No assumptions:** Works with any distribution, robust to outliers

**Formula:**
```
U = n₁n₂ + n₁(n₁+1)/2 - R₁
where R₁ = sum of ranks for group 1

Z = (U - μᵤ) / σᵤ  [for large samples, n > 20]
```

**Example:**
```elixir
control = [0.89, 0.87, 0.90, 0.88, 0.91, 0.75]  # Outlier!
treatment = [0.95, 0.97, 0.94, 0.96, 0.98]

# Outlier violates normality, use Mann-Whitney
result = Bench.Stats.MannWhitney.test(control, treatment)
# => %{
#   u_statistic: 2.0,
#   p_value: 0.015,
#   effect_size_r: 0.73  [large effect]
# }
```

#### Wilcoxon Signed-Rank Test

**Use:** Non-parametric alternative to paired t-test

**Example:**
```elixir
before = [0.72, 0.68, 0.75, 0.71, 0.69]
after = [0.78, 0.73, 0.81, 0.76, 0.74]

result = Bench.Stats.Wilcoxon.test(before, after)
# => %{
#   w_statistic: 15.0,
#   p_value: 0.043,
#   effect_size_r: 0.90
# }
```

#### Kruskal-Wallis H Test

**Use:** Non-parametric alternative to one-way ANOVA

**Example:**
```elixir
group1 = [0.89, 0.87, 0.90]
group2 = [0.94, 0.92, 0.95]
group3 = [0.97, 0.96, 0.98]

result = Bench.Stats.KruskalWallis.test([group1, group2, group3])
# => %{
#   h_statistic: 8.12,
#   df: 2,
#   p_value: 0.017,
#   epsilon_squared: 0.75  [large effect]
# }
```

### Effect Sizes

**Always report effect sizes!** P-values only tell if an effect exists, not how large.

#### Cohen's d

**Use:** Standardized mean difference

**Formula:**
```
d = (M₁ - M₂) / SDpooled

SDpooled = √[(s₁² + s₂²) / 2]  [for equal n]
SDpooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]  [for unequal n]
```

**Interpretation:**
- d = 0.2: small effect
- d = 0.5: medium effect
- d = 0.8: large effect
- d = 1.3+: very large effect

**Example:**
```elixir
control = [0.89, 0.87, 0.90, 0.88, 0.91]
treatment = [0.95, 0.97, 0.94, 0.96, 0.98]

effect = Bench.Stats.EffectSize.cohens_d(control, treatment)
# => %{
#   cohens_d: 4.52,
#   interpretation: "very large effect",
#   ci: {2.8, 6.2}  [95% CI]
# }
```

#### Partial Eta-Squared (η²)

**Use:** Effect size for ANOVA

**Formula:**
```
η²partial = SSeffect / (SSeffect + SSerror)
```

**Interpretation:**
- η² = 0.01: small effect
- η² = 0.06: medium effect
- η² = 0.14: large effect

#### Odds Ratio (OR)

**Use:** Effect size for binary outcomes

**Formula:**
```
OR = (success₁/failure₁) / (success₂/failure₂)
```

**Interpretation:**
- OR = 1: No effect
- OR = 2: 2× higher odds
- OR = 0.5: 50% lower odds

### Multiple Comparison Correction

**Problem:** Testing multiple hypotheses increases false positives

**Family-Wise Error Rate (FWER):**
```
P(at least one false positive) = 1 - (1 - α)ᵐ
where m = number of tests

Example: 20 tests at α=0.05
FWER = 1 - 0.95²⁰ = 0.64  [64% chance of false positive!]
```

#### Bonferroni Correction

**Most conservative:** Divide α by number of tests

```
α' = α / m

Example: 20 tests, α = 0.05
α' = 0.05 / 20 = 0.0025
```

**Use:** When tests are independent and FWER control is critical

**Example:**
```elixir
p_values = [0.001, 0.02, 0.045, 0.08, 0.15]
alpha = 0.05

corrected = Bench.Stats.correct_multiple_comparisons(p_values,
  method: :bonferroni, alpha: alpha
)
# => [
#   {0.001, true},   # 0.001 < 0.01 ✓
#   {0.02, false},   # 0.02 > 0.01 ✗
#   {0.045, false},
#   {0.08, false},
#   {0.15, false}
# ]
```

#### Holm-Bonferroni Correction

**Less conservative:** Step-down procedure

**Algorithm:**
```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Compare:
   - p₁ with α/m
   - p₂ with α/(m-1)
   - p₃ with α/(m-2)
   - ...
3. Stop at first non-significant
```

**More powerful than Bonferroni:** Rejects more false nulls

#### Benjamini-Hochberg (FDR)

**Controls False Discovery Rate:** Proportion of false positives among rejections

**Use:** When some false positives acceptable (exploratory research)

**Algorithm:**
```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest i where: pᵢ ≤ (i/m) × α
3. Reject H₁, H₂, ..., Hᵢ
```

**More powerful:** Especially with many tests

**Example:**
```elixir
p_values = [0.001, 0.02, 0.045, 0.08, 0.15]
alpha = 0.05

corrected = Bench.Stats.correct_multiple_comparisons(p_values,
  method: :benjamini_hochberg, alpha: alpha, fdr: 0.05
)
# => [
#   {0.001, true},   # 0.001 < (1/5)*0.05 = 0.01 ✓
#   {0.02, true},    # 0.02 < (2/5)*0.05 = 0.02 ✓
#   {0.045, true},   # 0.045 < (3/5)*0.05 = 0.03... actually false
#   {0.08, false},
#   {0.15, false}
# ]
```

### Confidence Intervals

**Always report CIs:** Quantify uncertainty around estimates

#### Analytical CI for Mean Difference

**Formula (independent t-test):**
```
CI = (M₁ - M₂) ± t* × SE
where:
  t* = critical value from t-distribution
  SE = √(s²₁/n₁ + s²₂/n₂)
```

**Example:**
```elixir
control = [0.89, 0.87, 0.90, 0.88, 0.91]  # M=0.89, s=0.015
treatment = [0.95, 0.97, 0.94, 0.96, 0.98]  # M=0.96, s=0.015

ci = Bench.Stats.ConfidenceInterval.mean_difference(control, treatment,
  confidence_level: 0.95
)
# => %{
#   estimate: 0.07,
#   ci: {0.048, 0.092},
#   confidence_level: 0.95,
#   se: 0.0095
# }

# Interpretation: We're 95% confident the true difference is between 4.8% and 9.2%
```

#### Bootstrap CI

**Use:** When analytical formula unavailable or assumptions violated

**Algorithm:**
```
1. Resample data with replacement (B times, e.g., B=10,000)
2. Calculate statistic for each resample
3. CI = [percentile(2.5), percentile(97.5)] for 95% CI
```

**Example:**
```elixir
data = [0.89, 0.87, 0.90, 0.88, 0.91]

ci = Bench.Stats.ConfidenceInterval.bootstrap(data, :median,
  confidence_level: 0.95,
  iterations: 10_000
)
# => %{
#   estimate: 0.89,
#   ci: {0.87, 0.91},
#   confidence_level: 0.95,
#   bootstrap_distribution: [...]  # 10k values
# }
```

---

## Datasets and Benchmarks

### Primary Datasets

#### MMLU (Massive Multitask Language Understanding)

**Description:** 57 subjects spanning STEM, humanities, social sciences

**Size:** 15,908 questions (test set)

**Format:** Multiple choice (4 options)

**Subsets:**
- **MMLU-STEM:** 2,563 questions (physics, chemistry, biology, math, CS)
- **MMLU-Humanities:** 4,399 questions (history, philosophy, law)
- **MMLU-Social:** 3,784 questions (psychology, economics, sociology)
- **MMLU-Other:** 5,162 questions (miscellaneous)

**License:** MIT

**Citation:**
```bibtex
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

**Usage:**
```elixir
# Load full MMLU
{:ok, dataset} = DatasetManager.load(:mmlu)

# Load STEM subset
{:ok, dataset} = DatasetManager.load(:mmlu_stem, sample_size: 200)

# Load with stratification
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 200,
  stratify_by: :subject
)
```

**Evaluation Metric:** Exact match (0 or 1)

#### HumanEval

**Description:** 164 programming problems for code generation

**Format:** Function signature + docstring + test cases

**Language:** Python

**Difficulty:** Entry-level to intermediate

**License:** MIT

**Citation:**
```bibtex
@article{chen2021evaluating,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}
```

**Usage:**
```elixir
{:ok, dataset} = DatasetManager.load(:humaneval)

# Generate code
{:ok, code} = MyModel.generate(dataset.items[0].prompt)

# Evaluate (runs test cases)
{:ok, result} = DatasetManager.evaluate([
  %{id: dataset.items[0].id, generated_code: code}
], dataset: dataset)
```

**Evaluation Metric:** pass@k (percentage passing all test cases)

#### GSM8K

**Description:** 8,500 grade school math word problems

**Format:** Natural language question + numerical answer

**Difficulty:** Grade school level (ages 7-12)

**Features:** Multi-step reasoning required

**License:** MIT

**Citation:**
```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**Usage:**
```elixir
{:ok, dataset} = DatasetManager.load(:gsm8k, sample_size: 100)

# Evaluate
{:ok, result} = DatasetManager.evaluate(predictions,
  dataset: dataset,
  metrics: [:exact_match, :numeric_match]
)
```

**Evaluation Metrics:**
- Exact match (string)
- Numeric match (parse and compare numbers)

### Custom Datasets

**Format:** JSONL (one JSON object per line)

```jsonl
{"id": "custom_1", "input": "Question 1?", "expected": "Answer 1", "metadata": {"difficulty": "easy"}}
{"id": "custom_2", "input": "Question 2?", "expected": "Answer 2", "metadata": {"difficulty": "hard"}}
```

**Loading:**
```elixir
{:ok, dataset} = DatasetManager.load("my_dataset",
  source: "path/to/data.jsonl",
  id_field: "id",
  input_field: "input",
  expected_field: "expected"
)
```

### Dataset Selection Guidelines

| Research Question | Recommended Dataset | Sample Size | Rationale |
|-------------------|---------------------|-------------|-----------|
| General reliability | MMLU-STEM | 200-500 | Broad coverage, moderate difficulty |
| Code generation | HumanEval | 164 (full) | Standard benchmark, objective evaluation |
| Reasoning | GSM8K | 100-200 | Multi-step reasoning, numerical verification |
| Domain-specific | Custom | Varies | Domain expertise required |

---

## Metrics and Measurements

### Accuracy Metrics

#### Exact Match

**Definition:** Prediction exactly equals expected answer (case-insensitive, whitespace-normalized)

**Formula:**
```
accuracy = (# exact matches) / (# total queries)
```

**Use:** Multiple choice, short answers

**Example:**
```elixir
predictions = [
  %{predicted: "Paris", expected: "Paris"},      # match
  %{predicted: "paris", expected: "Paris"},      # match (case-insensitive)
  %{predicted: "Paris ", expected: "Paris"},     # match (whitespace)
  %{predicted: "London", expected: "Paris"}      # no match
]

accuracy = 3 / 4 = 0.75
```

#### F1 Score

**Definition:** Harmonic mean of precision and recall

**Formula:**
```
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 × (precision × recall) / (precision + recall)
```

**Use:** When false positives and false negatives have different costs

**Example:**
```elixir
# Token-level F1 for longer answers
predicted = "The capital of France is Paris"
expected = "Paris, France"

# Tokenize: {"The", "capital", "of", "France", "is", "Paris"} vs {"Paris", "France"}
# TP = {"Paris", "France"} = 2
# FP = {"The", "capital", "of", "is"} = 4
# FN = {} = 0

precision = 2 / (2 + 4) = 0.33
recall = 2 / (2 + 0) = 1.0
F1 = 2 × (0.33 × 1.0) / (0.33 + 1.0) = 0.50
```

### Latency Metrics

#### Percentiles

**Definition:** Value below which P% of observations fall

**Common percentiles:**
- **P50 (median):** Typical user experience
- **P95:** Worst 5% of users
- **P99:** Worst 1% of users
- **P99.9:** Worst 0.1% of users

**Why percentiles?** Mean is misleading with long-tail distributions.

**Example:**
```elixir
latencies = [100, 110, 105, 120, 115, 108, 5000]  # One outlier

mean = 938ms  # Misleading!
median = 110ms  # Typical experience
p95 = 5000ms  # Tail experience
p99 = 5000ms
```

**Calculation:**
```elixir
defmodule Metrics do
  def percentile(data, p) do
    sorted = Enum.sort(data)
    n = length(data)
    k = p * (n - 1)
    floor_k = floor(k)
    ceil_k = ceil(k)

    if floor_k == ceil_k do
      Enum.at(sorted, floor_k)
    else
      # Linear interpolation
      lower = Enum.at(sorted, floor_k)
      upper = Enum.at(sorted, ceil_k)
      lower + (k - floor_k) * (upper - lower)
    end
  end
end

Metrics.percentile([100, 110, 105, 120, 115], 0.95)
# => 119.0  (interpolated between 115 and 120)
```

### Cost Metrics

#### Cost Per Query

**Definition:** Total API cost divided by number of queries

**Formula:**
```
cost_per_query = Σ(tokens × price_per_token) / n_queries
```

**Example:**
```elixir
queries = 100
total_tokens = 50_000  # Input + output
price = 0.01 / 1000  # $0.01 per 1k tokens

cost_per_query = (50_000 × 0.01 / 1000) / 100 = $0.005
```

#### Cost-Accuracy Ratio

**Definition:** Cost to achieve 1% accuracy improvement

**Formula:**
```
cost_accuracy_ratio = Δcost / Δaccuracy
```

**Example:**
```elixir
baseline_cost = $0.01, baseline_accuracy = 0.89
ensemble_cost = $0.05, ensemble_accuracy = 0.97

Δcost = $0.04
Δaccuracy = 0.08 (8 percentage points)

cost_accuracy_ratio = $0.04 / 0.08 = $0.50 per percentage point
```

**Use:** Compare cost-effectiveness of different strategies

### Ensemble Metrics

#### Consensus

**Definition:** Agreement level among ensemble models

**Formula:**
```
consensus = (votes for winner) / (total votes)
```

**Interpretation:**
- consensus = 1.0: Unanimous
- consensus = 0.6: Majority but weak
- consensus = 0.33: No majority (3-way tie)

**Example:**
```elixir
votes = %{
  "Paris" => 4,
  "London" => 1
}

consensus = 4 / 5 = 0.80  # 80% agreement
```

#### Diversity

**Definition:** How different are the ensemble models?

**Formula (pairwise disagreement):**
```
diversity = (# pairwise disagreements) / (# pairs)
```

**Example:**
```elixir
predictions = ["Paris", "Paris", "Paris", "London", "London"]

# Pairs: 5 choose 2 = 10 pairs
# Disagreements: Paris-London pairs = 3×2 = 6

diversity = 6 / 10 = 0.60
```

**Ideal:** High diversity (models make different mistakes) but low error rate

### Reliability Metrics

#### Success Rate

**Definition:** Percentage of queries answered correctly

**Formula:**
```
success_rate = (# correct) / (# total)
```

**Complement:** Failure rate = 1 - success_rate

#### Mean Time Between Failures (MTBF)

**Definition:** Average queries until failure

**Formula:**
```
MTBF = (total queries) / (# failures)
```

**Example:**
```elixir
queries = 10_000
failures = 50

MTBF = 10_000 / 50 = 200 queries
```

**Interpretation:** On average, one failure every 200 queries

---

## Reproducibility Protocol

### Seed Management

**All randomness must be seeded:**

```elixir
defmodule Experiment do
  def run(seed \\ 42) do
    # Seed Erlang RNG
    :rand.seed(:exsplus, {seed, seed * 2, seed * 3})

    # Seed any other RNGs
    :random.seed(seed)

    # Now all randomness is deterministic
    shuffled = Enum.shuffle([1, 2, 3, 4, 5])
    # Same seed => same shuffle
  end
end
```

**Seed sources:**
- Query shuffling
- Dataset sampling
- Ensemble model order
- Tie-breaking in voting

### Version Tracking

**Record all versions:**

```yaml
experiment:
  id: exp_abc123
  timestamp: 2025-10-08T12:00:00Z

framework:
  elixir_version: 1.14.0
  otp_version: 25.0
  framework_version: 0.1.0

libraries:
  ensemble: 0.1.0
  hedging: 0.1.0
  bench: 0.1.0
  dataset_manager: 0.1.0

datasets:
  mmlu_stem:
    version: 1.0.0
    sha256: abc123...
    sample_size: 200
    seed: 42

models:
  gpt4:
    version: gpt-4-0613
    temperature: 0.7
    max_tokens: 1000
  claude:
    version: claude-3-opus-20240229
    temperature: 0.7
    max_tokens: 1000
```

### Environment Recording

**Capture system state:**

```elixir
defmodule Experiment.Environment do
  def capture do
    %{
      node: node(),
      hostname: :inet.gethostname() |> elem(1) |> to_string(),
      os: :os.type(),
      cores: :erlang.system_info(:logical_processors),
      memory: :erlang.memory(),
      beam_version: :erlang.system_info(:otp_release),
      elixir_version: System.version(),
      timestamp: DateTime.utc_now(),
      timezone: "UTC"
    }
  end
end
```

### Artifact Preservation

**Save everything needed to reproduce:**

```
results/exp_abc123/
├── config.json          # Experiment configuration
├── environment.json     # System environment
├── dataset.jsonl        # Exact dataset used (or SHA256)
├── results.csv          # Raw results
├── analysis.json        # Statistical analysis
├── report.md            # Human-readable report
├── code/                # Code snapshot
│   ├── experiment.ex
│   └── custom_module.ex
└── checkpoints/         # Intermediate checkpoints
    ├── checkpoint_50.json
    ├── checkpoint_100.json
    └── checkpoint_150.json
```

### Verification Protocol

**To verify reproduction:**

1. **Clone repository** at specific commit
2. **Load same framework version**
3. **Use same dataset version**
4. **Set same seed**
5. **Run experiment**
6. **Compare results** (should be bit-identical)

```bash
# Reproduce experiment
git clone https://github.com/user/elixir_ai_research.git
cd elixir_ai_research
git checkout abc123  # Specific commit

mix deps.get
mix compile

# Set same seed
export EXPERIMENT_SEED=42

# Run
mix run experiments/h1_ensemble_reliability.exs

# Compare results
diff results/exp_abc123/results.csv results/exp_xyz789/results.csv
# Should be identical
```

---

## Ethical Considerations

### Dataset Ethics

**MMLU, HumanEval, GSM8K:**
- Public datasets, permissive licenses
- No personal information
- No human subjects

**Custom datasets:**
- Ensure proper consent if using proprietary data
- Respect copyright and terms of service
- Consider privacy implications

### Environmental Impact

**Carbon footprint of experiments:**

```
Estimated CO2 per experiment:
- 1000 queries × 5 models × 0.001 kWh = 5 kWh
- 5 kWh × 0.5 kg CO2/kWh = 2.5 kg CO2

Equivalent to:
- Driving 6 miles in an average car
- One hour of laptop use
```

**Mitigation:**
- Use smaller samples for pilot studies
- Optimize experiments to reduce redundant queries
- Use carbon-neutral compute when possible

### Fairness and Bias

**Dataset bias:**
- MMLU skews toward Western knowledge
- HumanEval uses Python-specific idioms
- GSM8K uses US cultural references

**Mitigation:**
- Acknowledge limitations in papers
- Test on diverse datasets
- Report demographic breakdowns if available

**Model bias:**
- LLMs reflect training data biases
- Ensemble may amplify majority biases

**Mitigation:**
- Include diverse models in ensemble
- Test for bias using fairness metrics
- Report limitations

---

## Publication Guidelines

### Reporting Standards

**Follow CONSORT guidelines (adapted for AI research):**

**Title and Abstract:**
- Clearly state it's a randomized experiment
- Report primary results (p-value, effect size, CI)

**Introduction:**
- State hypotheses explicitly
- Justify sample size (power analysis)

**Methods:**
- Describe experiment design (factorial, repeated measures, etc.)
- Report all conditions
- Specify randomization method
- State primary and secondary outcomes
- Preregister if possible (OSF, AsPredicted)

**Results:**
- Report descriptive statistics (M, SD, n)
- Report test statistics (t, F, U, etc.)
- Report p-values (exact if p > 0.001)
- Report effect sizes with CIs
- Report multiple comparison corrections
- Include all planned analyses

**Discussion:**
- Relate to hypotheses
- Discuss limitations
- Avoid HARKing (Hypothesizing After Results Known)

**Reproducibility:**
- Share code on GitHub
- Share data or data access method
- Report software versions
- Include seed values

### Example Results Section

```markdown
## Results

### H1: Ensemble Reliability

We compared a 5-model ensemble (GPT-4, Claude Opus, Gemini Pro, GPT-3.5,
Claude Sonnet) with majority voting to a single GPT-4 baseline on 200
randomly sampled MMLU-STEM questions across 3 independent runs (seed=42, 43, 44).

**Descriptive Statistics:**

| Condition | M | SD | 95% CI | n |
|-----------|---|----|----|---|
| Baseline (GPT-4) | 0.89 | 0.02 | [0.87, 0.91] | 600 |
| Ensemble (5-model) | 0.97 | 0.01 | [0.96, 0.98] | 600 |

**Inferential Statistics:**

The ensemble achieved significantly higher accuracy than the baseline,
t(1142) = 8.45, p < .001, d = 4.52, 95% CI [0.06, 0.10]. This represents
a very large effect size. The lower bound of the 95% CI for ensemble
accuracy (0.96) exceeds our target of 0.95, supporting H1.

**Cost Analysis:**

The ensemble incurred 5× higher cost per query (M = $0.05, SD = $0.01)
compared to baseline (M = $0.01, SD = $0.002), t(1142) = 127.3, p < .001,
d = 8.21. The cost-accuracy ratio was $0.50 per percentage point improvement.

**Conclusion:**

H1 is supported. The 5-model ensemble achieved ≥99% accuracy target with
statistical significance and large effect size, albeit at 5× cost increase.
```

### Data Sharing

**Share publicly when possible:**

**GitHub:**
- Code repository
- Experiment scripts
- Analysis notebooks

**OSF (Open Science Framework):**
- Preregistration
- Data files
- Supplementary materials

**Zenodo:**
- Archived code (DOI)
- Archived data (DOI)

**Format:**
- CSV for tabular data
- JSONL for structured data
- Jupyter notebooks for analysis

### Citation Format

**Framework:**
```bibtex
@software{elixir_ai_research2025,
  author = {Research Infrastructure Team},
  title = {Elixir AI Research Framework},
  year = {2025},
  url = {https://github.com/user/elixir_ai_research},
  version = {0.1.0}
}
```

**Experiment:**
```bibtex
@misc{smith2025ensemble,
  author = {Smith, John and Doe, Jane},
  title = {Multi-Model Ensemble Reliability Experiment},
  year = {2025},
  howpublished = {Open Science Framework},
  url = {https://osf.io/abc123/},
  doi = {10.17605/OSF.IO/ABC123}
}
```

---

## Conclusion

This methodology provides a rigorous, reproducible framework for LLM reliability research. Key principles:

1. **Preregister hypotheses** to prevent HARKing
2. **Power analysis** to ensure adequate sample sizes
3. **Effect sizes** to quantify practical significance
4. **Multiple comparison correction** to control false positives
5. **Complete reproducibility** via seeds and version tracking
6. **Transparent reporting** following publication standards

**Next Steps:**

1. Review and adapt methodology for your research question
2. Conduct power analysis to determine sample size
3. Preregister study on OSF
4. Run pilot study (n=20-30)
5. Refine design based on pilot
6. Run full study
7. Analyze with statistical rigor
8. Publish with complete transparency

---

## Further Reading

- [GETTING_STARTED.md](./GETTING_STARTED.md) - Installation and first experiment
- [ENSEMBLE_GUIDE.md](./ENSEMBLE_GUIDE.md) - Ensemble strategies in depth
- [HEDGING_GUIDE.md](./HEDGING_GUIDE.md) - Request hedging theory
- [STATISTICAL_TESTING.md](./STATISTICAL_TESTING.md) - Using Bench
- [PUBLICATIONS.md](./PUBLICATIONS.md) - Publication templates

**References:**

1. Dean, J., & Barroso, L. A. (2013). The tail at scale. *Communications of the ACM*, 56(2), 74-80.
2. Hendrycks, D., et al. (2021). Measuring massive multitask language understanding. *ICLR*.
3. Chen, M., et al. (2021). Evaluating large language models trained on code. *arXiv preprint*.
4. Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Erlbaum.
5. Cumming, G. (2014). The new statistics: Why and how. *Psychological Science*, 25(1), 7-29.

---

**Document Status:** Complete
**Review Status:** Pending peer review
**Version:** 0.1.0
**Last Updated:** 2025-10-08
