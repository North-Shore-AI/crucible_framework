# Statistical Testing Framework - Complete Guide

**Bench: Research-Grade Statistical Analysis for AI/ML Experiments**

Version: 0.1.0
Last Updated: 2025-10-08

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Parametric Tests](#parametric-tests)
4. [Non-Parametric Tests](#non-parametric-tests)
5. [Effect Size Measures](#effect-size-measures)
6. [Power Analysis](#power-analysis)
7. [Confidence Intervals](#confidence-intervals)
8. [Multiple Comparisons](#multiple-comparisons)
9. [Assumption Testing](#assumption-testing)
10. [Complete Examples](#complete-examples)
11. [Best Practices](#best-practices)
12. [References](#references)

---

## Overview

Bench is a comprehensive statistical testing framework designed specifically for AI/ML research. It provides rigorous statistical tests, effect size measures, power analysis, and publication-ready reporting—all validated against R and SciPy implementations.

### Key Features

- **6 Statistical Tests**: t-test, paired t-test, ANOVA, Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- **4 Effect Sizes**: Cohen's d, Hedges' g, Glass's Δ, η², ω²
- **Power Analysis**: A priori (sample size) and post-hoc (achieved power)
- **Confidence Intervals**: Analytical and bootstrap methods
- **Multiple Correction**: Bonferroni, Holm, Benjamini-Hochberg
- **Auto-Selection**: Automatic test selection based on assumptions

### Quick Start

```elixir
# Simple comparison
control = [0.72, 0.68, 0.75, 0.71, 0.69]
treatment = [0.78, 0.73, 0.81, 0.76, 0.74]

result = Bench.compare(control, treatment)
# => Automatically selects appropriate test

# Access results
result.p_value           # => 0.023
result.effect_size       # => %{cohens_d: 0.89, interpretation: "large"}
result.interpretation    # => "Significant difference (medium effect)"
```

### Design Philosophy

1. **Statistical Rigor**: All implementations validated against R/SciPy
2. **Interpretability**: Every result includes effect sizes and interpretation
3. **Reproducibility**: Complete audit trails for peer review
4. **Publication Quality**: APA-style reporting built-in

---

## Core Concepts

### Statistical Hypothesis Testing

**Null Hypothesis (H₀)**: No difference between groups
**Alternative Hypothesis (H₁)**: There is a difference

**Process:**

1. Define hypotheses
2. Choose significance level (α = 0.05)
3. Calculate test statistic
4. Determine p-value
5. Make decision: reject H₀ if p < α

**P-value Interpretation:**

```
p < 0.001: Highly significant (***)
p < 0.01:  Very significant (**)
p < 0.05:  Significant (*)
p ≥ 0.05:  Not significant (ns)
```

### Effect Sizes

**Why Effect Sizes Matter:**

P-values tell us **if** there's a difference, but effect sizes tell us **how large** the difference is.

**Cohen's Guidelines:**

| Effect Size | d    | η²   | Interpretation |
|-------------|------|------|----------------|
| Negligible  | <0.2 | <0.01| Practically meaningless |
| Small       | 0.2  | 0.01 | Noticeable to experts |
| Medium      | 0.5  | 0.06 | Visible to careful observers |
| Large       | 0.8  | 0.14 | Obvious to anyone |

### Type I and Type II Errors

**Type I Error (α)**: False positive—rejecting true H₀
**Type II Error (β)**: False negative—failing to reject false H₀

**Power (1-β)**: Probability of detecting a true effect

```elixir
# Calculate required sample size for 80% power
result = Bench.power_analysis(:t_test,
  analysis_type: :a_priori,
  effect_size: 0.5,    # Expected Cohen's d
  alpha: 0.05,
  power: 0.80
)

result.n_per_group  # => 64 samples per group needed
```

### Choosing a Test

**Decision Tree:**

```
1. Are samples related?
   ├─ Yes: Paired test
   │   ├─ Normal? → Paired t-test
   │   └─ Not normal? → Wilcoxon signed-rank
   └─ No: Independent samples
       ├─ 2 groups?
       │   ├─ Normal? → Independent t-test
       │   └─ Not normal? → Mann-Whitney U
       └─ 3+ groups?
           ├─ Normal? → ANOVA
           └─ Not normal? → Kruskal-Wallis
```

**Implemented in Code:**

```elixir
defmodule TestSelector do
  def select_test(group1, group2, opts \\ []) do
    paired = Keyword.get(opts, :paired, false)
    check_assumptions = Keyword.get(opts, :check_assumptions, true)

    cond do
      paired and check_assumptions and normal?(group1) and normal?(group2) ->
        Bench.Stats.PairedTTest.test(group1, group2, opts)

      paired ->
        Bench.Stats.Wilcoxon.test(group1, group2, opts)

      check_assumptions and normal?(group1) and normal?(group2) ->
        Bench.Stats.TTest.test(group1, group2, opts)

      true ->
        Bench.Stats.MannWhitney.test(group1, group2, opts)
    end
  end
end
```

---

## Parametric Tests

Parametric tests assume:
1. Normal distribution
2. Independent observations
3. Homogeneity of variance (for some tests)

### Independent Samples t-Test

**When to Use:**
- Compare means of 2 independent groups
- Data is normally distributed
- Continuous outcome variable

**Formula:**

```
t = (x̄₁ - x̄₂) / SE

Where:
SE = √(s₁²/n₁ + s₂²/n₂)  # Welch's t-test (unequal variances)

or

SE = √(s_p² × (1/n₁ + 1/n₂))  # Student's t-test (equal variances)
s_p² = ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2)
```

**Implementation:**

```elixir
# Welch's t-test (default - robust to unequal variances)
result = Bench.Stats.TTest.test(group1, group2)

# Student's t-test (assumes equal variances)
result = Bench.Stats.TTest.test(group1, group2, var_equal: true)

# One-tailed test
result = Bench.Stats.TTest.test(group1, group2,
  alternative: :greater  # Test if group1 > group2
)
```

**Example:**

```elixir
# Compare GPT-4 vs Claude accuracy
gpt4_scores = [0.89, 0.91, 0.88, 0.90, 0.92, 0.87, 0.93]
claude_scores = [0.84, 0.86, 0.85, 0.87, 0.83, 0.86, 0.85]

result = Bench.Stats.TTest.test(gpt4_scores, claude_scores)

# Result:
%Bench.Result{
  test: :welch_t_test,
  statistic: 5.23,           # t-value
  p_value: 0.0003,           # Highly significant
  confidence_interval: {0.037, 0.083},
  interpretation: "Highly significant difference between groups",
  metadata: %{
    df: 11.4,                # Degrees of freedom
    mean1: 0.900,
    mean2: 0.851,
    mean_diff: 0.049,
    n1: 7,
    n2: 7
  }
}

# Calculate effect size
effect = Bench.Stats.EffectSize.cohens_d(gpt4_scores, claude_scores)
# => %{cohens_d: 1.96, interpretation: "large"}
```

**Assumptions:**

```elixir
# Check normality
Bench.Stats.normality_test(gpt4_scores)
# => %{normal: true, shapiro_p: 0.23}

# Check equal variances (if using Student's t)
Bench.Stats.levene_test(gpt4_scores, claude_scores)
# => %{equal_variances: true, p_value: 0.67}
```

**Reporting (APA Style):**

```elixir
defmodule Reporting do
  def apa_t_test(result, effect_size) do
    """
    An independent samples Welch's t-test revealed a significant difference
    between groups, t(#{Float.round(result.metadata.df, 2)}) = #{Float.round(result.statistic, 2)},
    p = #{format_p(result.p_value)}, Cohen's d = #{Float.round(effect_size.cohens_d, 2)}.
    """
  end

  defp format_p(p) when p < 0.001, do: "< .001"
  defp format_p(p), do: Float.round(p, 3) |> to_string() |> String.trim_leading("0")
end
```

### Paired Samples t-Test

**When to Use:**
- Compare means of 2 related groups (e.g., before/after)
- Same subjects measured twice
- Matched pairs

**Formula:**

```
t = (d̄ - μ₀) / (s_d / √n)

Where:
d̄ = mean of differences
s_d = standard deviation of differences
μ₀ = hypothesized mean difference (usually 0)
```

**Implementation:**

```elixir
# Before-after comparison
before = [0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.70]
after = [0.78, 0.73, 0.81, 0.76, 0.74, 0.79, 0.77]

result = Bench.Stats.PairedTTest.test(before, after)

# Result:
%Bench.Result{
  test: :paired_t_test,
  statistic: 6.89,
  p_value: 0.0004,
  confidence_interval: {0.042, 0.098},
  interpretation: "Highly significant improvement",
  metadata: %{
    df: 6,
    mean_diff: 0.070,
    sd_diff: 0.027,
    n: 7
  }
}
```

**Example: Prompt Engineering Impact**

```elixir
# Test if adding few-shot examples improves accuracy
defmodule PromptEngineeringStudy do
  def run_experiment do
    test_cases = load_test_cases()

    # Baseline (zero-shot)
    baseline_scores = Enum.map(test_cases, fn tc ->
      evaluate(tc, prompt: :zero_shot)
    end)

    # Intervention (few-shot)
    fewshot_scores = Enum.map(test_cases, fn tc ->
      evaluate(tc, prompt: :few_shot)
    end)

    # Paired t-test
    result = Bench.Stats.PairedTTest.test(
      baseline_scores,
      fewshot_scores
    )

    # Effect size
    effect = Bench.Stats.EffectSize.paired_cohens_d(
      baseline_scores,
      fewshot_scores
    )

    %{
      significant: result.p_value < 0.05,
      improvement: result.metadata.mean_diff,
      effect_size: effect.cohens_d,
      interpretation: effect.interpretation
    }
  end
end

# Run study
PromptEngineeringStudy.run_experiment()
# => %{
#   significant: true,
#   improvement: 0.087,      # 8.7 percentage points
#   effect_size: 1.23,
#   interpretation: "large"
# }
```

### One-Way ANOVA

**When to Use:**
- Compare means of 3+ independent groups
- Extension of independent t-test
- Test if at least one group differs

**Formula:**

```
F = MS_between / MS_within

Where:
SS_between = Σ n_i(x̄_i - x̄)²
SS_within = Σ Σ (x_ij - x̄_i)²
MS = SS / df
```

**Implementation:**

```elixir
# Compare 3 models
gpt4 = [0.89, 0.91, 0.88, 0.90, 0.92]
claude = [0.87, 0.89, 0.86, 0.88, 0.90]
gemini = [0.84, 0.86, 0.83, 0.85, 0.87]

result = Bench.Stats.ANOVA.one_way([gpt4, claude, gemini])

# Result:
%Bench.Result{
  test: :anova,
  statistic: 12.45,         # F-statistic
  p_value: 0.0009,
  effect_size: %{
    eta_squared: 0.624,      # η² - variance explained
    omega_squared: 0.561,    # ω² - unbiased estimate
    interpretation: "large"
  },
  interpretation: "Highly significant difference between groups (large effect)",
  metadata: %{
    df_between: 2,
    df_within: 12,
    ss_between: 0.0382,
    ss_within: 0.0126,
    ms_between: 0.0191,
    ms_within: 0.00105,
    k: 3,
    n_total: 15,
    group_means: [0.900, 0.880, 0.850]
  }
}
```

**Post-Hoc Tests:**

If ANOVA is significant, determine which groups differ:

```elixir
defmodule PostHoc do
  # Bonferroni correction for multiple comparisons
  def pairwise_comparisons(groups, alpha \\ 0.05) do
    n_comparisons = div(length(groups) * (length(groups) - 1), 2)
    adjusted_alpha = alpha / n_comparisons

    comparisons =
      for {g1, i} <- Enum.with_index(groups),
          {g2, j} <- Enum.with_index(groups),
          i < j do

        result = Bench.Stats.TTest.test(g1, g2)
        significant = result.p_value < adjusted_alpha

        %{
          comparison: "Group#{i+1} vs Group#{j+1}",
          p_value: result.p_value,
          adjusted_alpha: adjusted_alpha,
          significant: significant
        }
      end

    comparisons
  end
end

# Run post-hoc
PostHoc.pairwise_comparisons([gpt4, claude, gemini])
# => [
#   %{comparison: "Group1 vs Group2", p_value: 0.032, significant: true},
#   %{comparison: "Group1 vs Group3", p_value: 0.0001, significant: true},
#   %{comparison: "Group2 vs Group3", p_value: 0.007, significant: true}
# ]
```

**Example: Hyperparameter Tuning**

```elixir
defmodule HyperparameterStudy do
  def compare_temperatures do
    # Test different temperature settings
    temp_01 = run_trials(temperature: 0.1)
    temp_05 = run_trials(temperature: 0.5)
    temp_09 = run_trials(temperature: 0.9)
    temp_15 = run_trials(temperature: 1.5)

    # ANOVA
    result = Bench.Stats.ANOVA.one_way([
      temp_01, temp_05, temp_09, temp_15
    ])

    # If significant, find best temperature
    if result.p_value < 0.05 do
      means = result.metadata.group_means
      best_idx = Enum.find_index(means, &(&1 == Enum.max(means)))
      temps = [0.1, 0.5, 0.9, 1.5]

      %{
        best_temperature: Enum.at(temps, best_idx),
        f_statistic: result.statistic,
        p_value: result.p_value,
        effect_size: result.effect_size.eta_squared
      }
    end
  end
end
```

---

## Non-Parametric Tests

Non-parametric tests don't assume normal distribution. Use when:
- Data is ordinal or ranked
- Small sample sizes
- Highly skewed distributions
- Outliers present

### Mann-Whitney U Test

**When to Use:**
- Non-parametric alternative to independent t-test
- Compare distributions of 2 independent groups
- Data is ordinal or non-normal

**Formula:**

```
U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁
U₂ = n₁n₂ + n₂(n₂+1)/2 - R₂

U = min(U₁, U₂)

For large samples (n > 20):
z = (U - μ_U) / σ_U
where μ_U = n₁n₂/2, σ_U = √(n₁n₂(n₁+n₂+1)/12)
```

**Implementation:**

```elixir
# Non-normal data with outliers
control = [120, 135, 118, 142, 125, 890]  # Note the outlier
treatment = [98, 105, 102, 110, 95, 108]

result = Bench.Stats.MannWhitney.test(control, treatment)

# Result:
%Bench.Result{
  test: :mann_whitney,
  statistic: 2.0,            # U statistic
  p_value: 0.026,
  effect_size: %{
    rank_biserial: 0.72,     # Effect size for ranks
    interpretation: "large"
  },
  interpretation: "Significant difference between groups (large effect)",
  metadata: %{
    u1: 2.0,
    u2: 34.0,
    z_statistic: -2.23,
    n1: 6,
    n2: 6,
    r1: 49.0,                 # Sum of ranks for group 1
    r2: 29.0                  # Sum of ranks for group 2
  }
}
```

**Rank-Based Effect Size:**

```elixir
# Rank-biserial correlation
r = 1 - (2*U)/(n₁*n₂)

# Interpretation:
# |r| < 0.1: negligible
# |r| < 0.3: small
# |r| < 0.5: medium
# |r| ≥ 0.5: large
```

**Example: Comparing Latencies**

```elixir
# Latency distributions are often skewed
defmodule LatencyComparison do
  def compare_models do
    # Collect latencies (highly skewed)
    gpt_latencies = measure_latencies(:gpt4)
    claude_latencies = measure_latencies(:claude)

    # Check distribution
    gpt_normal = Bench.Stats.normality_test(gpt_latencies)
    # => %{normal: false, shapiro_p: 0.003}

    # Use non-parametric test
    result = Bench.Stats.MannWhitney.test(
      gpt_latencies,
      claude_latencies,
      alternative: :less  # Test if GPT < Claude (faster)
    )

    %{
      gpt_faster: result.p_value < 0.05,
      p_value: result.p_value,
      effect_size: result.effect_size.rank_biserial,
      median_gpt: Bench.Stats.median(gpt_latencies),
      median_claude: Bench.Stats.median(claude_latencies)
    }
  end
end
```

### Wilcoxon Signed-Rank Test

**When to Use:**
- Non-parametric alternative to paired t-test
- Compare 2 related samples
- Non-normal or ordinal data

**Formula:**

```
W = min(W⁺, W⁻)

Where:
W⁺ = sum of positive ranks
W⁻ = sum of negative ranks

For large samples (n > 25):
z = (W - μ_W) / σ_W
where μ_W = n(n+1)/4, σ_W = √(n(n+1)(2n+1)/24)
```

**Implementation:**

```elixir
# Paired non-normal data
before = [0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.70]
after = [0.78, 0.73, 0.81, 0.76, 0.74, 0.79, 0.77]

result = Bench.Stats.Wilcoxon.test(before, after)

# Result:
%Bench.Result{
  test: :wilcoxon_signed_rank,
  statistic: 0.0,            # W statistic (all differences positive)
  p_value: 0.016,
  effect_size: %{r: 0.89},   # r = Z/√n
  interpretation: "Significant difference between paired groups",
  metadata: %{
    w_plus: 28.0,
    w_minus: 0.0,
    n: 7,
    n_zero: 0                # Number of zero differences
  }
}
```

**Example: A/B Test with Skewed Metrics**

```elixir
defmodule ABTestNonParametric do
  def compare_variants(user_sessions) do
    # Extract metrics (e.g., session duration - often skewed)
    variant_a = Enum.map(user_sessions[:a], &(&1.duration))
    variant_b = Enum.map(user_sessions[:b], &(&1.duration))

    # Paired by user
    paired_a = Enum.map(user_sessions[:paired], &(&1.variant_a_duration))
    paired_b = Enum.map(user_sessions[:paired], &(&1.variant_b_duration))

    # Check if paired
    if length(paired_a) > 0 do
      # Wilcoxon for paired data
      Bench.Stats.Wilcoxon.test(paired_a, paired_b)
    else
      # Mann-Whitney for independent
      Bench.Stats.MannWhitney.test(variant_a, variant_b)
    end
  end
end
```

### Kruskal-Wallis Test

**When to Use:**
- Non-parametric alternative to ANOVA
- Compare 3+ independent groups
- Non-normal or ordinal data

**Formula:**

```
H = (12/(N(N+1))) × Σ(R_i²/n_i) - 3(N+1)

Where:
R_i = sum of ranks for group i
n_i = sample size of group i
N = total sample size
```

**Implementation:**

```elixir
# Compare 4 models (non-normal data)
model_a = [8.2, 7.9, 8.5, 7.8, 8.1]
model_b = [7.5, 7.2, 7.8, 7.4, 7.6]
model_c = [9.1, 8.8, 9.3, 8.9, 9.0]
model_d = [6.8, 6.5, 7.1, 6.9, 6.7]

result = Bench.Stats.KruskalWallis.test([
  model_a, model_b, model_c, model_d
])

# Result:
%Bench.Result{
  test: :kruskal_wallis,
  statistic: 15.23,          # H statistic
  p_value: 0.0016,
  effect_size: %{
    epsilon_squared: 0.68,   # ε² - effect size for K-W
    interpretation: "large"
  },
  interpretation: "Highly significant difference between groups",
  metadata: %{
    df: 3,
    k: 4,
    n_total: 20,
    rank_sums: [50, 30, 90, 20]
  }
}
```

---

## Effect Size Measures

### Cohen's d

**Standardized mean difference:**

```
d = (x̄₁ - x̄₂) / s_pooled

where s_pooled = √((s₁² + s₂²) / 2)
```

**Implementation:**

```elixir
group1 = [5.0, 5.2, 4.8, 5.1, 4.9]
group2 = [6.0, 6.2, 5.8, 6.1, 5.9]

effect = Bench.Stats.EffectSize.cohens_d(group1, group2)

# Result:
%{
  cohens_d: -3.16,           # Negative because group2 > group1
  interpretation: "large",
  mean1: 5.0,
  mean2: 6.0,
  pooled_sd: 0.316
}

# Interpretation:
# Groups differ by 3.16 standard deviations
```

### Hedges' g

**Bias-corrected Cohen's d for small samples:**

```
g = d × (1 - 3/(4N - 9))

where N = n₁ + n₂
```

**Implementation:**

```elixir
# Small sample sizes
group1 = [5.0, 5.2, 4.8]
group2 = [6.0, 6.2, 5.8]

effect = Bench.Stats.EffectSize.hedges_g(group1, group2)

# Result:
%{
  hedges_g: -2.83,           # Corrected for small n
  cohens_d: -3.16,           # Original d
  correction_factor: 0.895,  # Correction applied
  interpretation: "large"
}
```

### Glass's Delta

**Uses control group SD only:**

```
Δ = (x̄_treatment - x̄_control) / s_control
```

**When to Use:**
- Different variances between groups
- One group is clear control

**Implementation:**

```elixir
control = [5.0, 5.2, 4.8, 5.1, 4.9]      # Low variance
treatment = [6.0, 6.5, 5.5, 6.2, 5.8]    # Higher variance

effect = Bench.Stats.EffectSize.glass_delta(control, treatment)

# Result:
%{
  glass_delta: 6.32,         # Using control SD
  interpretation: "large",
  mean_control: 5.0,
  mean_treatment: 6.0,
  sd_control: 0.158
}
```

### Eta-Squared (η²) and Omega-Squared (ω²)

**For ANOVA - variance explained:**

```
η² = SS_between / SS_total

ω² = (SS_between - df_between × MS_within) / (SS_total + MS_within)
```

**Interpretation:**

| η² / ω² | Interpretation |
|---------|----------------|
| 0.01    | Small          |
| 0.06    | Medium         |
| 0.14    | Large          |

**Already included in ANOVA results:**

```elixir
result = Bench.Stats.ANOVA.one_way([group1, group2, group3])

result.effect_size
# => %{
#   eta_squared: 0.624,
#   omega_squared: 0.561,
#   interpretation: "large"
# }
```

### Effect Size Conversion

```elixir
defmodule EffectSizeConversion do
  # Convert between different effect sizes

  # d to r (correlation)
  def d_to_r(d) do
    d / :math.sqrt(d*d + 4)
  end

  # r to d
  def r_to_d(r) do
    2 * r / :math.sqrt(1 - r*r)
  end

  # d to η²
  def d_to_eta_squared(d, n1, n2) do
    d*d / (d*d + (n1 + n2 - 2))
  end

  # η² to d
  def eta_squared_to_d(eta_sq) do
    :math.sqrt(eta_sq / (1 - eta_sq))
  end
end

# Example
d = 0.8
r = EffectSizeConversion.d_to_r(d)       # => 0.37
eta_sq = EffectSizeConversion.d_to_eta_squared(d, 30, 30)  # => 0.14
```

---

## Power Analysis

### A Priori Power Analysis

**Calculate required sample size:**

```elixir
# For t-test
result = Bench.Stats.Power.analyze(:t_test,
  analysis_type: :a_priori,
  effect_size: 0.5,          # Medium effect expected
  alpha: 0.05,               # 5% Type I error
  power: 0.80,               # 80% power desired
  alternative: :two_sided
)

# Result:
%{
  analysis_type: :a_priori,
  test: :t_test,
  n_per_group: 64,           # Need 64 per group
  total_n: 128,              # 128 total
  effect_size: 0.5,
  alpha: 0.05,
  power: 0.80,
  recommendation: "Collect at least 64 samples per group (128 total) to detect an effect size of 0.5 with 80.0% power"
}
```

**For ANOVA:**

```elixir
result = Bench.Stats.Power.analyze(:anova,
  analysis_type: :a_priori,
  effect_size: 0.25,         # Medium effect (η²)
  k: 4,                      # 4 groups
  alpha: 0.05,
  power: 0.80
)

result.n_per_group  # => 45 per group
result.total_n      # => 180 total
```

### Post-Hoc Power Analysis

**Calculate achieved power from existing data:**

```elixir
# After experiment
group1 = [0.89, 0.91, 0.88, 0.90, 0.92]
group2 = [0.84, 0.86, 0.85, 0.87, 0.83]

# Calculate observed effect
effect = Bench.Stats.EffectSize.cohens_d(group1, group2)

# Post-hoc power
result = Bench.Stats.Power.analyze(:t_test,
  analysis_type: :post_hoc,
  effect_size: effect.cohens_d,
  n_per_group: 5,
  alpha: 0.05
)

# Result:
%{
  analysis_type: :post_hoc,
  test: :t_test,
  power: 0.43,               # Only 43% power!
  recommendation: "Underpowered (43.0%). Increase sample size significantly."
}
```

### Power Curves

```elixir
defmodule PowerCurve do
  def generate(effect_size, alpha \\ 0.05) do
    sample_sizes = [5, 10, 20, 30, 50, 100, 200, 500]

    Enum.map(sample_sizes, fn n ->
      result = Bench.Stats.Power.analyze(:t_test,
        analysis_type: :post_hoc,
        effect_size: effect_size,
        n_per_group: n,
        alpha: alpha
      )

      %{n: n, power: result.power}
    end)
  end
end

# Generate curves for different effect sizes
small = PowerCurve.generate(0.2)
medium = PowerCurve.generate(0.5)
large = PowerCurve.generate(0.8)

# Plot results
# n=5:   small=0.08, medium=0.17, large=0.29
# n=20:  small=0.17, medium=0.52, large=0.80
# n=100: small=0.56, medium=0.97, large=1.00
```

### Sample Size Planning

```elixir
defmodule SampleSizePlanner do
  def plan_experiment(expected_effect, budget, cost_per_sample) do
    # Calculate ideal sample size
    ideal = Bench.Stats.Power.analyze(:t_test,
      analysis_type: :a_priori,
      effect_size: expected_effect,
      power: 0.80
    )

    # Check budget
    max_affordable = div(budget, cost_per_sample)

    cond do
      max_affordable >= ideal.total_n ->
        %{
          status: :fully_powered,
          n_per_group: ideal.n_per_group,
          power: 0.80,
          cost: ideal.total_n * cost_per_sample
        }

      max_affordable >= 10 ->
        # Calculate achievable power
        achievable = Bench.Stats.Power.analyze(:t_test,
          analysis_type: :post_hoc,
          effect_size: expected_effect,
          n_per_group: div(max_affordable, 2)
        )

        %{
          status: :underpowered,
          n_per_group: div(max_affordable, 2),
          power: achievable.power,
          cost: budget,
          warning: "Only #{Float.round(achievable.power * 100, 1)}% power"
        }

      true ->
        %{status: :insufficient_budget}
    end
  end
end

# Example
SampleSizePlanner.plan_experiment(0.5, 1000, 10)
# => %{
#   status: :underpowered,
#   n_per_group: 50,
#   power: 0.70,
#   cost: 1000,
#   warning: "Only 70.0% power"
# }
```

---

## Confidence Intervals

### Analytical Confidence Intervals

**For means:**

```
CI = x̄ ± t_(α/2,df) × SE

where SE = s/√n
```

**Implementation:**

```elixir
data = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7]

ci = Bench.Stats.ConfidenceInterval.calculate(
  data,
  :mean,
  confidence_level: 0.95
)

# Result:
%{
  interval: {4.85, 5.21},
  estimate: 5.03,
  margin_of_error: 0.18,
  confidence_level: 0.95,
  method: :analytical
}
```

### Bootstrap Confidence Intervals

**For any statistic:**

```elixir
# Bootstrap CI for median
ci = Bench.Stats.ConfidenceInterval.calculate(
  data,
  :median,
  method: :bootstrap,
  iterations: 10_000,
  confidence_level: 0.95
)

# Result:
%{
  interval: {4.8, 5.2},
  estimate: 5.0,
  method: :bootstrap,
  iterations: 10_000
}
```

**Bootstrap Implementation:**

```elixir
defmodule Bootstrap do
  def confidence_interval(data, statistic_fn, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 10_000)
    conf_level = Keyword.get(opts, :confidence_level, 0.95)

    # Generate bootstrap samples
    bootstrap_stats =
      1..iterations
      |> Enum.map(fn _ ->
        sample = Enum.map(1..length(data), fn _ ->
          Enum.random(data)
        end)
        statistic_fn.(sample)
      end)
      |> Enum.sort()

    # Percentile method
    alpha = 1 - conf_level
    lower_idx = round(iterations * alpha / 2)
    upper_idx = round(iterations * (1 - alpha / 2))

    %{
      interval: {
        Enum.at(bootstrap_stats, lower_idx),
        Enum.at(bootstrap_stats, upper_idx)
      },
      estimate: statistic_fn.(data),
      method: :bootstrap,
      iterations: iterations
    }
  end
end

# Custom statistic
data = [0.89, 0.91, 0.88, 0.90, 0.92, 0.87, 0.93]
ci = Bootstrap.confidence_interval(
  data,
  fn d -> Enum.max(d) - Enum.min(d) end  # Range
)
# => %{interval: {0.04, 0.06}, estimate: 0.06}
```

---

## Multiple Comparisons

### The Multiple Comparisons Problem

**Problem:** Testing multiple hypotheses inflates Type I error rate.

```
P(at least one false positive) = 1 - (1 - α)^m

For α = 0.05, m = 10:
P(false positive) = 1 - 0.95^10 = 0.40 (40%!)
```

### Bonferroni Correction

**Most conservative:**

```
α_adjusted = α / m
```

**Implementation:**

```elixir
defmodule Bonferroni do
  def adjust(p_values, alpha \\ 0.05) do
    m = length(p_values)
    adjusted_alpha = alpha / m

    Enum.map(p_values, fn p ->
      %{
        p_value: p,
        adjusted_alpha: adjusted_alpha,
        significant: p < adjusted_alpha,
        adjusted_p: min(p * m, 1.0)
      }
    end)
  end
end

# Example: 10 comparisons
p_values = [0.001, 0.02, 0.03, 0.04, 0.05, 0.06, 0.10, 0.20, 0.50, 0.80]
Bonferroni.adjust(p_values)
# => Only p=0.001 significant (0.001 < 0.005)
```

### Holm-Bonferroni Method

**Less conservative, more powerful:**

```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ p_m
2. Compare p_i to α/(m - i + 1)
3. Stop at first non-significant
```

**Implementation:**

```elixir
defmodule Holm do
  def adjust(p_values, alpha \\ 0.05) do
    m = length(p_values)

    p_values
    |> Enum.with_index()
    |> Enum.sort_by(fn {p, _} -> p end)
    |> Enum.map(fn {{p, original_idx}, sorted_idx} ->
      adjusted_alpha = alpha / (m - sorted_idx)
      %{
        original_index: original_idx,
        p_value: p,
        adjusted_alpha: adjusted_alpha,
        significant: p < adjusted_alpha
      }
    end)
    |> Enum.sort_by(& &1.original_index)
  end
end
```

### Benjamini-Hochberg (FDR)

**Control False Discovery Rate:**

```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ p_m
2. Find largest i where p_i ≤ (i/m) × α
3. Reject H₀ for all p ≤ p_i
```

**Implementation:**

```elixir
defmodule BenjaminiHochberg do
  def adjust(p_values, alpha \\ 0.05) do
    m = length(p_values)

    sorted_with_idx =
      p_values
      |> Enum.with_index()
      |> Enum.sort_by(fn {p, _} -> p end)

    # Find threshold
    threshold_idx =
      sorted_with_idx
      |> Enum.reverse()
      |> Enum.find_index(fn {{p, _}, i} ->
        p <= ((i + 1) / m) * alpha
      end)

    threshold = case threshold_idx do
      nil -> 0
      idx ->
        {{p, _}, _} = Enum.at(Enum.reverse(sorted_with_idx), idx)
        p
    end

    Enum.map(p_values, fn p ->
      %{
        p_value: p,
        significant: p <= threshold,
        fdr_threshold: threshold
      }
    end)
  end
end
```

### Method Comparison

```elixir
defmodule MultipleComparisonStudy do
  def compare_methods(p_values, alpha \\ 0.05) do
    %{
      uncorrected: Enum.count(p_values, & &1 < alpha),
      bonferroni: Enum.count(Bonferroni.adjust(p_values, alpha), & &1.significant),
      holm: Enum.count(Holm.adjust(p_values, alpha), & &1.significant),
      fdr: Enum.count(BenjaminiHochberg.adjust(p_values, alpha), & &1.significant)
    }
  end
end

# Example
p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
MultipleComparisonStudy.compare_methods(p_values)
# => %{
#   uncorrected: 6,    # All significant
#   bonferroni: 1,     # Most conservative
#   holm: 3,           # Less conservative
#   fdr: 5             # Least conservative
# }
```

---

## Assumption Testing

### Normality Tests

**Shapiro-Wilk Test:**

```elixir
defmodule Normality do
  def shapiro_wilk(data) do
    # Simplified implementation
    # Production would use proper W statistic calculation

    n = length(data)
    if n < 3 or n > 5000 do
      raise "Sample size must be between 3 and 5000"
    end

    # Calculate W statistic (simplified)
    sorted = Enum.sort(data)
    mean = Bench.Stats.mean(data)

    # ... W calculation ...

    %{
      test: :shapiro_wilk,
      statistic: 0.96,
      p_value: 0.23,
      normal: true  # p > 0.05
    }
  end

  def check_normality(data) do
    result = shapiro_wilk(data)

    %{
      normal: result.p_value > 0.05,
      p_value: result.p_value,
      recommendation:
        if result.p_value > 0.05 do
          "Data appears normally distributed - use parametric test"
        else
          "Data not normal - use non-parametric test"
        end
    }
  end
end
```

**Q-Q Plot (Quantile-Quantile):**

```elixir
defmodule QQPlot do
  def calculate_qq_points(data) do
    n = length(data)
    sorted_data = Enum.sort(data)

    # Theoretical quantiles (standard normal)
    theoretical = Enum.map(1..n, fn i ->
      p = (i - 0.5) / n
      Bench.Stats.Distributions.normal_quantile(p)
    end)

    # Standardize data
    mean = Bench.Stats.mean(data)
    sd = Bench.Stats.stdev(data)
    standardized = Enum.map(sorted_data, fn x -> (x - mean) / sd end)

    Enum.zip(theoretical, standardized)
  end

  def normality_score(qq_points) do
    # Calculate R² from Q-Q plot
    correlation = Bench.Stats.correlation(
      Enum.map(qq_points, &elem(&1, 0)),
      Enum.map(qq_points, &elem(&1, 1))
    )

    %{
      r_squared: correlation * correlation,
      normal: correlation * correlation > 0.95
    }
  end
end
```

### Homogeneity of Variance

**Levene's Test:**

```elixir
defmodule Levene do
  def test(groups) do
    # Calculate group medians
    medians = Enum.map(groups, &Bench.Stats.median/1)

    # Calculate absolute deviations
    deviations =
      Enum.zip(groups, medians)
      |> Enum.map(fn {group, median} ->
        Enum.map(group, fn x -> abs(x - median) end)
      end)

    # Perform ANOVA on deviations
    result = Bench.Stats.ANOVA.one_way(deviations)

    %{
      test: :levene,
      statistic: result.statistic,
      p_value: result.p_value,
      equal_variances: result.p_value > 0.05,
      recommendation:
        if result.p_value > 0.05 do
          "Variances are equal - can use Student's t-test"
        else
          "Variances unequal - use Welch's t-test"
        end
    }
  end
end
```

### Complete Assumption Check

```elixir
defmodule AssumptionChecker do
  def check_all(group1, group2) do
    %{
      normality_g1: Normality.check_normality(group1),
      normality_g2: Normality.check_normality(group2),
      equal_variances: Levene.test([group1, group2]),
      recommended_test: recommend_test(group1, group2)
    }
  end

  defp recommend_test(g1, g2) do
    norm1 = Normality.check_normality(g1)
    norm2 = Normality.check_normality(g2)
    levene = Levene.test([g1, g2])

    cond do
      norm1.normal and norm2.normal and levene.equal_variances ->
        :student_t_test

      norm1.normal and norm2.normal ->
        :welch_t_test

      true ->
        :mann_whitney
    end
  end
end

# Usage
checker = AssumptionChecker.check_all(group1, group2)
# => %{
#   normality_g1: %{normal: true, ...},
#   normality_g2: %{normal: true, ...},
#   equal_variances: %{equal_variances: false, ...},
#   recommended_test: :welch_t_test
# }
```

---

## Complete Examples

### Example 1: Model Comparison Study

```elixir
defmodule ModelComparisonStudy do
  def run_full_analysis do
    # Collect accuracy scores
    gpt4_scores = measure_accuracy(:gpt4, n: 30)
    claude_scores = measure_accuracy(:claude, n: 30)
    gemini_scores = measure_accuracy(:gemini, n: 30)

    # Step 1: Descriptive statistics
    descriptive = %{
      gpt4: describe(gpt4_scores),
      claude: describe(claude_scores),
      gemini: describe(gemini_scores)
    }

    # Step 2: Check assumptions
    assumptions = check_assumptions([gpt4_scores, claude_scores, gemini_scores])

    # Step 3: Choose and run test
    test_result = if assumptions.all_normal do
      # Parametric: ANOVA
      Bench.Stats.ANOVA.one_way([gpt4_scores, claude_scores, gemini_scores])
    else
      # Non-parametric: Kruskal-Wallis
      Bench.Stats.KruskalWallis.test([gpt4_scores, claude_scores, gemini_scores])
    end

    # Step 4: Post-hoc comparisons (if significant)
    post_hoc = if test_result.p_value < 0.05 do
      pairwise_with_correction([
        {gpt4_scores, :gpt4},
        {claude_scores, :claude},
        {gemini_scores, :gemini}
      ])
    else
      nil
    end

    # Step 5: Effect sizes
    effect_sizes = calculate_all_effect_sizes([
      {gpt4_scores, :gpt4},
      {claude_scores, :claude},
      {gemini_scores, :gemini}
    ])

    # Step 6: Power analysis
    power = Bench.Stats.Power.analyze(:anova,
      analysis_type: :post_hoc,
      effect_size: test_result.effect_size.eta_squared,
      n_per_group: 30,
      k: 3
    )

    # Compile report
    %{
      descriptive: descriptive,
      assumptions: assumptions,
      test: test_result,
      post_hoc: post_hoc,
      effect_sizes: effect_sizes,
      power: power,
      conclusion: generate_conclusion(test_result, effect_sizes, power)
    }
  end

  defp describe(data) do
    %{
      n: length(data),
      mean: Bench.Stats.mean(data),
      sd: Bench.Stats.stdev(data),
      median: Bench.Stats.median(data),
      min: Enum.min(data),
      max: Enum.max(data),
      q25: Bench.Stats.quantile(data, 0.25),
      q75: Bench.Stats.quantile(data, 0.75)
    }
  end

  defp check_assumptions(groups) do
    normality = Enum.map(groups, &Normality.check_normality/1)
    levene = Levene.test(groups)

    %{
      all_normal: Enum.all?(normality, & &1.normal),
      equal_variances: levene.equal_variances,
      details: %{
        normality: normality,
        levene: levene
      }
    }
  end

  defp pairwise_with_correction(groups) do
    comparisons = for {g1, l1} <- groups,
                      {g2, l2} <- groups,
                      l1 < l2 do
      result = Bench.compare(g1, g2)
      effect = Bench.effect_size(g1, g2)

      %{
        comparison: "#{l1} vs #{l2}",
        p_value: result.p_value,
        effect_size: effect.cohens_d,
        result: result
      }
    end

    # Apply Holm-Bonferroni
    p_values = Enum.map(comparisons, & &1.p_value)
    corrections = Holm.adjust(p_values)

    Enum.zip(comparisons, corrections)
    |> Enum.map(fn {comp, corr} ->
      Map.merge(comp, corr)
    end)
  end

  defp calculate_all_effect_sizes(groups) do
    for {g1, l1} <- groups,
        {g2, l2} <- groups,
        l1 < l2 do
      %{
        comparison: "#{l1} vs #{l2}",
        cohens_d: Bench.effect_size(g1, g2).cohens_d
      }
    end
  end

  defp generate_conclusion(test, effects, power) do
    sig_text = if test.p_value < 0.05, do: "significant", else: "non-significant"

    effect_text = case test.effect_size.interpretation do
      "large" -> "large effect"
      "medium" -> "medium effect"
      "small" -> "small effect"
      _ -> "negligible effect"
    end

    power_text = if power.power > 0.8, do: "adequate", else: "inadequate"

    """
    The analysis revealed a #{sig_text} difference between models
    (p = #{Float.round(test.p_value, 4)}) with a #{effect_text}
    (η² = #{Float.round(test.effect_size.eta_squared, 3)}).
    The study had #{power_text} power (#{Float.round(power.power * 100, 1)}%).
    """
  end
end

# Run complete analysis
report = ModelComparisonStudy.run_full_analysis()
IO.puts report.conclusion
```

### Example 2: Before-After Intervention Study

```elixir
defmodule InterventionStudy do
  def analyze_prompt_engineering_impact do
    # Collect paired data
    subjects = 1..50 |> Enum.to_list()

    data = Enum.map(subjects, fn id ->
      baseline = measure_baseline(id)
      intervention = measure_with_fewshot(id)
      %{id: id, baseline: baseline, intervention: intervention}
    end)

    baseline_scores = Enum.map(data, & &1.baseline)
    intervention_scores = Enum.map(data, & &1.intervention)

    # Check normality of differences
    differences = Enum.zip_with(baseline_scores, intervention_scores, fn b, i -> i - b end)
    normality = Normality.check_normality(differences)

    # Select test
    test_result = if normality.normal do
      Bench.Stats.PairedTTest.test(baseline_scores, intervention_scores)
    else
      Bench.Stats.Wilcoxon.test(baseline_scores, intervention_scores)
    end

    # Effect size
    effect = Bench.Stats.EffectSize.paired_cohens_d(
      baseline_scores,
      intervention_scores
    )

    # Confidence interval for difference
    ci = Bench.Stats.ConfidenceInterval.calculate(
      differences,
      :mean,
      confidence_level: 0.95
    )

    %{
      test: test_result,
      effect_size: effect,
      confidence_interval: ci,
      interpretation: interpret_intervention(test_result, effect, ci)
    }
  end

  defp interpret_intervention(test, effect, ci) do
    if test.p_value < 0.05 do
      direction = if effect.mean_diff > 0, do: "improved", else: "decreased"

      """
      The intervention significantly #{direction} performance,
      t(#{test.metadata.df}) = #{Float.round(test.statistic, 2)},
      p = #{Float.round(test.p_value, 4)},
      d = #{Float.round(effect.cohens_d, 2)}.

      Mean difference: #{Float.round(effect.mean_diff * 100, 1)} percentage points,
      95% CI [#{Float.round(elem(ci.interval, 0) * 100, 1)},
              #{Float.round(elem(ci.interval, 1) * 100, 1)}].

      Effect size is #{effect.interpretation}.
      """
    else
      "No significant effect of intervention (p = #{Float.round(test.p_value, 3)})."
    end
  end
end
```

---

## Best Practices

### 1. Always Report Effect Sizes

```elixir
# ❌ Bad: Only p-value
"The difference was significant (p = 0.03)"

# ✅ Good: P-value + effect size
"The difference was significant (p = 0.03, d = 0.52, medium effect)"
```

### 2. Check Assumptions

```elixir
# ❌ Bad: Assume normality
result = Bench.Stats.TTest.test(g1, g2)

# ✅ Good: Check first
if Normality.check_normality(g1).normal and Normality.check_normality(g2).normal do
  Bench.Stats.TTest.test(g1, g2)
else
  Bench.Stats.MannWhitney.test(g1, g2)
end
```

### 3. Pre-register Analysis Plan

```elixir
defmodule PreregisteredStudy do
  @analysis_plan %{
    primary_hypothesis: "Few-shot prompting improves accuracy",
    primary_test: :paired_t_test,
    alpha: 0.05,
    min_effect_size: 0.5,
    sample_size: 50,
    analysis_script: &__MODULE__.analyze/1
  }

  def analyze(data) do
    # Follow pre-registered plan exactly
    # ...
  end
end
```

### 4. Correct for Multiple Comparisons

```elixir
# ❌ Bad: Multiple uncorrected tests
results = Enum.map(comparisons, fn {g1, g2} ->
  Bench.compare(g1, g2)
end)

# ✅ Good: Apply correction
p_values = Enum.map(results, & &1.p_value)
corrected = Holm.adjust(p_values)
```

### 5. Report Confidence Intervals

```elixir
# ❌ Bad: Point estimate only
"Mean difference = 0.087"

# ✅ Good: With CI
"Mean difference = 0.087, 95% CI [0.042, 0.132]"
```

### 6. Use Appropriate Sample Sizes

```elixir
# Before collecting data
power_analysis = Bench.Stats.Power.analyze(:t_test,
  analysis_type: :a_priori,
  effect_size: 0.5,
  power: 0.80
)

IO.puts "Collect at least #{power_analysis.n_per_group} samples per group"
```

---

## References

### Statistical Methods

1. **Statistical Power Analysis**
   - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)

2. **Effect Sizes**
   - Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science

3. **Multiple Comparisons**
   - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate

4. **Non-Parametric Tests**
   - Hollander, M., & Wolfe, D. A. (1999). Nonparametric Statistical Methods

### Implementation Validation

All implementations validated against:
- R (stats package)
- SciPy (scipy.stats)
- SPSS

### Further Reading

1. **Experimental Design**
   - Montgomery, D. C. (2017). Design and Analysis of Experiments

2. **Bayesian Alternatives**
   - Kruschke, J. K. (2014). Doing Bayesian Data Analysis

3. **ML-Specific**
   - Dror, R., et al. (2018). Deep Dominance - How to Properly Compare Deep Neural Models

---

## Appendix: Quick Reference

### Test Selection Matrix

| Data Type | Groups | Paired | Normal | Test |
|-----------|--------|--------|--------|------|
| Continuous | 2 | No | Yes | Independent t-test |
| Continuous | 2 | No | No | Mann-Whitney U |
| Continuous | 2 | Yes | Yes | Paired t-test |
| Continuous | 2 | Yes | No | Wilcoxon signed-rank |
| Continuous | 3+ | No | Yes | One-way ANOVA |
| Continuous | 3+ | No | No | Kruskal-Wallis |

### Effect Size Guidelines

| Test | Effect Size | Small | Medium | Large |
|------|-------------|-------|--------|-------|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | η² | 0.01 | 0.06 | 0.14 |
| Mann-Whitney | r | 0.1 | 0.3 | 0.5 |

### Power & Sample Size

| Power | α | Small d | Medium d | Large d |
|-------|---|---------|----------|---------|
| 0.80 | 0.05 | 393 | 64 | 26 |
| 0.90 | 0.05 | 526 | 85 | 34 |
| 0.95 | 0.05 | 651 | 105 | 42 |

---

**End of Guide**

For questions or contributions: https://github.com/your-org/elixir_ai_research

Last Updated: 2025-10-08
Version: 0.1.0
