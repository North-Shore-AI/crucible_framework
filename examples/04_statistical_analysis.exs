#!/usr/bin/env elixir

# Statistical Analysis Example
# Demonstrates comprehensive statistical analysis for research

alias CrucibleFramework.Statistics

IO.puts("\n=== Crucible Framework: Statistical Analysis Example ===\n")
IO.puts("Simulating a complete experiment with statistical analysis")
IO.puts("=" |> String.duplicate(70))

# Simulate experimental data
# Hypothesis: 3-model ensemble improves accuracy over single model

defmodule ExperimentSimulator do
  @moduledoc """
  Simulates experimental results for demonstration
  """

  def generate_baseline_results(n, seed \\ 42) do
    :rand.seed(:exsplus, {seed, seed, seed})

    Enum.map(1..n, fn _ ->
      # Baseline: ~89% accuracy with some variance
      base = 0.89
      variance = (:rand.uniform() - 0.5) * 0.06
      Float.round(base + variance, 3)
    end)
  end

  def generate_ensemble_results(n, seed \\ 42) do
    :rand.seed(:exsplus, {seed + 1, seed + 1, seed + 1})

    Enum.map(1..n, fn _ ->
      # Ensemble: ~96% accuracy with less variance
      base = 0.96
      variance = (:rand.uniform() - 0.5) * 0.04
      Float.round(base + variance, 3)
    end)
  end

  def generate_latency_results(n, base_latency, variance_pct, seed \\ 42) do
    :rand.seed(:exsplus, {seed + 2, seed + 2, seed + 2})

    Enum.map(1..n, fn _ ->
      variance = base_latency * variance_pct * (:rand.uniform() - 0.5)
      round(base_latency + variance)
    end)
  end
end

# Generate experiment data
n_samples = 50
baseline_accuracy = ExperimentSimulator.generate_baseline_results(n_samples)
ensemble_accuracy = ExperimentSimulator.generate_ensemble_results(n_samples)
baseline_latency = ExperimentSimulator.generate_latency_results(n_samples, 800, 0.4)
ensemble_latency = ExperimentSimulator.generate_latency_results(n_samples, 1200, 0.3)

# Helper function for interpreting Cohen's d
defmodule StatHelpers do
  def interpret_cohens_d(d) do
    cond do
      abs(d) < 0.2 -> "negligible effect"
      abs(d) < 0.5 -> "small effect"
      abs(d) < 0.8 -> "medium effect"
      abs(d) < 1.2 -> "large effect"
      true -> "very large effect"
    end
  end
end

# Analysis 1: Accuracy Comparison
IO.puts("\n1. ACCURACY ANALYSIS")
IO.puts("=" |> String.duplicate(70))

baseline_acc_stats = Statistics.summary(baseline_accuracy)
ensemble_acc_stats = Statistics.summary(ensemble_accuracy)

IO.puts("\nBaseline Model (Single GPT-4):")
IO.puts("  Sample Size:    #{baseline_acc_stats.count}")

IO.puts(
  "  Mean:           #{Float.round(baseline_acc_stats.mean, 4)} (#{Float.round(baseline_acc_stats.mean * 100, 2)}%)"
)

IO.puts("  Median:         #{Float.round(baseline_acc_stats.median, 4)}")
IO.puts("  Std Dev:        #{Float.round(baseline_acc_stats.std_dev, 4)}")

IO.puts(
  "  95% CI:         [#{Float.round(baseline_acc_stats.mean - 1.96 * baseline_acc_stats.std_dev / :math.sqrt(baseline_acc_stats.count), 4)}, #{Float.round(baseline_acc_stats.mean + 1.96 * baseline_acc_stats.std_dev / :math.sqrt(baseline_acc_stats.count), 4)}]"
)

IO.puts("  Range:          [#{baseline_acc_stats.min}, #{baseline_acc_stats.max}]")

IO.puts("\nEnsemble Model (3-model voting):")
IO.puts("  Sample Size:    #{ensemble_acc_stats.count}")

IO.puts(
  "  Mean:           #{Float.round(ensemble_acc_stats.mean, 4)} (#{Float.round(ensemble_acc_stats.mean * 100, 2)}%)"
)

IO.puts("  Median:         #{Float.round(ensemble_acc_stats.median, 4)}")
IO.puts("  Std Dev:        #{Float.round(ensemble_acc_stats.std_dev, 4)}")

IO.puts(
  "  95% CI:         [#{Float.round(ensemble_acc_stats.mean - 1.96 * ensemble_acc_stats.std_dev / :math.sqrt(ensemble_acc_stats.count), 4)}, #{Float.round(ensemble_acc_stats.mean + 1.96 * ensemble_acc_stats.std_dev / :math.sqrt(ensemble_acc_stats.count), 4)}]"
)

IO.puts("  Range:          [#{ensemble_acc_stats.min}, #{ensemble_acc_stats.max}]")

# Effect size calculation (Cohen's d)
mean_diff = ensemble_acc_stats.mean - baseline_acc_stats.mean
pooled_std = :math.sqrt((baseline_acc_stats.variance + ensemble_acc_stats.variance) / 2)
cohens_d = mean_diff / pooled_std

IO.puts("\nEffect Size:")
IO.puts("  Difference:     +#{Float.round(mean_diff * 100, 2)} percentage points")
IO.puts("  Cohen's d:      #{Float.round(cohens_d, 2)}")
IO.puts("  Interpretation: #{StatHelpers.interpret_cohens_d(cohens_d)}")

# Analysis 2: Latency Comparison
IO.puts("\n2. LATENCY ANALYSIS")
IO.puts("=" |> String.duplicate(70))

baseline_lat_stats = Statistics.summary(baseline_latency)
ensemble_lat_stats = Statistics.summary(ensemble_latency)

IO.puts("\nBaseline Latency:")
IO.puts("  Mean:           #{Float.round(baseline_lat_stats.mean, 2)} ms")
IO.puts("  Median (P50):   #{Float.round(baseline_lat_stats.p50, 2)} ms")
IO.puts("  P95:            #{Float.round(baseline_lat_stats.p95, 2)} ms")
IO.puts("  P99:            #{Float.round(baseline_lat_stats.p99, 2)} ms")
IO.puts("  Range:          [#{baseline_lat_stats.min}, #{baseline_lat_stats.max}] ms")

IO.puts("\nEnsemble Latency:")
IO.puts("  Mean:           #{Float.round(ensemble_lat_stats.mean, 2)} ms")
IO.puts("  Median (P50):   #{Float.round(ensemble_lat_stats.p50, 2)} ms")
IO.puts("  P95:            #{Float.round(ensemble_lat_stats.p95, 2)} ms")
IO.puts("  P99:            #{Float.round(ensemble_lat_stats.p99, 2)} ms")
IO.puts("  Range:          [#{ensemble_lat_stats.min}, #{ensemble_lat_stats.max}] ms")

latency_increase_pct =
  (ensemble_lat_stats.mean - baseline_lat_stats.mean) / baseline_lat_stats.mean * 100

IO.puts("\nLatency Impact:")

IO.puts(
  "  Mean Increase:  +#{Float.round(ensemble_lat_stats.mean - baseline_lat_stats.mean, 2)} ms (+#{Float.round(latency_increase_pct, 1)}%)"
)

# Analysis 3: Cost-Benefit Analysis
IO.puts("\n3. COST-BENEFIT ANALYSIS")
IO.puts("=" |> String.duplicate(70))

# Assume costs (in USD per query)
baseline_cost = 0.01
# 3 models
ensemble_cost = 0.03

# in percentage points
accuracy_gain = mean_diff * 100
cost_increase = ensemble_cost - baseline_cost
cost_per_accuracy_point = cost_increase / accuracy_gain

IO.puts("\nCost Structure:")
IO.puts("  Baseline cost:        $#{Float.round(baseline_cost, 4)} per query")
IO.puts("  Ensemble cost:        $#{Float.round(ensemble_cost, 4)} per query")

IO.puts(
  "  Cost increase:        $#{Float.round(cost_increase, 4)} per query (#{Float.round(cost_increase / baseline_cost * 100, 0)}%)"
)

IO.puts("\nAccuracy vs Cost:")
IO.puts("  Accuracy gain:        +#{Float.round(accuracy_gain, 2)} percentage points")
IO.puts("  Cost per pp gain:     $#{Float.round(cost_per_accuracy_point, 4)}")

# At scale
queries_per_month = 100_000
monthly_baseline = baseline_cost * queries_per_month
monthly_ensemble = ensemble_cost * queries_per_month

IO.puts("\nAt Scale (#{queries_per_month} queries/month):")
IO.puts("  Baseline:             $#{Float.round(monthly_baseline, 2)}/month")
IO.puts("  Ensemble:             $#{Float.round(monthly_ensemble, 2)}/month")
IO.puts("  Additional cost:      $#{Float.round(monthly_ensemble - monthly_baseline, 2)}/month")

# Final Recommendation
IO.puts("\n4. RESEARCH CONCLUSION")
IO.puts("=" |> String.duplicate(70))

IO.puts("\nHypothesis: 3-model ensemble achieves significantly higher accuracy")
IO.puts("Result: SUPPORTED")
IO.puts("")
IO.puts("Key Findings:")
IO.puts("  1. Ensemble shows #{Float.round(accuracy_gain, 2)}pp accuracy improvement")

IO.puts(
  "  2. Effect size (d=#{Float.round(cohens_d, 2)}) indicates #{StatHelpers.interpret_cohens_d(cohens_d)}"
)

IO.puts("  3. Cost increases by #{Float.round(cost_increase / baseline_cost * 100, 0)}%")
IO.puts("  4. Latency increases by #{Float.round(latency_increase_pct, 1)}%")
IO.puts("")
IO.puts("Recommendation:")

cond do
  accuracy_gain > 5 and cost_per_accuracy_point < 0.01 ->
    IO.puts("  ✓ STRONGLY RECOMMENDED: High accuracy gain with reasonable cost")

  accuracy_gain > 3 and cost_per_accuracy_point < 0.02 ->
    IO.puts("  ✓ RECOMMENDED: Good accuracy improvement, acceptable cost increase")

  accuracy_gain > 2 ->
    IO.puts("  ⚠ CONDITIONAL: Consider for high-stakes applications where accuracy is critical")

  true ->
    IO.puts("  ✗ NOT RECOMMENDED: Insufficient accuracy improvement for cost")
end

IO.puts("")
IO.puts("Note: For publication-quality statistical testing (t-tests, p-values, etc.),")
IO.puts("      use the Bench library which provides 15+ statistical tests with")
IO.puts("      automatic assumption checking and effect size calculations.")

IO.puts("\n=== Example Complete ===\n")
