#!/usr/bin/env elixir

# Statistics Example
# Demonstrates statistical analysis capabilities

alias CrucibleFramework.Statistics

IO.puts("\n=== Crucible Framework: Statistics Example ===\n")

# Sample data: Latency measurements in milliseconds
baseline_latencies = [850, 920, 780, 1050, 890, 950, 810, 1100, 870, 940]
ensemble_latencies = [1200, 1350, 1180, 1420, 1250, 1300, 1220, 1380, 1270, 1320]

IO.puts("Comparing Baseline vs Ensemble Latencies")
IO.puts("=" |> String.duplicate(50))

# Baseline statistics
IO.puts("\nBaseline Model:")
baseline_summary = Statistics.summary(baseline_latencies)
IO.puts("  Count:      #{baseline_summary.count}")
IO.puts("  Mean:       #{Float.round(baseline_summary.mean, 2)} ms")
IO.puts("  Median:     #{Float.round(baseline_summary.median, 2)} ms")
IO.puts("  Std Dev:    #{Float.round(baseline_summary.std_dev, 2)} ms")
IO.puts("  Min:        #{baseline_summary.min} ms")
IO.puts("  Max:        #{baseline_summary.max} ms")
IO.puts("  P50:        #{Float.round(baseline_summary.p50, 2)} ms")
IO.puts("  P95:        #{Float.round(baseline_summary.p95, 2)} ms")
IO.puts("  P99:        #{Float.round(baseline_summary.p99, 2)} ms")

# Ensemble statistics
IO.puts("\nEnsemble Model:")
ensemble_summary = Statistics.summary(ensemble_latencies)
IO.puts("  Count:      #{ensemble_summary.count}")
IO.puts("  Mean:       #{Float.round(ensemble_summary.mean, 2)} ms")
IO.puts("  Median:     #{Float.round(ensemble_summary.median, 2)} ms")
IO.puts("  Std Dev:    #{Float.round(ensemble_summary.std_dev, 2)} ms")
IO.puts("  Min:        #{ensemble_summary.min} ms")
IO.puts("  Max:        #{ensemble_summary.max} ms")
IO.puts("  P50:        #{Float.round(ensemble_summary.p50, 2)} ms")
IO.puts("  P95:        #{Float.round(ensemble_summary.p95, 2)} ms")
IO.puts("  P99:        #{Float.round(ensemble_summary.p99, 2)} ms")

# Comparison
IO.puts("\nComparison:")
mean_diff = ensemble_summary.mean - baseline_summary.mean
mean_pct = mean_diff / baseline_summary.mean * 100
IO.puts("  Mean Difference:   #{Float.round(mean_diff, 2)} ms (#{Float.round(mean_pct, 1)}%)")

p99_diff = ensemble_summary.p99 - baseline_summary.p99
p99_pct = p99_diff / baseline_summary.p99 * 100
IO.puts("  P99 Difference:    #{Float.round(p99_diff, 2)} ms (#{Float.round(p99_pct, 1)}%)")

# Accuracy data
IO.puts("\n\nAccuracy Comparison")
IO.puts("=" |> String.duplicate(50))

baseline_accuracy = [0.89, 0.87, 0.90, 0.88, 0.91, 0.89, 0.90, 0.88, 0.87, 0.90]
ensemble_accuracy = [0.96, 0.97, 0.94, 0.95, 0.98, 0.96, 0.97, 0.95, 0.94, 0.96]

baseline_acc_summary = Statistics.summary(baseline_accuracy)
ensemble_acc_summary = Statistics.summary(ensemble_accuracy)

IO.puts("\nBaseline Accuracy:")
IO.puts("  Mean:       #{Float.round(baseline_acc_summary.mean, 4)}")
IO.puts("  Std Dev:    #{Float.round(baseline_acc_summary.std_dev, 4)}")
IO.puts("  Min:        #{baseline_acc_summary.min}")
IO.puts("  Max:        #{baseline_acc_summary.max}")

IO.puts("\nEnsemble Accuracy:")
IO.puts("  Mean:       #{Float.round(ensemble_acc_summary.mean, 4)}")
IO.puts("  Std Dev:    #{Float.round(ensemble_acc_summary.std_dev, 4)}")
IO.puts("  Min:        #{ensemble_acc_summary.min}")
IO.puts("  Max:        #{ensemble_acc_summary.max}")

accuracy_improvement = (ensemble_acc_summary.mean - baseline_acc_summary.mean) * 100
IO.puts("\nAccuracy Improvement: +#{Float.round(accuracy_improvement, 2)} percentage points")

IO.puts("\n=== Example Complete ===\n")
