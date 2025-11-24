defmodule Crucible.Stage.Bench do
  @moduledoc """
  Statistical benchmarking stage using crucible_bench.

  This stage performs rigorous statistical analysis on experiment outputs,
  including significance tests, effect sizes, confidence intervals, and
  power analysis. Results are publication-quality.

  ## Supported Tests

  - `:ttest` - Two-sample t-test for comparing means
  - `:paired_ttest` - Paired t-test for matched samples
  - `:bootstrap` - Bootstrap confidence intervals
  - `:wilcoxon` - Wilcoxon signed-rank test (non-parametric)
  - `:mann_whitney` - Mann-Whitney U test (non-parametric)
  - `:anova` - One-way ANOVA for multiple groups
  - `:kruskal_wallis` - Kruskal-Wallis test (non-parametric ANOVA)

  ## Configuration

  Tests and parameters are configured via `experiment.reliability.stats`:

      %StatsConfig{
        tests: [:ttest, :bootstrap],
        alpha: 0.05,
        options: %{
          bootstrap_iterations: 10_000,
          effect_size: :cohens_d
        }
      }

  ## Output

  Results stored in `context.metrics.bench`:

      %{
        ttest: %{statistic: 2.45, p_value: 0.02, significant: true},
        effect_size: %{cohens_d: 0.8, interpretation: "large"},
        confidence_interval: %{lower: 0.3, upper: 0.9, level: 0.95}
      }
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.Context
  alias CrucibleBench
  alias CrucibleBench.Stats

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    stats_config = experiment.reliability.stats

    # Merge stage options with config options
    alpha = Map.get(opts, :alpha, stats_config.alpha)
    tests = Map.get(opts, :tests, stats_config.tests)
    bench_opts = Map.merge(stats_config.options, Map.get(opts, :options, %{}))

    Logger.debug("Running statistical benchmarking with tests: #{inspect(tests)}")

    # Extract data for analysis from context
    data_groups = extract_data_groups(ctx, opts)

    if data_groups == %{} do
      Logger.warning("No data available for statistical analysis")
      {:ok, %Context{ctx | metrics: Map.put(ctx.metrics, :bench, %{no_data: true})}}
    else
      # Run statistical analysis based on configured tests
      results = run_statistical_tests(data_groups, tests, alpha, bench_opts)

      # Store results in context metrics
      bench_metrics = %{
        tests_run: tests,
        alpha: alpha,
        results: results,
        timestamp: DateTime.utc_now()
      }

      {:ok, %Context{ctx | metrics: Map.put(ctx.metrics, :bench, bench_metrics)}}
    end
  rescue
    error ->
      Logger.error("Statistical benchmarking failed: #{inspect(error)}")
      {:error, {:bench_failed, error}}
  end

  @impl true
  def describe(opts) do
    %{
      stage: :bench,
      description: "Statistical benchmarking using crucible_bench",
      tests: Map.get(opts, :tests, []),
      alpha: Map.get(opts, :alpha, 0.05)
    }
  end

  # Extract data groups from context for analysis
  defp extract_data_groups(%Context{} = ctx, opts) do
    mode = Map.get(opts, :data_source, :outputs)

    case mode do
      :outputs ->
        extract_from_outputs(ctx.outputs)

      :metrics ->
        extract_from_metrics(ctx.metrics)

      :backend ->
        extract_backend_metrics(ctx.metrics)

      {:custom, extractor} when is_function(extractor, 1) ->
        extractor.(ctx)

      _ ->
        %{}
    end
  end

  defp extract_from_outputs([]), do: %{}

  defp extract_from_outputs(outputs) when is_list(outputs) do
    # Group outputs by prompt for comparison
    outputs
    |> Enum.group_by(& &1[:prompt])
    |> Enum.reduce(%{}, fn {prompt, responses}, acc ->
      # Extract numeric scores if available
      scores = extract_scores(responses)

      if length(scores) > 0 do
        Map.put(acc, prompt, scores)
      else
        acc
      end
    end)
  end

  defp extract_from_metrics(%{} = metrics) do
    # Extract numeric values from metrics for analysis
    Enum.reduce(metrics, %{}, fn {key, value}, acc ->
      case extract_numeric_values(value) do
        [] -> acc
        values -> Map.put(acc, key, values)
      end
    end)
  end

  defp extract_backend_metrics(%{backend: backend_metrics}) when is_map(backend_metrics) do
    # Extract loss values or other backend-specific metrics
    case backend_metrics do
      %{raw_steps: steps} when is_list(steps) ->
        losses = Enum.map(steps, & &1[:loss]) |> Enum.filter(&is_number/1)
        if losses != [], do: %{losses: losses}, else: %{}

      _ ->
        %{}
    end
  end

  defp extract_backend_metrics(_), do: %{}

  defp extract_scores(responses) do
    responses
    |> Enum.flat_map(fn response ->
      case response do
        %{score: score} when is_number(score) -> [score]
        %{accuracy: acc} when is_number(acc) -> [acc]
        %{loss: loss} when is_number(loss) -> [loss]
        _ -> []
      end
    end)
  end

  defp extract_numeric_values(value) when is_list(value) do
    Enum.filter(value, &is_number/1)
  end

  defp extract_numeric_values(value) when is_number(value), do: [value]

  defp extract_numeric_values(%{} = map) do
    map
    |> Map.values()
    |> Enum.flat_map(&extract_numeric_values/1)
  end

  defp extract_numeric_values(_), do: []

  # Run configured statistical tests
  defp run_statistical_tests(data_groups, _tests, _alpha, _opts)
       when map_size(data_groups) == 0 do
    %{error: "No data available for analysis"}
  end

  defp run_statistical_tests(data_groups, _tests, alpha, opts) when map_size(data_groups) == 1 do
    # Single group - run descriptive statistics only
    {group_name, values} = data_groups |> Map.to_list() |> List.first()

    %{
      group: group_name,
      n: length(values),
      mean: Stats.mean(values),
      std: Stats.stdev(values),
      confidence_interval: calculate_ci(values, alpha, opts),
      tests_skipped: "Single group - no comparison tests possible"
    }
  end

  defp run_statistical_tests(data_groups, tests, alpha, opts) when map_size(data_groups) == 2 do
    # Two groups - run comparison tests
    [{group1_name, group1}, {group2_name, group2}] = Map.to_list(data_groups)

    test_results =
      Enum.reduce(tests, %{}, fn test, acc ->
        result = run_single_test(test, group1, group2, alpha, opts)
        Map.put(acc, test, result)
      end)

    # Add effect size calculation if not already included
    effect_size =
      if :effect_size in tests do
        test_results[:effect_size]
      else
        calculate_effect_size(group1, group2, opts)
      end

    %{
      groups: [group1_name, group2_name],
      sample_sizes: [length(group1), length(group2)],
      test_results: test_results,
      effect_size: effect_size,
      alpha: alpha
    }
  end

  defp run_statistical_tests(data_groups, tests, alpha, opts) do
    # Multiple groups - run ANOVA or Kruskal-Wallis
    groups = Map.values(data_groups)
    group_names = Map.keys(data_groups)

    test_results =
      Enum.reduce(tests, %{}, fn test, acc ->
        result = run_multiple_groups_test(test, groups, alpha, opts)
        Map.put(acc, test, result)
      end)

    %{
      groups: group_names,
      sample_sizes: Enum.map(groups, &length/1),
      test_results: test_results,
      alpha: alpha
    }
  end

  # Run individual statistical tests
  defp run_single_test(:ttest, group1, group2, alpha, _opts) do
    try do
      result =
        CrucibleBench.compare(group1, group2,
          test: :t_test,
          confidence_level: 1 - alpha
        )

      %{
        test: :ttest,
        statistic: result.statistic,
        p_value: result.p_value,
        significant: result.p_value < alpha,
        confidence_interval: result.confidence_interval,
        interpretation: interpret_p_value(result.p_value, alpha)
      }
    rescue
      e -> %{test: :ttest, error: Exception.message(e)}
    end
  end

  defp run_single_test(:welch_ttest, group1, group2, alpha, _opts) do
    try do
      result =
        CrucibleBench.compare(group1, group2,
          test: :welch_t_test,
          confidence_level: 1 - alpha
        )

      %{
        test: :welch_ttest,
        statistic: result.statistic,
        p_value: result.p_value,
        significant: result.p_value < alpha,
        confidence_interval: result.confidence_interval,
        interpretation: interpret_p_value(result.p_value, alpha)
      }
    rescue
      e -> %{test: :welch_ttest, error: Exception.message(e)}
    end
  end

  defp run_single_test(:mann_whitney, group1, group2, alpha, _opts) do
    try do
      result =
        CrucibleBench.compare(group1, group2,
          test: :mann_whitney,
          confidence_level: 1 - alpha
        )

      %{
        test: :mann_whitney,
        statistic: result.statistic,
        p_value: result.p_value,
        significant: result.p_value < alpha,
        interpretation: interpret_p_value(result.p_value, alpha)
      }
    rescue
      e -> %{test: :mann_whitney, error: Exception.message(e)}
    end
  end

  defp run_single_test(:paired_ttest, group1, group2, alpha, _opts) do
    if length(group1) != length(group2) do
      %{test: :paired_ttest, error: "Groups must have same length for paired test"}
    else
      try do
        result = CrucibleBench.compare_paired(group1, group2, confidence_level: 1 - alpha)

        %{
          test: :paired_ttest,
          statistic: result.statistic,
          p_value: result.p_value,
          significant: result.p_value < alpha,
          confidence_interval: result.confidence_interval,
          interpretation: interpret_p_value(result.p_value, alpha)
        }
      rescue
        e -> %{test: :paired_ttest, error: Exception.message(e)}
      end
    end
  end

  defp run_single_test(:wilcoxon, group1, group2, alpha, _opts) do
    if length(group1) != length(group2) do
      %{test: :wilcoxon, error: "Groups must have same length for paired test"}
    else
      try do
        result =
          CrucibleBench.compare_paired(group1, group2,
            test: :wilcoxon,
            confidence_level: 1 - alpha
          )

        %{
          test: :wilcoxon,
          statistic: result.statistic,
          p_value: result.p_value,
          significant: result.p_value < alpha,
          interpretation: interpret_p_value(result.p_value, alpha)
        }
      rescue
        e -> %{test: :wilcoxon, error: Exception.message(e)}
      end
    end
  end

  defp run_single_test(:bootstrap, group1, group2, alpha, opts) do
    iterations = Map.get(opts, :bootstrap_iterations, 10_000)

    try do
      ci1 =
        CrucibleBench.confidence_interval(group1, :mean,
          method: :bootstrap,
          confidence_level: 1 - alpha,
          iterations: iterations
        )

      ci2 =
        CrucibleBench.confidence_interval(group2, :mean,
          method: :bootstrap,
          confidence_level: 1 - alpha,
          iterations: iterations
        )

      # Calculate difference CI
      mean_diff = Stats.mean(group2) - Stats.mean(group1)

      %{
        test: :bootstrap,
        group1_ci: ci1.interval,
        group2_ci: ci2.interval,
        mean_difference: mean_diff,
        iterations: iterations,
        confidence_level: 1 - alpha
      }
    rescue
      e -> %{test: :bootstrap, error: Exception.message(e)}
    end
  end

  defp run_single_test(test, _group1, _group2, _alpha, _opts) do
    %{test: test, error: "Unknown test: #{test}"}
  end

  defp run_multiple_groups_test(:anova, groups, alpha, _opts) do
    try do
      result = CrucibleBench.compare_multiple(groups, test: :anova)

      %{
        test: :anova,
        statistic: result.statistic,
        p_value: result.p_value,
        significant: result.p_value < alpha,
        interpretation: interpret_p_value(result.p_value, alpha)
      }
    rescue
      e -> %{test: :anova, error: Exception.message(e)}
    end
  end

  defp run_multiple_groups_test(:kruskal_wallis, groups, alpha, _opts) do
    try do
      result = CrucibleBench.compare_multiple(groups, test: :kruskal_wallis)

      %{
        test: :kruskal_wallis,
        statistic: result.statistic,
        p_value: result.p_value,
        significant: result.p_value < alpha,
        interpretation: interpret_p_value(result.p_value, alpha)
      }
    rescue
      e -> %{test: :kruskal_wallis, error: Exception.message(e)}
    end
  end

  defp run_multiple_groups_test(test, _groups, _alpha, _opts) do
    %{test: test, error: "Test #{test} not applicable for multiple groups"}
  end

  # Calculate confidence interval
  defp calculate_ci(values, alpha, _opts) do
    try do
      result = CrucibleBench.confidence_interval(values, :mean, confidence_level: 1 - alpha)
      result.interval
    rescue
      _ -> nil
    end
  end

  # Calculate effect size
  defp calculate_effect_size(group1, group2, _opts) do
    try do
      CrucibleBench.effect_size(group1, group2)
    rescue
      _ -> %{error: "Could not calculate effect size"}
    end
  end

  # Interpret p-value
  defp interpret_p_value(p_value, alpha) when p_value < alpha / 100 do
    "Extremely significant (p < #{alpha / 100})"
  end

  defp interpret_p_value(p_value, alpha) when p_value < alpha / 10 do
    "Highly significant (p < #{alpha / 10})"
  end

  defp interpret_p_value(p_value, alpha) when p_value < alpha do
    "Significant (p < #{alpha})"
  end

  defp interpret_p_value(p_value, _alpha) do
    "Not significant (p = #{Float.round(p_value, 4)})"
  end
end
