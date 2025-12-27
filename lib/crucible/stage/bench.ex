defmodule Crucible.Stage.Bench do
  @moduledoc """
  Statistical benchmarking stage using crucible_bench.

  This stage performs statistical analysis on experiment outputs or metrics.
  Configure tests via stage options or experiment reliability config.

  ## Supported Tests

  - `:ttest` - Two-sample t-test for comparing means
  - `:paired_ttest` - Paired t-test for matched samples
  - `:bootstrap` - Bootstrap confidence intervals
  - `:wilcoxon` - Wilcoxon signed-rank test (non-parametric)
  - `:mann_whitney` - Mann-Whitney U test (non-parametric)
  - `:anova` - One-way ANOVA for multiple groups
  - `:kruskal_wallis` - Kruskal-Wallis test (non-parametric ANOVA)

  ## Configuration

      %StageDef{
        name: :bench,
        options: %{
          tests: [:ttest, :bootstrap],
          alpha: 0.05,
          data_source: :outputs  # or :metrics, or {:custom, fn ctx -> ... end}
        }
      }

  ## Output

  Results stored in `context.metrics.bench`.
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.Context
  alias CrucibleBench
  alias CrucibleBench.Stats

  @compile {:no_warn_undefined, CrucibleBench}
  @compile {:no_warn_undefined, CrucibleBench.Stats}

  @default_alpha 0.05
  @default_tests [:ttest]

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    case ensure_bench_available() do
      :ok ->
        # Get config from experiment.reliability.stats if available, otherwise use opts
        {alpha, tests, bench_opts} = extract_config(experiment, opts)

        Logger.debug("Running statistical benchmarking with tests: #{inspect(tests)}")

        # Extract data for analysis from context
        data_groups = extract_data_groups(ctx, opts)

        if data_groups == %{} do
          Logger.warning("No data available for statistical analysis")
          {:ok, Context.put_metric(ctx, :bench, %{no_data: true})}
        else
          # Run statistical analysis based on configured tests
          results = run_statistical_tests(data_groups, tests, alpha, bench_opts)

          bench_metrics = %{
            tests_run: tests,
            alpha: alpha,
            results: results,
            timestamp: DateTime.utc_now()
          }

          {:ok, Context.put_metric(ctx, :bench, bench_metrics)}
        end

      {:error, _} = error ->
        error
    end
  rescue
    error ->
      Logger.error("Statistical benchmarking failed: #{inspect(error)}")
      {:error, {:bench_failed, error}}
  end

  @impl true
  def describe(_opts) do
    %{
      name: :bench,
      description: "Statistical benchmarking and hypothesis testing using crucible_bench",
      required: [],
      optional: [:tests, :alpha, :data_source, :options],
      types: %{
        tests:
          {:list,
           {:enum,
            [:ttest, :paired_ttest, :bootstrap, :wilcoxon, :mann_whitney, :anova, :kruskal_wallis]}},
        alpha: :float,
        data_source: {:enum, [:outputs, :metrics, {:custom, :function}]},
        options: :map
      }
    }
  end

  # ============================================================================
  # Configuration Extraction
  # ============================================================================

  defp ensure_bench_available do
    if Code.ensure_loaded?(CrucibleBench) do
      :ok
    else
      Logger.error(
        "crucible_bench dependency not available; add {:crucible_bench, \"~> 0.4.0\"} or remove :bench from the pipeline"
      )

      {:error, {:missing_dependency, :crucible_bench}}
    end
  end

  defp extract_config(experiment, opts) do
    stats_config = get_stats_config(experiment)
    alpha = get_config_value(opts, stats_config, :alpha, @default_alpha)
    tests = get_config_value(opts, stats_config, :tests, @default_tests)
    base_opts = Map.get(stats_config || %{}, :options, %{})
    bench_opts = Map.merge(base_opts, Map.get(opts, :options, %{}))

    {alpha, tests, bench_opts}
  end

  defp get_stats_config(%{reliability: %{stats: stats}}) when not is_nil(stats), do: stats
  defp get_stats_config(_), do: nil

  defp get_config_value(opts, stats_config, key, default) do
    Map.get(opts, key) || Map.get(stats_config || %{}, key) || default
  end

  # ============================================================================
  # Data Extraction
  # ============================================================================

  defp extract_data_groups(%Context{} = ctx, opts) do
    mode = Map.get(opts, :data_source, :outputs)

    case mode do
      :outputs ->
        extract_from_outputs(ctx.outputs)

      :metrics ->
        extract_from_metrics(ctx.metrics)

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
      scores = extract_scores(responses)
      if Enum.empty?(scores), do: acc, else: Map.put(acc, prompt, scores)
    end)
  end

  defp extract_from_metrics(%{} = metrics) do
    Enum.reduce(metrics, %{}, fn {key, value}, acc ->
      case extract_numeric_values(value) do
        [] -> acc
        values -> Map.put(acc, key, values)
      end
    end)
  end

  defp extract_scores(responses) do
    Enum.flat_map(responses, fn response ->
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

  # ============================================================================
  # Statistical Tests
  # ============================================================================

  defp run_statistical_tests(data_groups, _tests, _alpha, _opts)
       when map_size(data_groups) == 0 do
    %{error: "No data available for analysis"}
  end

  defp run_statistical_tests(data_groups, _tests, alpha, opts)
       when map_size(data_groups) == 1 do
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

  defp run_statistical_tests(data_groups, tests, alpha, opts)
       when map_size(data_groups) == 2 do
    [{group1_name, group1}, {group2_name, group2}] = Map.to_list(data_groups)

    test_results =
      Enum.reduce(tests, %{}, fn test, acc ->
        result = run_single_test(test, group1, group2, alpha, opts)
        Map.put(acc, test, result)
      end)

    effect_size = calculate_effect_size(group1, group2, opts)

    %{
      groups: [group1_name, group2_name],
      sample_sizes: [length(group1), length(group2)],
      test_results: test_results,
      effect_size: effect_size,
      alpha: alpha
    }
  end

  defp run_statistical_tests(data_groups, tests, alpha, opts) do
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

  defp run_single_test(:ttest, group1, group2, alpha, _opts) do
    result = CrucibleBench.compare(group1, group2, test: :t_test, confidence_level: 1 - alpha)

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

  defp run_single_test(:mann_whitney, group1, group2, alpha, _opts) do
    result =
      CrucibleBench.compare(group1, group2, test: :mann_whitney, confidence_level: 1 - alpha)

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

  defp run_single_test(:bootstrap, group1, group2, alpha, opts) do
    iterations = Map.get(opts, :bootstrap_iterations, 10_000)

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

  defp run_single_test(test, _group1, _group2, _alpha, _opts) do
    %{test: test, error: "Unknown test: #{test}"}
  end

  defp run_multiple_groups_test(:anova, groups, alpha, _opts) do
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

  defp run_multiple_groups_test(:kruskal_wallis, groups, alpha, _opts) do
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

  defp run_multiple_groups_test(test, _groups, _alpha, _opts) do
    %{test: test, error: "Test #{test} not applicable for multiple groups"}
  end

  defp calculate_ci(values, alpha, _opts) do
    result = CrucibleBench.confidence_interval(values, :mean, confidence_level: 1 - alpha)
    result.interval
  rescue
    _ -> nil
  end

  defp calculate_effect_size(group1, group2, _opts) do
    CrucibleBench.effect_size(group1, group2)
  rescue
    _ -> %{error: "Could not calculate effect size"}
  end

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
