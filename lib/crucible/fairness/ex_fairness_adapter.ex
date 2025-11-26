defmodule Crucible.Fairness.ExFairnessAdapter do
  @moduledoc """
  Fairness adapter that bridges to the ExFairness library.

  This adapter provides full integration with ExFairness metrics:
  - Demographic Parity
  - Equalized Odds
  - Equal Opportunity
  - Predictive Parity

  ## Requirements

  This adapter requires the `ex_fairness` package to be available.
  If ExFairness is not installed, use `Crucible.Fairness.Noop` instead.

  ## Configuration

  Configure this adapter in your application:

      config :crucible_framework,
        fairness_adapter: Crucible.Fairness.ExFairnessAdapter

  ## Usage

  The adapter is called automatically by `Crucible.Stage.Fairness` when
  fairness evaluation is enabled in the experiment's reliability config:

      %Experiment{
        reliability: %ReliabilityConfig{
          fairness: %FairnessConfig{
            enabled: true,
            metrics: [:demographic_parity, :equalized_odds],
            group_by: :gender,
            options: %{threshold: 0.1}
          }
        }
      }

  """

  @behaviour Crucible.Fairness.Adapter

  require Logger

  @compile {:no_warn_undefined, ExFairness}

  @available_metrics [
    :demographic_parity,
    :equalized_odds,
    :equal_opportunity,
    :predictive_parity
  ]
  @default_threshold 0.1
  @default_min_per_group 10

  @doc """
  Returns true if ExFairness is available.
  """
  def available?, do: ex_fairness_available?()

  @impl true
  def evaluate(predictions, labels, sensitive_attr, opts) do
    if not ex_fairness_available?() do
      {:error,
       {:ex_fairness_not_available,
        "ExFairness library is not installed. Add {:ex_fairness, path: \"../ExFairness\"} to your dependencies."}}
    else
      do_evaluate(predictions, labels, sensitive_attr, opts)
    end
  end

  defp do_evaluate(predictions, labels, sensitive_attr, opts) do
    metrics = Keyword.get(opts, :metrics, @available_metrics)
    threshold = Keyword.get(opts, :threshold, @default_threshold)
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)
    fail_fast = Keyword.get(opts, :fail_fast, false)

    with {:ok, pred_tensor} <- to_tensor(predictions, "predictions"),
         {:ok, label_tensor} <- to_tensor(labels, "labels"),
         {:ok, sensitive_tensor} <- to_tensor(sensitive_attr, "sensitive_attr"),
         :ok <- validate_shapes(pred_tensor, label_tensor, sensitive_tensor) do
      metric_opts = [threshold: threshold, min_per_group: min_per_group]

      {results, stopped_early?} =
        compute_metrics(
          metrics,
          pred_tensor,
          label_tensor,
          sensitive_tensor,
          metric_opts,
          fail_fast
        )

      # Aggregate results
      passed_count = Enum.count(results, fn {_m, r} -> r[:passes] == true end)
      failed_count = Enum.count(results, fn {_m, r} -> r[:passes] == false end)
      error_count = Enum.count(results, fn {_m, r} -> Map.has_key?(r, :error) end)
      total_count = map_size(results)

      overall_passes = failed_count == 0 and error_count == 0

      # Collect violations for failed metrics
      violations =
        results
        |> Enum.filter(fn {_m, r} -> r[:passes] == false end)
        |> Enum.map(fn {metric, result} ->
          %{
            metric: metric,
            disparity: result[:disparity] || result[:tpr_disparity] || 0.0,
            threshold: result[:threshold] || threshold,
            interpretation: result[:interpretation]
          }
        end)

      {:ok,
       %{
         metrics: results,
         overall_passes: overall_passes,
         passed_count: passed_count,
         failed_count: failed_count,
         error_count: error_count,
         total_count: total_count,
         violations: violations,
         stopped_early: stopped_early?,
         threshold: threshold,
         timestamp: DateTime.utc_now()
       }}
    end
  rescue
    error ->
      Logger.error("Fairness evaluation failed: #{inspect(error)}")
      {:error, {:evaluation_failed, Exception.message(error)}}
  end

  @impl true
  def generate_report(evaluation_result, format) do
    case format do
      :markdown -> generate_markdown_report(evaluation_result)
      :json -> generate_json_report(evaluation_result)
      :text -> generate_text_report(evaluation_result)
      _ -> {:error, {:unsupported_format, format}}
    end
  end

  # Private functions

  defp to_tensor(data, _name) when is_struct(data, Nx.Tensor), do: {:ok, data}

  defp to_tensor(data, name) when is_list(data) do
    if Enum.all?(data, &is_number/1) do
      {:ok, Nx.tensor(data)}
    else
      {:error, {:invalid_data, "#{name} must contain only numbers"}}
    end
  end

  defp to_tensor(data, name) do
    {:error, {:invalid_data, "#{name} must be a list or Nx.Tensor, got: #{inspect(data)}"}}
  end

  defp validate_shapes(pred, label, sensitive) do
    pred_shape = Nx.shape(pred)
    label_shape = Nx.shape(label)
    sensitive_shape = Nx.shape(sensitive)

    cond do
      pred_shape != label_shape ->
        {:error, {:shape_mismatch, "predictions and labels shapes don't match"}}

      pred_shape != sensitive_shape ->
        {:error, {:shape_mismatch, "predictions and sensitive_attr shapes don't match"}}

      true ->
        :ok
    end
  end

  defp compute_metrics(metrics, predictions, labels, sensitive, opts, fail_fast) do
    Enum.reduce_while(metrics, {%{}, false}, fn metric, {acc, _stopped} ->
      result = compute_single_metric(metric, predictions, labels, sensitive, opts)

      new_acc = Map.put(acc, metric, result)

      if fail_fast and result[:passes] == false do
        {:halt, {new_acc, true}}
      else
        {:cont, {new_acc, false}}
      end
    end)
  end

  defp compute_single_metric(:demographic_parity, predictions, _labels, sensitive, opts) do
    try do
      ExFairness.demographic_parity(predictions, sensitive, opts)
    rescue
      e -> %{error: Exception.message(e), passes: false}
    end
  end

  defp compute_single_metric(:equalized_odds, predictions, labels, sensitive, opts) do
    try do
      ExFairness.equalized_odds(predictions, labels, sensitive, opts)
    rescue
      e -> %{error: Exception.message(e), passes: false}
    end
  end

  defp compute_single_metric(:equal_opportunity, predictions, labels, sensitive, opts) do
    try do
      ExFairness.equal_opportunity(predictions, labels, sensitive, opts)
    rescue
      e -> %{error: Exception.message(e), passes: false}
    end
  end

  defp compute_single_metric(:predictive_parity, predictions, labels, sensitive, opts) do
    try do
      ExFairness.predictive_parity(predictions, labels, sensitive, opts)
    rescue
      e -> %{error: Exception.message(e), passes: false}
    end
  end

  defp compute_single_metric(unknown_metric, _predictions, _labels, _sensitive, _opts) do
    %{error: "Unknown metric: #{unknown_metric}", passes: false}
  end

  defp ex_fairness_available?, do: Code.ensure_loaded?(ExFairness)

  # Report generation

  defp generate_markdown_report(result) do
    report = """
    # Fairness Evaluation Report

    **Generated:** #{result.timestamp}
    **Overall Status:** #{if result.overall_passes, do: "✓ PASSED", else: "✗ FAILED"}
    **Threshold:** #{result.threshold}

    ## Summary

    | Metric | Passed | Failed | Errors |
    |--------|--------|--------|--------|
    | Total  | #{result.passed_count} | #{result.failed_count} | #{result.error_count} |

    ## Metric Results

    #{format_metrics_markdown(result.metrics)}

    #{if result.violations != [], do: format_violations_markdown(result.violations), else: ""}
    """

    {:ok, report}
  end

  defp format_metrics_markdown(metrics) do
    metrics
    |> Enum.map(fn {metric, result} ->
      status = if result[:passes], do: "✓", else: "✗"
      disparity = get_primary_disparity(metric, result)

      """
      ### #{format_metric_name(metric)} #{status}

      - **Disparity:** #{Float.round(disparity, 4)}
      - **Threshold:** #{result[:threshold] || "N/A"}
      - **Interpretation:** #{result[:interpretation] || "N/A"}
      """
    end)
    |> Enum.join("\n")
  end

  defp format_violations_markdown(violations) do
    """
    ## Fairness Violations

    | Metric | Disparity | Threshold | Issue |
    |--------|-----------|-----------|-------|
    #{Enum.map(violations, fn v -> "| #{format_metric_name(v.metric)} | #{Float.round(v.disparity, 4)} | #{v.threshold} | #{v.interpretation} |" end) |> Enum.join("\n")}
    """
  end

  defp generate_json_report(result) do
    {:ok, Jason.encode!(result, pretty: true)}
  rescue
    e -> {:error, {:json_encoding_failed, Exception.message(e)}}
  end

  defp generate_text_report(result) do
    status = if result.overall_passes, do: "PASSED", else: "FAILED"

    report =
      """
      Fairness Evaluation: #{status}
      ========================
      Passed: #{result.passed_count}/#{result.total_count}
      Failed: #{result.failed_count}
      Errors: #{result.error_count}
      Threshold: #{result.threshold}

      Metrics:
      #{format_metrics_text(result.metrics)}
      """

    {:ok, report}
  end

  defp format_metrics_text(metrics) do
    metrics
    |> Enum.map(fn {metric, result} ->
      status = if result[:passes], do: "PASS", else: "FAIL"
      disparity = get_primary_disparity(metric, result)
      "  - #{format_metric_name(metric)}: #{status} (disparity: #{Float.round(disparity, 4)})"
    end)
    |> Enum.join("\n")
  end

  defp format_metric_name(:demographic_parity), do: "Demographic Parity"
  defp format_metric_name(:equalized_odds), do: "Equalized Odds"
  defp format_metric_name(:equal_opportunity), do: "Equal Opportunity"
  defp format_metric_name(:predictive_parity), do: "Predictive Parity"
  defp format_metric_name(other), do: to_string(other)

  defp get_primary_disparity(:equalized_odds, result) do
    max(result[:tpr_disparity] || 0, result[:fpr_disparity] || 0)
  end

  defp get_primary_disparity(_metric, result), do: result[:disparity] || 0.0
end
