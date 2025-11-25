defmodule Crucible.Stage.Fairness do
  @moduledoc """
  Fairness evaluation stage using ExFairness integration.

  This stage evaluates model outputs for fairness across protected groups,
  detecting bias and disparate impact. It integrates with the ExFairness
  library via a configurable adapter pattern.

  ## Supported Metrics

  - `:demographic_parity` - Equal positive prediction rates across groups
  - `:equalized_odds` - Equal TPR and FPR across groups
  - `:equal_opportunity` - Equal TPR across groups (for positive class)
  - `:predictive_parity` - Equal precision/PPV across groups

  ## Configuration

  Fairness evaluation is configured via `experiment.reliability.fairness`:

      %FairnessConfig{
        enabled: true,
        group_by: :gender,
        metrics: [:demographic_parity, :equalized_odds],
        options: %{
          threshold: 0.1,
          min_per_group: 10,
          fail_fast: false
        }
      }

  ## Data Extraction

  The stage extracts predictions, labels, and sensitive attributes from the
  context. Data can come from:

  1. **Context outputs** - Model predictions with associated metadata
  2. **Context assigns** - Pre-computed tensors passed explicitly
  3. **Stage options** - Direct data specification

  The `group_by` field specifies which attribute to use as the sensitive
  attribute for fairness evaluation.

  ## Output

  Results are stored in `context.metrics.fairness`:

      %{
        overall_passes: true,
        passed_count: 2,
        failed_count: 0,
        metrics: %{
          demographic_parity: %{disparity: 0.05, passes: true, ...},
          equalized_odds: %{tpr_disparity: 0.03, fpr_disparity: 0.02, passes: true, ...}
        },
        violations: [],
        report: "..."
      }

  ## Experiment Failure Mode

  When `options.fail_on_violation` is true (default: false), the stage will
  return an error if any fairness metric fails, failing the entire experiment.
  This is useful for enforcing fairness requirements in production.

  ## Example

      %Experiment{
        id: "hiring_model_v1",
        reliability: %ReliabilityConfig{
          fairness: %FairnessConfig{
            enabled: true,
            group_by: :gender,
            metrics: [:demographic_parity, :equal_opportunity],
            options: %{threshold: 0.1, fail_on_violation: true}
          }
        }
      }

  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.Context
  alias Crucible.IR.FairnessConfig

  @default_metrics [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity]
  @default_threshold 0.1

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    fairness_config = experiment.reliability.fairness

    if fairness_enabled?(fairness_config) do
      run_fairness_evaluation(ctx, fairness_config, opts)
    else
      Logger.debug("Fairness evaluation skipped (disabled in config)")
      {:ok, %Context{ctx | metrics: Map.put(ctx.metrics, :fairness, %{status: :disabled})}}
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: :fairness,
      description: "Fairness evaluation using ExFairness",
      metrics: Map.get(opts, :metrics, @default_metrics),
      threshold: Map.get(opts, :threshold, @default_threshold)
    }
  end

  # Check if fairness is enabled
  defp fairness_enabled?(%FairnessConfig{enabled: true}), do: true
  defp fairness_enabled?(_), do: false

  # Main evaluation logic
  defp run_fairness_evaluation(ctx, fairness_config, opts) do
    adapter = get_adapter()

    # Merge config options with stage options
    metrics = get_metrics(fairness_config, opts)
    threshold = Map.get(fairness_config.options, :threshold, @default_threshold)
    fail_on_violation = Map.get(fairness_config.options, :fail_on_violation, false)

    adapter_opts = [
      metrics: metrics,
      threshold: threshold,
      min_per_group: Map.get(fairness_config.options, :min_per_group, 10),
      fail_fast: Map.get(fairness_config.options, :fail_fast, false)
    ]

    Logger.info("Running fairness evaluation with metrics: #{inspect(metrics)}")

    # Extract data for evaluation
    case extract_fairness_data(ctx, fairness_config, opts) do
      {:ok, {predictions, labels, sensitive_attr}} ->
        evaluate_and_update_context(
          ctx,
          adapter,
          predictions,
          labels,
          sensitive_attr,
          adapter_opts,
          fail_on_violation
        )

      {:error, :no_data} ->
        Logger.warning("No data available for fairness evaluation")

        {:ok,
         %Context{
           ctx
           | metrics:
               Map.put(ctx.metrics, :fairness, %{status: :no_data, message: "No data available"})
         }}

      {:error, reason} ->
        Logger.error("Failed to extract fairness data: #{inspect(reason)}")
        {:error, {:fairness_data_extraction_failed, reason}}
    end
  end

  defp evaluate_and_update_context(
         ctx,
         adapter,
         predictions,
         labels,
         sensitive_attr,
         adapter_opts,
         fail_on_violation
       ) do
    case adapter.evaluate(predictions, labels, sensitive_attr, adapter_opts) do
      {:ok, result} ->
        # Optionally generate a report
        report =
          case adapter.generate_report(result, :markdown) do
            {:ok, report_str} -> report_str
            _ -> nil
          end

        fairness_metrics =
          result
          |> Map.put(:report, report)
          |> Map.put(:status, :completed)

        new_ctx = %Context{ctx | metrics: Map.put(ctx.metrics, :fairness, fairness_metrics)}

        # Check if we should fail the experiment
        if fail_on_violation and not result.overall_passes do
          Logger.error("Fairness violations detected with fail_on_violation=true")

          {:error,
           {:fairness_violations,
            %{
              violations: result.violations,
              passed: result.passed_count,
              failed: result.failed_count
            }}}
        else
          if not result.overall_passes do
            Logger.warning("Fairness violations detected: #{result.failed_count} metrics failed")
          end

          {:ok, new_ctx}
        end

      {:error, reason} ->
        Logger.error("Fairness evaluation failed: #{inspect(reason)}")
        {:error, {:fairness_evaluation_failed, reason}}
    end
  end

  # Get the configured adapter
  defp get_adapter do
    Application.get_env(:crucible_framework, :fairness_adapter, Crucible.Fairness.Noop)
  end

  # Get metrics to evaluate
  defp get_metrics(%FairnessConfig{metrics: []}, opts) do
    Map.get(opts, :metrics, @default_metrics)
  end

  defp get_metrics(%FairnessConfig{metrics: metrics}, _opts), do: metrics

  # Extract data for fairness evaluation
  defp extract_fairness_data(ctx, fairness_config, opts) do
    # Priority: 1) explicit opts, 2) assigns, 3) outputs extraction
    cond do
      # Check for explicitly provided data in opts
      opts[:predictions] && opts[:labels] && opts[:sensitive_attr] ->
        {:ok, {opts[:predictions], opts[:labels], opts[:sensitive_attr]}}

      # Check assigns for pre-computed data
      ctx.assigns[:fairness_predictions] && ctx.assigns[:fairness_labels] &&
          ctx.assigns[:fairness_sensitive_attr] ->
        {:ok,
         {ctx.assigns[:fairness_predictions], ctx.assigns[:fairness_labels],
          ctx.assigns[:fairness_sensitive_attr]}}

      # Extract from outputs
      ctx.outputs != [] ->
        extract_from_outputs(ctx.outputs, fairness_config.group_by, opts)

      # Check examples if outputs are empty
      ctx.examples != nil and ctx.examples != [] ->
        extract_from_examples(ctx.examples, fairness_config.group_by, opts)

      true ->
        {:error, :no_data}
    end
  end

  # Extract predictions, labels, and sensitive attributes from outputs
  defp extract_from_outputs(outputs, group_by, opts) do
    prediction_key = opts[:prediction_key] || :prediction
    label_key = opts[:label_key] || :label
    sensitive_key = group_by || opts[:sensitive_key] || :sensitive_attr

    predictions =
      outputs
      |> Enum.map(&extract_value(&1, prediction_key))
      |> Enum.filter(&(not is_nil(&1)))

    labels =
      outputs
      |> Enum.map(&extract_value(&1, label_key))
      |> Enum.filter(&(not is_nil(&1)))

    sensitive_attrs =
      outputs
      |> Enum.map(&extract_value(&1, sensitive_key))
      |> Enum.filter(&(not is_nil(&1)))

    if length(predictions) > 0 and length(labels) > 0 and length(sensitive_attrs) > 0 and
         length(predictions) == length(labels) and length(predictions) == length(sensitive_attrs) do
      {:ok, {predictions, labels, sensitive_attrs}}
    else
      Logger.debug(
        "Output extraction: predictions=#{length(predictions)}, labels=#{length(labels)}, sensitive=#{length(sensitive_attrs)}"
      )

      {:error, :insufficient_data}
    end
  end

  # Extract from examples (when outputs might be merged with examples)
  defp extract_from_examples(examples, group_by, opts) do
    prediction_key = opts[:prediction_key] || :prediction
    label_key = opts[:label_key] || :label
    sensitive_key = group_by || opts[:sensitive_key] || :sensitive_attr

    predictions =
      examples
      |> Enum.map(&extract_value(&1, prediction_key))
      |> Enum.filter(&(not is_nil(&1)))

    labels =
      examples
      |> Enum.map(&extract_value(&1, label_key))
      |> Enum.filter(&(not is_nil(&1)))

    sensitive_attrs =
      examples
      |> Enum.map(&extract_value(&1, sensitive_key))
      |> Enum.filter(&(not is_nil(&1)))

    if length(predictions) > 0 and length(labels) > 0 and length(sensitive_attrs) > 0 and
         length(predictions) == length(labels) and length(predictions) == length(sensitive_attrs) do
      {:ok, {predictions, labels, sensitive_attrs}}
    else
      {:error, :insufficient_data}
    end
  end

  # Extract a value from a map with various key formats
  defp extract_value(map, key) when is_map(map) do
    cond do
      is_atom(key) and Map.has_key?(map, key) ->
        Map.get(map, key)

      is_atom(key) and Map.has_key?(map, to_string(key)) ->
        Map.get(map, to_string(key))

      is_binary(key) and Map.has_key?(map, key) ->
        Map.get(map, key)

      is_binary(key) and Map.has_key?(map, String.to_existing_atom(key)) ->
        Map.get(map, String.to_existing_atom(key))

      true ->
        nil
    end
  rescue
    ArgumentError -> nil
  end

  defp extract_value(_, _), do: nil
end
