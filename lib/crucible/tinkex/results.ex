defmodule Crucible.Tinkex.Results do
  @moduledoc """
  Result aggregation and analysis for Tinkex experiments.

  This module provides functionality for collecting, summarizing, and
  exporting results from training and evaluation runs. It supports
  quality target validation and report generation.

  ## Example

      # Create results container
      results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")

      # Add training metrics
      results = Results.add_training_metric(results, %{step: 1, loss: 1.0})

      # Add evaluation results
      results = Results.add_evaluation_result(results, %{
        adapter_name: "v1",
        metrics: %{schema_compliance: 0.96}
      })

      # Get summaries
      training_summary = Results.summarize_training(results)
      evaluation_summary = Results.summarize_evaluation(results)

      # Validate against targets
      validation = Results.validate_against_targets(results, targets)

  """

  @type t :: %__MODULE__{
          experiment_id: String.t(),
          training_metrics: [map()],
          evaluation_metrics: [map()],
          checkpoints: [map()],
          created_at: DateTime.t()
        }

  defstruct [
    :experiment_id,
    :created_at,
    training_metrics: [],
    evaluation_metrics: [],
    checkpoints: []
  ]

  @doc """
  Creates a new results container.

  ## Options

  - `:experiment_id` - ID of the parent experiment

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> results.experiment_id
      "exp-123"
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      experiment_id: Keyword.get(opts, :experiment_id),
      training_metrics: [],
      evaluation_metrics: [],
      checkpoints: [],
      created_at: DateTime.utc_now()
    }
  end

  @doc """
  Adds a training metric to the results.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> results = Crucible.Tinkex.Results.add_training_metric(results, %{step: 1, loss: 1.0})
      iex> length(results.training_metrics)
      1
  """
  @spec add_training_metric(t(), map()) :: t()
  def add_training_metric(%__MODULE__{} = results, metric) when is_map(metric) do
    metric_with_timestamp = Map.put_new(metric, :timestamp, DateTime.utc_now())
    %{results | training_metrics: results.training_metrics ++ [metric_with_timestamp]}
  end

  @doc """
  Adds an evaluation result to the results.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> eval = %{adapter_name: "v1", metrics: %{accuracy: 0.95}}
      iex> results = Crucible.Tinkex.Results.add_evaluation_result(results, eval)
      iex> length(results.evaluation_metrics)
      1
  """
  @spec add_evaluation_result(t(), map()) :: t()
  def add_evaluation_result(%__MODULE__{} = results, evaluation) when is_map(evaluation) do
    eval_with_timestamp = Map.put_new(evaluation, :timestamp, DateTime.utc_now())
    %{results | evaluation_metrics: results.evaluation_metrics ++ [eval_with_timestamp]}
  end

  @doc """
  Adds a checkpoint record to the results.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> results = Crucible.Tinkex.Results.add_checkpoint(results, %{name: "step_100", step: 100})
      iex> length(results.checkpoints)
      1
  """
  @spec add_checkpoint(t(), map()) :: t()
  def add_checkpoint(%__MODULE__{} = results, checkpoint) when is_map(checkpoint) do
    checkpoint_with_timestamp = Map.put_new(checkpoint, :timestamp, DateTime.utc_now())
    %{results | checkpoints: results.checkpoints ++ [checkpoint_with_timestamp]}
  end

  @doc """
  Calculates summary statistics for training metrics.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> results = Crucible.Tinkex.Results.add_training_metric(results, %{step: 1, loss: 1.0, citation_invalid_rate: 0.0})
      iex> summary = Crucible.Tinkex.Results.summarize_training(results)
      iex> summary.total_steps
      1
  """
  @spec summarize_training(t()) :: map()
  def summarize_training(%__MODULE__{training_metrics: []}) do
    %{
      mean_loss: nil,
      final_loss: nil,
      loss_reduction: nil,
      mean_citation_invalid_rate: nil,
      total_steps: 0
    }
  end

  def summarize_training(%__MODULE__{training_metrics: metrics}) do
    losses = Enum.map(metrics, &Map.get(&1, :loss, 0))
    citation_rates = Enum.map(metrics, &Map.get(&1, :citation_invalid_rate, 0))

    mean_loss = Enum.sum(losses) / length(losses)
    first_loss = hd(losses)
    final_loss = List.last(losses)

    %{
      mean_loss: mean_loss,
      final_loss: final_loss,
      loss_reduction: first_loss - final_loss,
      min_loss: Enum.min(losses),
      max_loss: Enum.max(losses),
      mean_citation_invalid_rate: Enum.sum(citation_rates) / length(citation_rates),
      total_steps: length(metrics)
    }
  end

  @doc """
  Calculates summary statistics for evaluation metrics.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> summary = Crucible.Tinkex.Results.summarize_evaluation(results)
      iex> summary.count
      0
  """
  @spec summarize_evaluation(t()) :: map()
  def summarize_evaluation(%__MODULE__{evaluation_metrics: []}) do
    %{count: 0}
  end

  def summarize_evaluation(%__MODULE__{evaluation_metrics: evals}) do
    count = length(evals)

    # Extract all unique metric keys
    all_metrics =
      evals
      |> Enum.flat_map(fn eval -> Map.keys(eval.metrics) end)
      |> Enum.uniq()

    # Calculate mean for each metric
    means =
      Enum.reduce(all_metrics, %{}, fn metric_key, acc ->
        values =
          evals
          |> Enum.map(fn eval -> Map.get(eval.metrics, metric_key, 0) end)

        mean = Enum.sum(values) / length(values)
        Map.put(acc, :"mean_#{metric_key}", mean)
      end)

    Map.merge(%{count: count}, means)
  end

  @doc """
  Finds the best run by a target metric.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> Crucible.Tinkex.Results.best_run(results, :accuracy)
      nil
  """
  @spec best_run(t(), atom()) :: map() | nil
  def best_run(%__MODULE__{evaluation_metrics: []}, _metric), do: nil

  def best_run(%__MODULE__{evaluation_metrics: evals}, metric) do
    Enum.max_by(evals, fn eval ->
      Map.get(eval.metrics, metric, 0)
    end)
  end

  @doc """
  Generates a report-ready data structure.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> data = Crucible.Tinkex.Results.to_report_data(results)
      iex> data.experiment_id
      "exp-123"
  """
  @spec to_report_data(t()) :: map()
  def to_report_data(%__MODULE__{} = results) do
    %{
      experiment_id: results.experiment_id,
      training_summary: summarize_training(results),
      evaluation_summary: summarize_evaluation(results),
      checkpoints: results.checkpoints,
      generated_at: DateTime.utc_now()
    }
  end

  @doc """
  Exports metrics to CSV format.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> results = Crucible.Tinkex.Results.add_training_metric(results, %{step: 1, loss: 1.0, citation_invalid_rate: 0.0})
      iex> csv = Crucible.Tinkex.Results.to_csv(results, :training)
      iex> String.contains?(csv, "step")
      true
  """
  @spec to_csv(t(), :training | :evaluation) :: String.t()
  def to_csv(%__MODULE__{training_metrics: metrics}, :training) do
    if metrics == [] do
      ""
    else
      headers = [:step, :loss, :citation_invalid_rate]
      header_row = Enum.join(headers, ",")

      data_rows =
        Enum.map(metrics, fn metric ->
          values = Enum.map(headers, fn h -> Map.get(metric, h, "") end)
          Enum.join(values, ",")
        end)

      [header_row | data_rows] |> Enum.join("\n")
    end
  end

  def to_csv(%__MODULE__{evaluation_metrics: evals}, :evaluation) do
    if evals == [] do
      ""
    else
      # Get all metric keys from first evaluation
      first_metrics = hd(evals).metrics
      metric_keys = Map.keys(first_metrics)

      headers = [:adapter_name | metric_keys]
      header_row = Enum.join(headers, ",")

      data_rows =
        Enum.map(evals, fn eval ->
          values = [
            eval.adapter_name | Enum.map(metric_keys, fn k -> Map.get(eval.metrics, k, "") end)
          ]

          Enum.join(values, ",")
        end)

      [header_row | data_rows] |> Enum.join("\n")
    end
  end

  @doc """
  Validates evaluation results against quality targets.

  ## Examples

      iex> results = Crucible.Tinkex.Results.new(experiment_id: "exp-123")
      iex> targets = %{accuracy: 0.90}
      iex> validation = Crucible.Tinkex.Results.validate_against_targets(results, targets)
      iex> validation.passed_count
      0
  """
  @spec validate_against_targets(t(), map()) :: map()
  def validate_against_targets(%__MODULE__{evaluation_metrics: []}, targets) do
    %{
      passed: false,
      passed_count: 0,
      total_count: map_size(targets),
      assessments: [],
      summary: "No evaluation results to validate"
    }
  end

  def validate_against_targets(%__MODULE__{evaluation_metrics: evals}, targets) do
    # Use the best/latest evaluation
    latest = List.last(evals)
    metrics = latest.metrics

    assessments =
      Enum.map(targets, fn {metric, target} ->
        actual = Map.get(metrics, metric, 0)
        passed = actual >= target

        %{
          metric: metric,
          target: target,
          actual: actual,
          passed: passed,
          delta: Float.round(actual - target, 3)
        }
      end)

    passed_count = Enum.count(assessments, & &1.passed)
    total_count = length(assessments)
    all_passed = passed_count == total_count

    %{
      passed: all_passed,
      passed_count: passed_count,
      total_count: total_count,
      assessments: assessments,
      summary: "#{passed_count}/#{total_count} quality targets met"
    }
  end
end
