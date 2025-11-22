defmodule Crucible.Telemetry.DashboardData do
  @moduledoc """
  Provides formatted data for crucible_ui dashboard.

  This module retrieves and formats telemetry data for visualization,
  supporting various chart types and real-time streaming.

  ## Usage

      # Get loss curve for an experiment
      curve = DashboardData.loss_curve("exp-123", smoothing: 0.9)

      # Format for chart
      chart_data = DashboardData.format_for_chart(curve, :line)

      # Generate VegaLite specification
      spec = DashboardData.to_vega_lite_spec(curve, :line)
  """

  alias Crucible.Telemetry.MLMetrics

  @type chart_type :: :line | :bar | :scatter | :heatmap

  # Public API

  @doc """
  Gets training progress data for an experiment.
  """
  @spec training_progress(String.t(), keyword()) :: map()
  def training_progress(experiment_id, _opts \\ []) do
    case get_collector(experiment_id) do
      nil ->
        %{current_step: 0, current_epoch: 0, total_steps: 0}

      collector ->
        metrics = MLMetrics.get_metrics(collector, type: :training)
        summary = MLMetrics.get_training_summary(collector)

        if metrics == [] do
          %{current_step: 0, current_epoch: 0, total_steps: 0}
        else
          latest = hd(metrics)

          %{
            current_step: latest.step,
            current_epoch: latest.epoch,
            total_steps: summary.total_steps,
            loss: latest.loss,
            grad_norm: latest.grad_norm,
            checkpoints: length(summary.checkpoints)
          }
        end
    end
  end

  @doc """
  Gets loss curve data for an experiment.

  ## Options

  - `:smoothing` - Exponential smoothing factor (0-1)
  - `:max_points` - Maximum number of points to return
  """
  @spec loss_curve(String.t(), keyword()) :: [map()]
  def loss_curve(experiment_id, opts \\ []) do
    smoothing = Keyword.get(opts, :smoothing, 0)
    max_points = Keyword.get(opts, :max_points)

    case get_collector(experiment_id) do
      nil ->
        []

      collector ->
        metrics =
          MLMetrics.get_metrics(collector, type: :training)
          |> Enum.sort_by(& &1.step)

        points =
          metrics
          |> Enum.map(fn m -> %{step: m.step, loss: m.loss} end)
          |> apply_smoothing(smoothing)
          |> maybe_downsample(max_points)

        points
    end
  end

  @doc """
  Gets quality metrics summary for an experiment.
  """
  @spec quality_metrics(String.t(), keyword()) :: map()
  def quality_metrics(experiment_id, _opts \\ []) do
    case get_collector(experiment_id) do
      nil ->
        %{loss: %{}}

      collector ->
        aggregates = MLMetrics.get_aggregates(collector, :loss)
        metrics = MLMetrics.get_metrics(collector, type: :training)

        if metrics == [] do
          %{loss: %{current: nil, best: nil, trend: :unknown}}
        else
          sorted = Enum.sort_by(metrics, & &1.step)
          current = List.last(sorted).loss
          best = Enum.min_by(sorted, & &1.loss).loss

          # Calculate trend (last 10 points)
          trend = calculate_trend(sorted)

          %{
            loss: %{
              current: current,
              best: best,
              mean: aggregates[:mean],
              trend: trend
            }
          }
        end
    end
  end

  @doc """
  Gets ensemble performance data.
  """
  @spec ensemble_performance(String.t(), keyword()) :: map()
  def ensemble_performance(experiment_id, _opts \\ []) do
    case get_collector(experiment_id) do
      nil ->
        %{}

      collector ->
        metrics = MLMetrics.get_metrics(collector, type: :ensemble)

        if metrics == [] do
          %{mean_latency: 0, mean_accuracy: 0, count: 0}
        else
          latencies = Enum.map(metrics, & &1.measurements.latency)

          accuracies =
            metrics
            |> Enum.map(& &1.measurements[:accuracy])
            |> Enum.reject(&is_nil/1)

          %{
            mean_latency: Enum.sum(latencies) / length(latencies),
            mean_accuracy:
              if(accuracies != [], do: Enum.sum(accuracies) / length(accuracies), else: 0),
            count: length(metrics),
            strategies: metrics |> Enum.map(& &1.metadata.strategy) |> Enum.uniq()
          }
        end
    end
  end

  @doc """
  Gets latency distribution data.
  """
  @spec latency_distribution(String.t(), keyword()) :: map()
  def latency_distribution(experiment_id, opts \\ []) do
    bins = Keyword.get(opts, :bins, 20)

    case get_collector(experiment_id) do
      nil ->
        %{histogram: [], percentiles: %{}}

      collector ->
        # Get inference metrics for all models
        metrics = MLMetrics.get_metrics(collector, type: :inference)
        latencies = Enum.map(metrics, & &1.measurements[:latency]) |> Enum.reject(&is_nil/1)

        # Also check dedicated inference stats
        inference_latencies = get_all_inference_latencies(collector)
        all_latencies = latencies ++ inference_latencies

        if all_latencies == [] do
          %{histogram: [], percentiles: %{}}
        else
          histogram = build_histogram(all_latencies, bins)
          sorted = Enum.sort(all_latencies)

          %{
            histogram: histogram,
            percentiles: %{
              p50: percentile(sorted, 50),
              p90: percentile(sorted, 90),
              p95: percentile(sorted, 95),
              p99: percentile(sorted, 99)
            },
            min: Enum.min(sorted),
            max: Enum.max(sorted),
            count: length(all_latencies)
          }
        end
    end
  end

  @doc """
  Compares a metric across multiple experiments.
  """
  @spec model_comparison([String.t()], atom()) :: map()
  def model_comparison(experiment_ids, metric) do
    experiment_ids
    |> Enum.map(fn exp_id ->
      case get_collector(exp_id) do
        nil ->
          {exp_id, %{final: nil, best: nil, mean: nil}}

        collector ->
          metrics = MLMetrics.get_metrics(collector, type: :training)

          if metrics == [] do
            {exp_id, %{final: nil, best: nil, mean: nil}}
          else
            sorted = Enum.sort_by(metrics, & &1.step)
            values = Enum.map(sorted, &Map.get(&1, metric))

            {exp_id,
             %{
               final: List.last(values),
               best: Enum.min(values),
               mean: Enum.sum(values) / length(values),
               steps: length(values)
             }}
          end
      end
    end)
    |> Map.new()
  end

  @doc """
  Formats data for a specific chart type.
  """
  @spec format_for_chart([map()], chart_type()) :: map()
  def format_for_chart(data, :line) do
    %{
      x: Enum.map(data, & &1.step),
      y: Enum.map(data, & &1.loss)
    }
  end

  def format_for_chart(data, :bar) do
    %{
      labels: Enum.map(data, & &1.step),
      values: Enum.map(data, & &1.loss)
    }
  end

  def format_for_chart(data, :scatter) do
    %{
      points: Enum.map(data, fn d -> {d.step, d.loss} end)
    }
  end

  def format_for_chart(data, :heatmap) do
    %{
      matrix: Enum.map(data, fn d -> [d.step, d.loss] end)
    }
  end

  @doc """
  Generates a VegaLite specification for the data.
  """
  @spec to_vega_lite_spec([map()], chart_type()) :: map()
  def to_vega_lite_spec(data, chart_type) do
    mark =
      case chart_type do
        :line -> "line"
        :bar -> "bar"
        :scatter -> "point"
        :heatmap -> "rect"
      end

    %{
      "$schema" => "https://vega.github.io/schema/vega-lite/v5.json",
      "data" => %{
        "values" => data
      },
      "mark" => mark,
      "encoding" => %{
        "x" => %{"field" => "step", "type" => "quantitative"},
        "y" => %{"field" => "loss", "type" => "quantitative"}
      }
    }
  end

  @doc """
  Subscribes to real-time updates for an experiment.
  """
  @spec subscribe(String.t()) :: :ok
  def subscribe(experiment_id) do
    try do
      Registry.register(Crucible.Telemetry.Registry, {:subscriber, experiment_id}, self())
      :ok
    rescue
      # Registry might not exist
      ArgumentError -> :ok
    end
  end

  @doc """
  Unsubscribes from real-time updates.
  """
  @spec unsubscribe(String.t()) :: :ok
  def unsubscribe(experiment_id) do
    try do
      Registry.unregister(Crucible.Telemetry.Registry, {:subscriber, experiment_id})
      :ok
    rescue
      # Registry might not exist
      ArgumentError -> :ok
    end
  end

  # Private helpers

  defp get_collector(experiment_id) do
    case Registry.lookup(Crucible.Telemetry.Registry, {:collector, experiment_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  rescue
    _ -> nil
  end

  defp apply_smoothing(points, 0), do: points
  defp apply_smoothing([], _), do: []

  defp apply_smoothing([first | rest], alpha) do
    {smoothed, _} =
      Enum.map_reduce(rest, first.loss, fn point, prev ->
        smoothed_loss = alpha * prev + (1 - alpha) * point.loss
        {%{point | loss: smoothed_loss}, smoothed_loss}
      end)

    [first | smoothed]
  end

  defp maybe_downsample(points, nil), do: points
  defp maybe_downsample(points, max) when length(points) <= max, do: points

  defp maybe_downsample(points, max) do
    step = length(points) / max

    points
    |> Enum.with_index()
    |> Enum.filter(fn {_, i} -> rem(trunc(i), trunc(step)) == 0 end)
    |> Enum.map(&elem(&1, 0))
    |> Enum.take(max)
  end

  defp calculate_trend(metrics) when length(metrics) < 2, do: :unknown

  defp calculate_trend(metrics) do
    recent = Enum.take(metrics, -10)
    losses = Enum.map(recent, & &1.loss)

    if length(losses) < 2 do
      :unknown
    else
      first_half = Enum.take(losses, div(length(losses), 2))
      second_half = Enum.drop(losses, div(length(losses), 2))

      avg_first = Enum.sum(first_half) / length(first_half)
      avg_second = Enum.sum(second_half) / length(second_half)

      cond do
        avg_second < avg_first * 0.95 -> :decreasing
        avg_second > avg_first * 1.05 -> :increasing
        true -> :stable
      end
    end
  end

  defp get_all_inference_latencies(_collector) do
    # Would iterate through all models, simplified for now
    []
  end

  defp build_histogram(values, bins) do
    min_val = Enum.min(values)
    max_val = Enum.max(values)
    range = max_val - min_val
    bin_width = if range == 0, do: 1, else: range / bins

    # Initialize bins
    bin_counts =
      for i <- 0..(bins - 1) do
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width

        count =
          Enum.count(values, fn v ->
            v >= bin_start and (v < bin_end or (i == bins - 1 and v <= bin_end))
          end)

        %{bin: i, start: bin_start, end: bin_end, count: count}
      end

    bin_counts
  end

  defp percentile(sorted, p) do
    k = (length(sorted) - 1) * p / 100
    f = floor(k)
    c = ceil(k)

    if f == c do
      Enum.at(sorted, trunc(k))
    else
      d0 = Enum.at(sorted, f) * (c - k)
      d1 = Enum.at(sorted, c) * (k - f)
      d0 + d1
    end
  end
end
