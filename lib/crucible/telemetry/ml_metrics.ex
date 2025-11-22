defmodule Crucible.Telemetry.MLMetrics do
  @moduledoc """
  Collects and aggregates ML training and inference metrics.

  This module provides a GenServer-based metrics collector that stores metrics
  in ETS for high performance. It supports training steps, checkpoints,
  inference tracking, and generic event recording.

  ## Usage

      {:ok, collector} = MLMetrics.start_link(experiment_id: "exp-123")

      # Record training metrics
      MLMetrics.record_training_step(collector, 1, 1, 2.5, 0.1)

      # Get aggregates
      aggregates = MLMetrics.get_aggregates(collector, :loss)
      # => %{mean: 2.5, min: 2.5, max: 2.5, p50: 2.5, p95: 2.5, p99: 2.5, count: 1}
  """

  use GenServer

  @type metric_type :: :training | :inference | :ensemble | :hedging | :checkpoint

  defstruct [
    :experiment_id,
    :metrics_table,
    :inference_table,
    :checkpoints,
    :flush_interval,
    :subscribers
  ]

  # Client API

  @doc """
  Starts the metrics collector.

  ## Options

  - `:experiment_id` - Required experiment identifier
  - `:flush_interval` - Optional auto-flush interval in ms
  - `:name` - Optional process name
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    name = Keyword.get(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Records a generic event with measurements and metadata.
  """
  @spec record(GenServer.server(), atom(), map(), map()) :: :ok
  def record(collector, event_type, measurements, metadata) do
    GenServer.cast(collector, {:record, event_type, measurements, metadata})
  end

  @doc """
  Gets all recorded metrics, optionally filtered.

  ## Options

  - `:type` - Filter by metric type
  - `:epoch` - Filter by epoch
  - `:from` - Filter from timestamp
  - `:to` - Filter to timestamp
  """
  @spec get_metrics(GenServer.server(), keyword()) :: [map()]
  def get_metrics(collector, opts \\ []) do
    GenServer.call(collector, {:get_metrics, opts})
  end

  @doc """
  Gets aggregated statistics for a metric.

  Returns mean, min, max, and percentiles (p50, p95, p99).
  """
  @spec get_aggregates(GenServer.server(), atom()) :: map()
  def get_aggregates(collector, metric_name) do
    GenServer.call(collector, {:get_aggregates, metric_name})
  end

  @doc """
  Flushes all metrics from the collector.
  """
  @spec flush(GenServer.server()) :: :ok
  def flush(collector) do
    GenServer.call(collector, :flush)
  end

  @doc """
  Resets the collector state completely.
  """
  @spec reset(GenServer.server()) :: :ok
  def reset(collector) do
    GenServer.call(collector, :reset)
  end

  @doc """
  Records a training step with loss and gradient norm.
  """
  @spec record_training_step(GenServer.server(), pos_integer(), pos_integer(), float(), float()) ::
          :ok
  def record_training_step(collector, step, epoch, loss, grad_norm) do
    GenServer.cast(collector, {:training_step, step, epoch, loss, grad_norm})
  end

  @doc """
  Records a checkpoint with associated metrics.
  """
  @spec record_checkpoint(GenServer.server(), pos_integer(), map()) :: :ok
  def record_checkpoint(collector, step, metrics) do
    GenServer.cast(collector, {:checkpoint, step, metrics})
  end

  @doc """
  Gets a summary of training progress.
  """
  @spec get_training_summary(GenServer.server()) :: map()
  def get_training_summary(collector) do
    GenServer.call(collector, :get_training_summary)
  end

  @doc """
  Records an inference call with latency and token count.
  """
  @spec record_inference(GenServer.server(), String.t(), number(), number()) :: :ok
  def record_inference(collector, model, latency, tokens) do
    GenServer.cast(collector, {:inference, model, latency, tokens})
  end

  @doc """
  Gets inference statistics for a model.
  """
  @spec get_inference_stats(GenServer.server(), String.t()) :: map()
  def get_inference_stats(collector, model) do
    GenServer.call(collector, {:get_inference_stats, model})
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    experiment_id = Keyword.fetch!(opts, :experiment_id)
    flush_interval = Keyword.get(opts, :flush_interval)

    # Create ETS tables for high-performance storage
    metrics_table = :ets.new(:metrics, [:ordered_set, :private])
    inference_table = :ets.new(:inference, [:bag, :private])

    state = %__MODULE__{
      experiment_id: experiment_id,
      metrics_table: metrics_table,
      inference_table: inference_table,
      checkpoints: [],
      flush_interval: flush_interval,
      subscribers: []
    }

    # Register globally for experiment if registry exists
    try do
      Registry.register(Crucible.Telemetry.Registry, {:collector, experiment_id}, self())
    rescue
      # Registry might not exist
      ArgumentError -> :ok
    end

    {:ok, state}
  end

  @impl true
  def handle_cast({:record, event_type, measurements, metadata}, state) do
    entry = %{
      type: event_type,
      measurements: measurements,
      metadata: metadata,
      timestamp: DateTime.utc_now()
    }

    key = System.unique_integer([:monotonic, :positive])
    :ets.insert(state.metrics_table, {key, entry})

    notify_subscribers(state, {:metric, entry})

    {:noreply, state}
  end

  @impl true
  def handle_cast({:training_step, step, epoch, loss, grad_norm}, state) do
    entry = %{
      type: :training,
      step: step,
      epoch: epoch,
      loss: loss,
      grad_norm: grad_norm,
      timestamp: DateTime.utc_now()
    }

    :ets.insert(state.metrics_table, {step, entry})
    notify_subscribers(state, {:training_step, entry})

    {:noreply, state}
  end

  @impl true
  def handle_cast({:checkpoint, step, metrics}, state) do
    checkpoint = %{
      step: step,
      metrics: metrics,
      timestamp: DateTime.utc_now()
    }

    new_checkpoints = [checkpoint | state.checkpoints]
    notify_subscribers(state, {:checkpoint, checkpoint})

    {:noreply, %{state | checkpoints: new_checkpoints}}
  end

  @impl true
  def handle_cast({:inference, model, latency, tokens}, state) do
    entry = {model, latency, tokens, DateTime.utc_now()}
    :ets.insert(state.inference_table, entry)

    {:noreply, state}
  end

  @impl true
  def handle_call({:get_metrics, opts}, _from, state) do
    metrics =
      :ets.tab2list(state.metrics_table)
      |> Enum.map(fn {_key, entry} -> entry end)
      |> Enum.sort_by(& &1.timestamp, {:desc, DateTime})
      |> filter_metrics(opts)

    {:reply, metrics, state}
  end

  @impl true
  def handle_call({:get_aggregates, metric_name}, _from, state) do
    values =
      :ets.tab2list(state.metrics_table)
      |> Enum.map(fn {_key, entry} -> Map.get(entry, metric_name) end)
      |> Enum.reject(&is_nil/1)

    aggregates = calculate_aggregates(values)
    {:reply, aggregates, state}
  end

  @impl true
  def handle_call(:flush, _from, state) do
    :ets.delete_all_objects(state.metrics_table)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    :ets.delete_all_objects(state.metrics_table)
    :ets.delete_all_objects(state.inference_table)
    {:reply, :ok, %{state | checkpoints: []}}
  end

  @impl true
  def handle_call(:get_training_summary, _from, state) do
    metrics =
      :ets.tab2list(state.metrics_table)
      |> Enum.map(fn {_key, entry} -> entry end)
      |> Enum.filter(&(&1.type == :training))

    summary = %{
      total_steps: length(metrics),
      epochs: metrics |> Enum.map(& &1.epoch) |> Enum.max(fn -> 0 end),
      checkpoints: Enum.reverse(state.checkpoints),
      final_loss:
        if(metrics != [], do: List.last(Enum.sort_by(metrics, & &1.step)).loss, else: nil),
      best_loss: if(metrics != [], do: metrics |> Enum.map(& &1.loss) |> Enum.min(), else: nil)
    }

    {:reply, summary, state}
  end

  @impl true
  def handle_call({:get_inference_stats, model}, _from, state) do
    entries =
      :ets.lookup(state.inference_table, model)
      |> Enum.map(fn {_model, latency, tokens, _ts} -> {latency, tokens} end)

    stats =
      if entries == [] do
        %{count: 0}
      else
        latencies = Enum.map(entries, &elem(&1, 0))
        tokens = Enum.map(entries, &elem(&1, 1))

        %{
          count: length(entries),
          mean_latency: Enum.sum(latencies) / length(latencies),
          min_latency: Enum.min(latencies),
          max_latency: Enum.max(latencies),
          total_tokens: Enum.sum(tokens),
          mean_tokens: Enum.sum(tokens) / length(tokens)
        }
      end

    {:reply, stats, state}
  end

  @impl true
  def terminate(_reason, state) do
    :ets.delete(state.metrics_table)
    :ets.delete(state.inference_table)
    :ok
  end

  # Private helpers

  defp filter_metrics(metrics, opts) do
    metrics
    |> maybe_filter_type(opts[:type])
    |> maybe_filter_epoch(opts[:epoch])
  end

  defp maybe_filter_type(metrics, nil), do: metrics

  defp maybe_filter_type(metrics, type) do
    Enum.filter(metrics, &(&1.type == type))
  end

  defp maybe_filter_epoch(metrics, nil), do: metrics

  defp maybe_filter_epoch(metrics, epoch) do
    Enum.filter(metrics, &(Map.get(&1, :epoch) == epoch))
  end

  defp calculate_aggregates([]), do: %{count: 0}

  defp calculate_aggregates(values) do
    sorted = Enum.sort(values)
    count = length(sorted)

    %{
      count: count,
      mean: Enum.sum(values) / count,
      min: hd(sorted),
      max: List.last(sorted),
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
      p99: percentile(sorted, 99)
    }
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

  defp notify_subscribers(%{subscribers: subscribers}, event) do
    for pid <- subscribers do
      send(pid, {:telemetry_event, event})
    end
  end
end
