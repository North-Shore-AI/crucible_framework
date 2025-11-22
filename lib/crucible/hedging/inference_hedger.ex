defmodule Crucible.Hedging.InferenceHedger do
  @moduledoc """
  Hedging strategies for ML inference with Tinkex.

  Implements multiple hedging strategies to reduce tail latency in ML inference:
  - `:fixed` - Send backup after fixed delay
  - `:percentile` - Send backup at latency percentile (e.g., P75)
  - `:adaptive` - Adjust delay based on recent performance
  - `:workload_aware` - Consider model load when hedging

  ## Examples

      # Create a fixed delay hedger
      hedger = InferenceHedger.new(strategy: :fixed, delay_ms: 100)

      # Create percentile-based hedger
      hedger = InferenceHedger.new(strategy: :percentile, percentile: 75)

      # Dispatch inference with hedging
      {:ok, result, latency} = InferenceHedger.dispatch(
        hedger,
        clients,
        fn client -> Tinkex.SamplingClient.generate(client, prompt, params) end,
        timeout: 5000
      )
  """

  require Logger

  @type strategy :: :fixed | :percentile | :adaptive | :workload_aware
  @type result :: {:ok, response :: term(), latency_ms :: integer()} | {:error, term()}

  defstruct [
    :strategy,
    :delay_ms,
    :percentile,
    :history_window,
    :latency_history,
    :workload_config
  ]

  @type t :: %__MODULE__{
          strategy: strategy(),
          delay_ms: non_neg_integer(),
          percentile: non_neg_integer(),
          history_window: non_neg_integer(),
          latency_history: %{String.t() => [non_neg_integer()]},
          workload_config: map() | nil
        }

  # Public API

  @doc """
  Creates a new inference hedger with the given options.

  ## Options
    * `:strategy` - Hedging strategy (default: `:percentile`)
    * `:delay_ms` - Fixed delay in ms for `:fixed` strategy (default: 100)
    * `:percentile` - Percentile for `:percentile` strategy (default: 75)
    * `:history_window` - Number of latencies to track (default: 100)
    * `:workload_config` - Configuration for `:workload_aware` strategy
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      strategy: Keyword.get(opts, :strategy, :percentile),
      delay_ms: Keyword.get(opts, :delay_ms, 100),
      percentile: Keyword.get(opts, :percentile, 75),
      history_window: Keyword.get(opts, :history_window, 100),
      latency_history: %{},
      workload_config:
        Keyword.get(opts, :workload_config, %{
          high_load_multiplier: 1.5,
          low_load_threshold: 0.3
        })
    }
  end

  @doc """
  Dispatches inference with hedging strategy.

  ## Options
    * `:timeout` - Maximum time to wait for response (default: 30_000)
  """
  @spec dispatch(t(), [{map(), term()}], (term() -> {:ok, term()} | {:error, term()}), keyword()) ::
          result()
  def dispatch(hedger, clients, inference_fn, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 30_000)

    emit_hedging_telemetry(
      :dispatch,
      %{
        strategy: hedger.strategy,
        client_count: length(clients)
      },
      %{}
    )

    result =
      case hedger.strategy do
        :fixed ->
          fixed_hedging(clients, inference_fn, hedger.delay_ms, timeout)

        :percentile ->
          delay = compute_percentile_delay(hedger, clients)
          percentile_hedging(clients, inference_fn, delay, timeout)

        :adaptive ->
          delay = compute_adaptive_delay(hedger, clients)
          adaptive_hedging(clients, inference_fn, delay, timeout)

        :workload_aware ->
          workload_aware_hedging(clients, inference_fn, hedger.workload_config, timeout)
      end

    result
  end

  # Strategy implementations

  defp fixed_hedging(clients, inference_fn, delay_ms, timeout) do
    [{primary_adapter, primary_client} | rest] = clients

    # Start primary immediately
    primary_task =
      Task.async(fn ->
        start_time = System.monotonic_time(:millisecond)
        result = inference_fn.(primary_client)
        latency = System.monotonic_time(:millisecond) - start_time
        {primary_adapter, result, latency}
      end)

    # Start backup after delay
    backup_task =
      if rest != [] do
        Task.async(fn ->
          Process.sleep(delay_ms)

          emit_hedging_telemetry(:backup_sent, %{delay_ms: delay_ms}, %{
            primary: primary_adapter.name,
            backup: hd(rest) |> elem(0) |> Map.get(:name)
          })

          {backup_adapter, backup_client} = hd(rest)
          start_time = System.monotonic_time(:millisecond)
          result = inference_fn.(backup_client)
          latency = System.monotonic_time(:millisecond) - start_time
          {backup_adapter, result, latency}
        end)
      end

    tasks = [primary_task | if(backup_task, do: [backup_task], else: [])]

    await_first_success(tasks, timeout)
  end

  defp percentile_hedging(clients, inference_fn, delay_ms, timeout) do
    # Same as fixed but with computed delay
    fixed_hedging(clients, inference_fn, delay_ms, timeout)
  end

  defp adaptive_hedging(clients, inference_fn, delay_ms, timeout) do
    # Same implementation, delay computed adaptively
    fixed_hedging(clients, inference_fn, delay_ms, timeout)
  end

  defp workload_aware_hedging(clients, inference_fn, config, timeout) do
    # Sort clients by load (ascending)
    sorted_clients =
      Enum.sort_by(clients, fn {adapter, _client} ->
        Map.get(adapter, :load, 0.5)
      end)

    # Use least loaded as primary
    [{primary_adapter, primary_client} | rest] = sorted_clients

    # Adjust delay based on load
    load = Map.get(primary_adapter, :load, 0.5)
    base_delay = 100

    delay_ms =
      if load > 0.7 do
        trunc(base_delay * config.high_load_multiplier)
      else
        base_delay
      end

    # Start primary
    primary_task =
      Task.async(fn ->
        start_time = System.monotonic_time(:millisecond)
        result = inference_fn.(primary_client)
        latency = System.monotonic_time(:millisecond) - start_time
        {primary_adapter, result, latency}
      end)

    # Start backup
    backup_task =
      if rest != [] do
        Task.async(fn ->
          Process.sleep(delay_ms)
          {backup_adapter, backup_client} = hd(rest)
          start_time = System.monotonic_time(:millisecond)
          result = inference_fn.(backup_client)
          latency = System.monotonic_time(:millisecond) - start_time
          {backup_adapter, result, latency}
        end)
      end

    tasks = [primary_task | if(backup_task, do: [backup_task], else: [])]

    await_first_success(tasks, timeout)
  end

  defp await_first_success(tasks, timeout) do
    # Use Task.await_many with timeout, but we want the first completed
    # Instead, poll repeatedly until one completes
    deadline = System.monotonic_time(:millisecond) + timeout
    await_first_loop(tasks, deadline)
  end

  defp await_first_loop(tasks, deadline) do
    remaining_time = deadline - System.monotonic_time(:millisecond)

    if remaining_time <= 0 do
      # Timeout - kill all tasks
      Enum.each(tasks, &Task.shutdown(&1, :brutal_kill))
      {:error, :timeout}
    else
      # Check each task for completion
      result =
        Enum.find_value(tasks, fn task ->
          case Task.yield(task, 0) do
            {:ok, {adapter, {:ok, response}, latency}} ->
              {adapter, response, latency}

            {:ok, {_adapter, {:error, _reason}, _latency}} ->
              # Task completed with error, don't return yet
              nil

            nil ->
              # Not done yet
              nil

            {:exit, _reason} ->
              nil
          end
        end)

      case result do
        {adapter, response, latency} ->
          # Got a successful result - kill remaining tasks
          Enum.each(tasks, &Task.shutdown(&1, :brutal_kill))

          emit_hedging_telemetry(:winner, %{latency_ms: latency}, %{
            model: adapter.name
          })

          {:ok, response, latency}

        nil ->
          # No result yet, sleep briefly and retry
          Process.sleep(1)
          await_first_loop(tasks, deadline)
      end
    end
  end

  defp compute_percentile_delay(hedger, clients) do
    [{primary_adapter, _} | _] = clients
    model_name = primary_adapter.name

    case Map.get(hedger.latency_history, model_name) do
      nil -> hedger.delay_ms
      [] -> hedger.delay_ms
      history -> calculate_percentile(history, hedger.percentile)
    end
  end

  defp compute_adaptive_delay(hedger, _clients) do
    # Use recent performance across all models
    all_latencies =
      hedger.latency_history
      |> Map.values()
      |> List.flatten()

    if length(all_latencies) > 10 do
      calculate_percentile(all_latencies, hedger.percentile)
    else
      hedger.delay_ms
    end
  end

  # History tracking

  @doc """
  Records a latency measurement for a model.
  """
  @spec record_latency(t(), String.t(), non_neg_integer()) :: t()
  def record_latency(hedger, model_name, latency_ms) do
    history = Map.get(hedger.latency_history, model_name, [])
    updated = [latency_ms | history] |> Enum.take(hedger.history_window)

    %{hedger | latency_history: Map.put(hedger.latency_history, model_name, updated)}
  end

  @doc """
  Gets latency statistics for a model.
  """
  @spec get_latency_stats(t(), String.t()) :: map()
  def get_latency_stats(hedger, model_name) do
    case Map.get(hedger.latency_history, model_name, []) do
      [] ->
        %{count: 0, min: 0, max: 0, mean: 0.0, p50: 0, p75: 0, p90: 0, p99: 0}

      history ->
        sorted = Enum.sort(history)
        count = length(sorted)

        %{
          count: count,
          min: Enum.min(sorted),
          max: Enum.max(sorted),
          mean: Enum.sum(sorted) / count,
          p50: calculate_percentile(sorted, 50),
          p75: calculate_percentile(sorted, 75),
          p90: calculate_percentile(sorted, 90),
          p99: calculate_percentile(sorted, 99)
        }
    end
  end

  @doc """
  Calculates the given percentile from a list of values.

  Uses linear interpolation for more accurate percentile calculation.
  """
  @spec calculate_percentile([number()], non_neg_integer()) :: number()
  def calculate_percentile([], _percentile), do: 100

  def calculate_percentile([single], _percentile), do: single

  def calculate_percentile(history, percentile) do
    sorted = Enum.sort(history)
    count = length(sorted)

    # Use nearest-rank method for percentiles
    # For percentile P and N values, index = ceil(P/100 * N) - 1
    index = ceil(percentile / 100 * count) - 1
    index = max(0, min(index, count - 1))

    Enum.at(sorted, index)
  end

  # Telemetry

  defp emit_hedging_telemetry(event, measurements, metadata) do
    :telemetry.execute(
      [:crucible, :hedging, event],
      Map.put(measurements, :timestamp, System.system_time(:millisecond)),
      metadata
    )
  end
end
