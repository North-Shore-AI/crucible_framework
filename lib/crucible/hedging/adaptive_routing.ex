defmodule Crucible.Hedging.AdaptiveRouting do
  @moduledoc """
  Routes requests based on model performance metrics.

  Implements multiple routing strategies:
  - `:round_robin` - Distribute requests evenly
  - `:least_loaded` - Route to model with fewest active requests
  - `:best_performing` - Route based on success rate and latency
  - `:weighted` - Probabilistic routing based on model weights

  ## Examples

      # Start router with models
      {:ok, router} = AdaptiveRouting.start_link(
        models: [
          %{name: "model1", weight: 0.6},
          %{name: "model2", weight: 0.4}
        ],
        strategy: :best_performing
      )

      # Select best model
      {:ok, model} = AdaptiveRouting.select_model(router)

      # Record request result
      AdaptiveRouting.record_request(router, "model1", true, 150)
  """

  use GenServer

  require Logger

  @type routing_strategy :: :round_robin | :least_loaded | :best_performing | :weighted

  defstruct [
    :models,
    :strategy,
    :metrics,
    :weights,
    :round_robin_index,
    :active_requests
  ]

  @type t :: %__MODULE__{
          models: [map()],
          strategy: routing_strategy(),
          metrics: %{String.t() => map()},
          weights: %{String.t() => float()},
          round_robin_index: non_neg_integer(),
          active_requests: %{String.t() => non_neg_integer()}
        }

  # Client API

  @doc """
  Starts the adaptive routing server.

  ## Options
    * `:models` - List of model specs with name and weight (required)
    * `:strategy` - Routing strategy (default: `:round_robin`)
  """
  @spec start_link(keyword() | map()) :: GenServer.on_start()
  def start_link(opts) do
    %{models: models, strategy: strategy} = normalize_init_args(opts)

    GenServer.start_link(__MODULE__, %{models: models, strategy: strategy})
  end

  @doc """
  Selects the best model based on current routing strategy.
  """
  @spec select_model(GenServer.server(), keyword()) :: {:ok, map()} | {:error, term()}
  def select_model(router, opts \\ []) do
    GenServer.call(router, {:select_model, opts})
  end

  @doc """
  Updates performance metrics for a model.
  """
  @spec update_metrics(GenServer.server(), String.t(), map()) :: :ok
  def update_metrics(router, model_name, metrics) do
    GenServer.cast(router, {:update_metrics, model_name, metrics})
  end

  @doc """
  Gets statistics for all models.
  """
  @spec get_model_stats(GenServer.server()) :: map()
  def get_model_stats(router) do
    GenServer.call(router, :get_model_stats)
  end

  @doc """
  Changes the routing strategy.
  """
  @spec set_strategy(GenServer.server(), routing_strategy()) :: :ok
  def set_strategy(router, strategy) do
    GenServer.cast(router, {:set_strategy, strategy})
  end

  @doc """
  Records a request result for a model.
  """
  @spec record_request(GenServer.server(), String.t(), boolean() | :started, non_neg_integer()) ::
          :ok
  def record_request(router, model_name, success_or_started, latency_ms) do
    GenServer.cast(router, {:record_request, model_name, success_or_started, latency_ms})
  end

  @doc """
  Gets the success rate for a model.
  """
  @spec get_success_rate(GenServer.server(), String.t()) :: float()
  def get_success_rate(router, model_name) do
    GenServer.call(router, {:get_success_rate, model_name})
  end

  @doc """
  Gets the average latency for a model.
  """
  @spec get_avg_latency(GenServer.server(), String.t()) :: float()
  def get_avg_latency(router, model_name) do
    GenServer.call(router, {:get_avg_latency, model_name})
  end

  # Server Callbacks

  @impl true
  def init(init_arg) do
    %{models: models, strategy: strategy} = normalize_init_args(init_arg)

    metrics =
      models
      |> Enum.map(fn model -> {model.name, initial_metrics()} end)
      |> Map.new()

    weights =
      models
      |> Enum.map(fn model -> {model.name, model.weight} end)
      |> Map.new()

    active_requests =
      models
      |> Enum.map(fn model -> {model.name, 0} end)
      |> Map.new()

    state = %__MODULE__{
      models: models,
      strategy: strategy,
      metrics: metrics,
      weights: weights,
      round_robin_index: 0,
      active_requests: active_requests
    }

    {:ok, state}
  end

  defp normalize_init_args(%{models: models} = map) do
    %{
      models: models,
      strategy: Map.get(map, :strategy, :round_robin)
    }
  end

  defp normalize_init_args(opts) when is_list(opts) do
    opts
    |> Map.new()
    |> normalize_init_args()
  end

  defp normalize_init_args(other) do
    raise ArgumentError,
          "AdaptiveRouting expected init args with :models and optional :strategy, got: #{inspect(other)}"
  end

  @impl true
  def handle_call({:select_model, _opts}, _from, state) do
    {model, new_state} =
      case state.strategy do
        :round_robin -> route_round_robin(state)
        :least_loaded -> route_least_loaded(state)
        :best_performing -> route_best_performing(state)
        :weighted -> route_weighted(state)
      end

    emit_routing_telemetry(:model_selected, %{}, %{
      model: model.name,
      strategy: state.strategy
    })

    {:reply, {:ok, model}, new_state}
  end

  @impl true
  def handle_call(:get_model_stats, _from, state) do
    {:reply, state.metrics, state}
  end

  @impl true
  def handle_call({:get_success_rate, model_name}, _from, state) do
    rate =
      case Map.get(state.metrics, model_name) do
        nil ->
          1.0

        %{total_requests: 0} ->
          1.0

        %{total_requests: total, successful_requests: successful} ->
          successful / total
      end

    {:reply, rate, state}
  end

  @impl true
  def handle_call({:get_avg_latency, model_name}, _from, state) do
    avg =
      case Map.get(state.metrics, model_name) do
        nil ->
          0.0

        %{total_requests: 0} ->
          0.0

        %{total_latency_ms: total, total_requests: count} ->
          total / count
      end

    {:reply, avg, state}
  end

  @impl true
  def handle_cast({:update_metrics, model_name, new_metrics}, state) do
    current = Map.get(state.metrics, model_name, initial_metrics())

    updated = %{
      current
      | total_requests: current.total_requests + 1,
        total_latency_ms: current.total_latency_ms + Map.get(new_metrics, :latency_ms, 0),
        successful_requests:
          current.successful_requests + if(new_metrics.success, do: 1, else: 0),
        avg_latency_ms:
          (current.total_latency_ms + Map.get(new_metrics, :latency_ms, 0)) /
            (current.total_requests + 1)
    }

    new_metrics_map = Map.put(state.metrics, model_name, updated)
    {:noreply, %{state | metrics: new_metrics_map}}
  end

  @impl true
  def handle_cast({:set_strategy, strategy}, state) do
    {:noreply, %{state | strategy: strategy}}
  end

  @impl true
  def handle_cast({:record_request, model_name, :started, _latency_ms}, state) do
    active = Map.update(state.active_requests, model_name, 1, &(&1 + 1))
    {:noreply, %{state | active_requests: active}}
  end

  @impl true
  def handle_cast({:record_request, model_name, success, latency_ms}, state)
      when is_boolean(success) do
    current = Map.get(state.metrics, model_name, initial_metrics())

    updated = %{
      current
      | total_requests: current.total_requests + 1,
        total_latency_ms: current.total_latency_ms + latency_ms,
        successful_requests: current.successful_requests + if(success, do: 1, else: 0),
        avg_latency_ms: (current.total_latency_ms + latency_ms) / (current.total_requests + 1)
    }

    new_metrics = Map.put(state.metrics, model_name, updated)

    # Decrement active requests
    active = Map.update(state.active_requests, model_name, 0, &max(0, &1 - 1))

    {:noreply, %{state | metrics: new_metrics, active_requests: active}}
  end

  # Routing Implementations

  defp route_round_robin(state) do
    index = rem(state.round_robin_index, length(state.models))
    model = Enum.at(state.models, index)
    {model, %{state | round_robin_index: index + 1}}
  end

  defp route_least_loaded(state) do
    model =
      state.models
      |> Enum.min_by(fn m -> Map.get(state.active_requests, m.name, 0) end)

    {model, state}
  end

  defp route_best_performing(state) do
    model =
      state.models
      |> Enum.max_by(fn m ->
        metrics = Map.get(state.metrics, m.name, initial_metrics())

        success_rate =
          if metrics.total_requests > 0 do
            metrics.successful_requests / metrics.total_requests
          else
            1.0
          end

        # Higher score is better: high success rate, low latency
        avg_latency =
          if metrics.total_requests > 0 do
            metrics.total_latency_ms / metrics.total_requests
          else
            0
          end

        # Normalize latency (lower is better, so invert)
        latency_score = if avg_latency > 0, do: 1000 / avg_latency, else: 1.0

        success_rate * 0.7 + latency_score * 0.3
      end)

    {model, state}
  end

  defp route_weighted(state) do
    # Weighted random selection
    total_weight = Enum.sum(Map.values(state.weights))
    random = :rand.uniform() * total_weight

    model =
      state.models
      |> Enum.reduce_while(0, fn model, acc ->
        weight = Map.get(state.weights, model.name, 0)
        new_acc = acc + weight

        if new_acc >= random do
          {:halt, model}
        else
          {:cont, new_acc}
        end
      end)

    # Handle edge case where reduce returns a number
    model = if is_number(model), do: List.last(state.models), else: model

    {model, state}
  end

  # Helpers

  defp initial_metrics do
    %{
      total_requests: 0,
      successful_requests: 0,
      total_latency_ms: 0,
      avg_latency_ms: 0.0
    }
  end

  defp emit_routing_telemetry(event, measurements, metadata) do
    :telemetry.execute(
      [:crucible, :routing, event],
      Map.put(measurements, :timestamp, System.system_time(:millisecond)),
      metadata
    )
  end
end
