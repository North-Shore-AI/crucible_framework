defmodule Crucible.Ensemble.ModelEnsemble do
  @moduledoc """
  High-level ensemble management for ML models.

  Provides a unified interface for creating and managing ensembles of ML models,
  with support for registry-based model selection, batch inference, and statistics.

  ## Examples

      # Create from registry
      {:ok, ensemble} = ModelEnsemble.from_registry(registry, %{
        sort_by: :accuracy,
        top_n: 3,
        tags: [:production]
      })

      # Run single inference
      {:ok, result} = ModelEnsemble.infer(ensemble, "What is ML?")

      # Run batch inference
      {:ok, results} = ModelEnsemble.batch_infer(ensemble, prompts,
        batch_size: 10,
        timeout: 60_000
      )

      # Get statistics
      stats = ModelEnsemble.get_stats(ensemble)
  """

  alias Crucible.Ensemble.{AdapterPool, MLVoting}
  alias Crucible.Tinkex.ModelRegistry

  require Logger

  defstruct [
    :name,
    :pool,
    :strategy,
    :execution_mode,
    :timeout,
    :hedging_config,
    :telemetry_prefix
  ]

  @type t :: %__MODULE__{
          name: String.t(),
          pool: AdapterPool.t(),
          strategy: MLVoting.strategy(),
          execution_mode: MLVoting.execution(),
          timeout: pos_integer(),
          hedging_config: map() | nil,
          telemetry_prefix: [atom()]
        }

  # Public API

  @doc """
  Creates a new model ensemble.

  ## Options
    * `:pool` - Adapter pool (required)
    * `:strategy` - Voting strategy (default: `:weighted`)
    * `:execution_mode` - Execution mode (default: `:parallel`)
    * `:timeout` - Default timeout in ms (default: 30_000)
    * `:hedging_config` - Configuration for hedged execution
    * `:telemetry_prefix` - Custom telemetry prefix
  """
  @spec create(String.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def create(name, opts) do
    pool = Keyword.fetch!(opts, :pool)

    ensemble = %__MODULE__{
      name: name,
      pool: pool,
      strategy: Keyword.get(opts, :strategy, :weighted),
      execution_mode: Keyword.get(opts, :execution_mode, :parallel),
      timeout: Keyword.get(opts, :timeout, 30_000),
      hedging_config: Keyword.get(opts, :hedging_config),
      telemetry_prefix: Keyword.get(opts, :telemetry_prefix, [:crucible, :ensemble])
    }

    emit_telemetry(ensemble, :created, %{})

    {:ok, ensemble}
  end

  @doc """
  Creates an ensemble from models in the registry.

  ## Criteria
    * `:sort_by` - Metric to sort by (e.g., :accuracy)
    * `:top_n` - Number of models to include (default: 5)
    * `:tags` - Required tags for filtering

  ## Options
    * `:strategy` - Voting strategy
    * `:execution_mode` - Execution mode
    * `:session` - Tinkex session for client creation
  """
  @spec from_registry(GenServer.server(), map(), keyword()) :: {:ok, t()} | {:error, term()}
  def from_registry(registry, criteria, opts \\ []) do
    {:ok, models} = ModelRegistry.find_for_ensemble(registry, criteria)

    if models == [] do
      {:error, :no_models_found}
    else
      # Convert registry models to adapter specs
      adapters =
        models
        |> Enum.with_index()
        |> Enum.map(fn {model, idx} ->
          weight = calculate_weight(model, idx, length(models))

          %{
            name: model.name,
            checkpoint_path: model.checkpoint_path,
            weight: weight,
            tags: model.tags
          }
        end)

      # Create adapter pool
      {:ok, pool} = AdapterPool.start_link([])

      # Add adapters to pool (without starting actual clients for testing)
      for adapter <- adapters do
        # In production, this would start actual sampling clients
        :ok = AdapterPool.add_client(pool, adapter, make_ref())
      end

      name = "ensemble-#{:erlang.unique_integer([:positive])}"

      create(name,
        pool: pool,
        strategy: Keyword.get(opts, :strategy, :weighted),
        execution_mode: Keyword.get(opts, :execution_mode, :parallel),
        timeout: Keyword.get(opts, :timeout, 30_000),
        hedging_config: Keyword.get(opts, :hedging_config)
      )
    end
  end

  @doc """
  Adds a model to the ensemble.
  """
  @spec add_model(t(), map()) :: {:ok, t()} | {:error, term()}
  def add_model(ensemble, model_spec) do
    client_ref = make_ref()
    :ok = AdapterPool.add_client(ensemble.pool, model_spec, client_ref)

    emit_telemetry(ensemble, :model_added, %{model_name: model_spec.name})

    {:ok, ensemble}
  end

  @doc """
  Removes a model from the ensemble.
  """
  @spec remove_model(t(), String.t()) :: {:ok, t()} | {:error, term()}
  def remove_model(ensemble, model_name) do
    case AdapterPool.remove_client(ensemble.pool, model_name) do
      :ok ->
        emit_telemetry(ensemble, :model_removed, %{model_name: model_name})
        {:ok, ensemble}

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Runs ensemble inference on a single prompt.

  ## Options
    * `:strategy` - Override default voting strategy
    * `:execution` - Override default execution mode
    * `:timeout` - Override default timeout
    * `:hedging` - Override hedging configuration
  """
  @spec infer(t(), String.t(), keyword()) :: {:ok, term()} | {:error, term()}
  def infer(ensemble, prompt, opts \\ []) do
    merged_opts =
      opts
      |> Keyword.put_new(:strategy, ensemble.strategy)
      |> Keyword.put_new(:execution, ensemble.execution_mode)
      |> Keyword.put_new(:timeout, ensemble.timeout)
      |> maybe_put_hedging(ensemble.hedging_config)

    start_time = System.monotonic_time(:millisecond)

    result = MLVoting.infer(ensemble.pool, prompt, merged_opts)

    duration = System.monotonic_time(:millisecond) - start_time

    emit_telemetry(ensemble, :infer_complete, %{
      duration_ms: duration,
      success: match?({:ok, _}, result)
    })

    result
  end

  @doc """
  Runs ensemble inference on multiple prompts.

  ## Options
    * `:batch_size` - Number of concurrent prompts (default: 10)
    * `:timeout` - Total timeout for batch (default: 60_000)
  """
  @spec batch_infer(t(), [String.t()], keyword()) :: {:ok, [term()]} | {:error, term()}
  def batch_infer(ensemble, prompts, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 10)
    timeout = Keyword.get(opts, :timeout, 60_000)

    start_time = System.monotonic_time(:millisecond)

    results =
      prompts
      |> Enum.chunk_every(batch_size)
      |> Enum.flat_map(fn batch ->
        tasks =
          Enum.map(batch, fn prompt ->
            Task.async(fn ->
              infer(ensemble, prompt, opts)
            end)
          end)

        Task.await_many(tasks, timeout)
      end)

    duration = System.monotonic_time(:millisecond) - start_time

    successes = Enum.count(results, &match?({:ok, _}, &1))

    emit_telemetry(ensemble, :batch_complete, %{
      total: length(prompts),
      successes: successes,
      duration_ms: duration
    })

    {:ok, results}
  end

  @doc """
  Returns statistics about the ensemble.
  """
  @spec get_stats(t()) :: map()
  def get_stats(ensemble) do
    clients = AdapterPool.all_clients(ensemble.pool)

    model_names = Enum.map(clients, fn {adapter, _} -> adapter.name end)
    weights = Enum.map(clients, fn {adapter, _} -> adapter.weight end)

    %{
      name: ensemble.name,
      model_count: length(clients),
      model_names: model_names,
      total_weight: Enum.sum(weights),
      strategy: ensemble.strategy,
      execution_mode: ensemble.execution_mode,
      timeout: ensemble.timeout
    }
  end

  # Private Functions

  defp calculate_weight(model, index, total) do
    # Weight based on position (top-ranked gets higher weight)
    # Can be customized based on metrics
    base_weight = 1.0 / total
    position_bonus = (total - index) / total * 0.5

    accuracy_bonus =
      case model.metrics[:accuracy] do
        nil -> 0
        acc -> acc * 0.3
      end

    base_weight + position_bonus + accuracy_bonus
  end

  defp maybe_put_hedging(opts, nil), do: opts
  defp maybe_put_hedging(opts, config), do: Keyword.put_new(opts, :hedging, config)

  defp emit_telemetry(ensemble, event, metadata) do
    :telemetry.execute(
      ensemble.telemetry_prefix ++ [event],
      %{timestamp: System.system_time(:millisecond)},
      Map.merge(metadata, %{ensemble_name: ensemble.name})
    )
  end
end
