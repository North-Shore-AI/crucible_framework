defmodule Crucible.Ensemble.AdapterPool do
  @moduledoc """
  Manages a pool of Tinkex sampling clients for ensemble inference.

  Provides lifecycle management for multiple sampling clients, with support
  for filtering by tags, weight-based selection, and dynamic client addition/removal.

  ## Examples

      {:ok, pool} = AdapterPool.start_link([])

      # Add clients with adapter specs
      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:prod]}
      :ok = AdapterPool.add_client(pool, adapter, client_pid)

      # Get all clients for parallel inference
      clients = AdapterPool.all_clients(pool)

      # Filter by tags
      prod_clients = AdapterPool.clients_by_tags(pool, [:prod])
  """

  use GenServer

  require Logger

  @type adapter_spec :: %{
          name: String.t(),
          checkpoint_path: String.t(),
          weight: float(),
          tags: [atom()]
        }

  @type t :: pid()

  defstruct adapters: %{}, clients: %{}

  # Client API

  @doc """
  Starts the adapter pool GenServer.

  ## Options
    * `:name` - GenServer name for registration
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    server_opts = Keyword.take(opts, [:name])
    GenServer.start_link(__MODULE__, %{}, server_opts)
  end

  @doc """
  Creates an adapter pool from adapter specs and a training session.

  Automatically starts sampling clients for each adapter.

  ## Options
    * `:adapters` - List of adapter specs (required)
    * `:session` - Tinkex training session (required)
    * `:config` - Additional pool configuration
  """
  @spec create(keyword()) :: {:ok, t()} | {:error, term()}
  def create(opts) do
    adapters = Keyword.fetch!(opts, :adapters)
    session = Keyword.fetch!(opts, :session)

    {:ok, pool} = start_link(Keyword.get(opts, :name, []))

    # Start sampling clients for each adapter
    results =
      Enum.map(adapters, fn adapter ->
        case start_sampling_client(session, adapter) do
          {:ok, client} ->
            :ok = add_client(pool, adapter, client)
            :ok

          {:error, reason} ->
            Logger.error("Failed to start client for #{adapter.name}: #{inspect(reason)}")
            {:error, adapter.name, reason}
        end
      end)

    errors = Enum.filter(results, &match?({:error, _, _}, &1))

    if errors == [] do
      emit_telemetry(:pool_created, %{adapter_count: length(adapters)})
      {:ok, pool}
    else
      {:error, {:client_creation_failed, errors}}
    end
  end

  @doc """
  Gets a specific sampling client by adapter name.
  """
  @spec get_client(t(), String.t()) :: {:ok, term()} | {:error, :not_found}
  def get_client(pool, adapter_name) do
    GenServer.call(pool, {:get_client, adapter_name})
  end

  @doc """
  Gets all clients with their adapter specs.

  Returns a list of `{adapter_spec, client}` tuples.
  """
  @spec all_clients(t()) :: [{adapter_spec(), term()}]
  def all_clients(pool) do
    GenServer.call(pool, :all_clients)
  end

  @doc """
  Gets clients matching all specified tags.
  """
  @spec clients_by_tags(t(), [atom()]) :: [{adapter_spec(), term()}]
  def clients_by_tags(pool, tags) do
    GenServer.call(pool, {:clients_by_tags, tags})
  end

  @doc """
  Adds a client to the pool.
  """
  @spec add_client(t(), adapter_spec(), term()) :: :ok
  def add_client(pool, adapter_spec, client) do
    GenServer.call(pool, {:add_client, adapter_spec, client})
  end

  @doc """
  Removes a client from the pool.
  """
  @spec remove_client(t(), String.t()) :: :ok | {:error, :not_found}
  def remove_client(pool, adapter_name) do
    GenServer.call(pool, {:remove_client, adapter_name})
  end

  @doc """
  Returns the count of clients in the pool.
  """
  @spec count(t()) :: non_neg_integer()
  def count(pool) do
    GenServer.call(pool, :count)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    {:ok, %__MODULE__{adapters: %{}, clients: %{}}}
  end

  @impl true
  def handle_call({:get_client, name}, _from, state) do
    case Map.get(state.clients, name) do
      nil -> {:reply, {:error, :not_found}, state}
      client -> {:reply, {:ok, client}, state}
    end
  end

  @impl true
  def handle_call(:all_clients, _from, state) do
    clients =
      state.adapters
      |> Enum.map(fn {name, adapter} ->
        {adapter, Map.get(state.clients, name)}
      end)

    {:reply, clients, state}
  end

  @impl true
  def handle_call({:clients_by_tags, tags}, _from, state) do
    clients =
      state.adapters
      |> Enum.filter(fn {_name, adapter} ->
        Enum.all?(tags, &(&1 in adapter.tags))
      end)
      |> Enum.map(fn {name, adapter} ->
        {adapter, Map.get(state.clients, name)}
      end)

    {:reply, clients, state}
  end

  @impl true
  def handle_call({:add_client, adapter_spec, client}, _from, state) do
    name = adapter_spec.name

    new_state = %{
      state
      | adapters: Map.put(state.adapters, name, adapter_spec),
        clients: Map.put(state.clients, name, client)
    }

    emit_telemetry(:client_added, %{adapter_name: name})

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:remove_client, name}, _from, state) do
    case Map.get(state.clients, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      _client ->
        new_state = %{
          state
          | adapters: Map.delete(state.adapters, name),
            clients: Map.delete(state.clients, name)
        }

        emit_telemetry(:client_removed, %{adapter_name: name})

        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call(:count, _from, state) do
    {:reply, map_size(state.clients), state}
  end

  # Private Functions

  defp start_sampling_client(session, adapter) do
    # Start sampling client from training session
    # This integrates with Tinkex.TrainingClient
    try do
      case Code.ensure_loaded(Tinkex.TrainingClient) do
        {:module, _} ->
          if function_exported?(Tinkex.TrainingClient, :create_sampling_client_async, 2) do
            task =
              apply(Tinkex.TrainingClient, :create_sampling_client_async, [
                session.training_client,
                adapter.checkpoint_path
              ])

            Task.await(task, :infinity)
          else
            # Fallback: return a mock client for development
            {:ok, {:mock_client, adapter.name}}
          end

        {:error, _} ->
          # Tinkex not available, return mock client
          {:ok, {:mock_client, adapter.name}}
      end
    rescue
      e ->
        {:error, Exception.message(e)}
    end
  end

  defp emit_telemetry(event, metadata) do
    :telemetry.execute(
      [:crucible, :ensemble, :adapter_pool, event],
      %{timestamp: System.system_time(:millisecond)},
      metadata
    )
  end
end
