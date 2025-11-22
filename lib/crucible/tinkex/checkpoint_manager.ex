defmodule Crucible.Tinkex.CheckpointManager do
  @moduledoc """
  Manages checkpoint storage, retrieval, versioning, and pruning.

  Provides a GenServer-based checkpoint management system that tracks
  training checkpoints, supports retrieval by name or metrics, and
  handles automatic pruning based on configurable limits.

  ## Examples

      {:ok, manager} = CheckpointManager.start_link(
        experiment_id: "exp-123",
        storage_dir: "/tmp/checkpoints",
        max_checkpoints: 10
      )

      # Save checkpoint
      {:ok, checkpoint} = CheckpointManager.save(manager, 1000, %{loss: 0.25})

      # Get best checkpoint
      {:ok, best} = CheckpointManager.get_best(manager, :loss, :min)

      # Prune to keep only top 5
      {:ok, pruned} = CheckpointManager.prune(manager, 5, :loss)
  """

  use GenServer

  require Logger

  @type checkpoint :: %{
          name: String.t(),
          experiment_id: String.t(),
          step: pos_integer(),
          path: String.t(),
          local_path: String.t() | nil,
          metrics: map(),
          created_at: DateTime.t()
        }

  defstruct [
    :experiment_id,
    :storage_dir,
    :max_checkpoints,
    checkpoints: []
  ]

  # Client API

  @doc """
  Starts the checkpoint manager.

  ## Options
    * `:experiment_id` - Required. The experiment identifier.
    * `:storage_dir` - Required. Directory for local checkpoint storage.
    * `:max_checkpoints` - Maximum checkpoints to keep (default: 10).
    * `:name` - GenServer name for registration.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    experiment_id = Keyword.fetch!(opts, :experiment_id)
    storage_dir = Keyword.fetch!(opts, :storage_dir)
    max_checkpoints = Keyword.get(opts, :max_checkpoints, 10)

    server_opts = Keyword.take(opts, [:name])

    GenServer.start_link(
      __MODULE__,
      %{
        experiment_id: experiment_id,
        storage_dir: storage_dir,
        max_checkpoints: max_checkpoints
      },
      server_opts
    )
  end

  @doc """
  Saves a checkpoint with metrics.

  Returns the checkpoint metadata including generated name and path.
  """
  @spec save(GenServer.server(), pos_integer(), map(), keyword()) ::
          {:ok, checkpoint()} | {:error, term()}
  def save(manager, step, metrics, opts \\ []) do
    GenServer.call(manager, {:save, step, metrics, opts})
  end

  @doc """
  Lists all checkpoints sorted by step (descending).
  """
  @spec list(GenServer.server()) :: [checkpoint()]
  def list(manager) do
    GenServer.call(manager, :list)
  end

  @doc """
  Gets a checkpoint by name.
  """
  @spec get(GenServer.server(), String.t()) :: {:ok, checkpoint()} | {:error, :not_found}
  def get(manager, name) do
    GenServer.call(manager, {:get, name})
  end

  @doc """
  Gets the best checkpoint by a metric.

  ## Options
    * `direction` - `:min` for lowest value, `:max` for highest (default: `:min`)
  """
  @spec get_best(GenServer.server(), atom(), :min | :max) ::
          {:ok, checkpoint()} | {:error, term()}
  def get_best(manager, metric, direction \\ :min) do
    GenServer.call(manager, {:get_best, metric, direction})
  end

  @doc """
  Loads a checkpoint for use with a sampling client.
  """
  @spec load_for_sampling(GenServer.server(), String.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def load_for_sampling(manager, name, opts \\ []) do
    GenServer.call(manager, {:load_for_sampling, name, opts})
  end

  @doc """
  Downloads a checkpoint to local storage.
  """
  @spec download(GenServer.server(), String.t(), String.t()) ::
          {:ok, String.t()} | {:error, term()}
  def download(manager, name, local_path) do
    GenServer.call(manager, {:download, name, local_path}, :infinity)
  end

  @doc """
  Prunes checkpoints keeping only the top N by metric.

  Returns the number of checkpoints removed.
  """
  @spec prune(GenServer.server(), pos_integer(), atom()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def prune(manager, keep_count, metric) do
    GenServer.call(manager, {:prune, keep_count, metric})
  end

  @doc """
  Deletes a checkpoint by name.
  """
  @spec delete(GenServer.server(), String.t()) :: :ok | {:error, :not_found}
  def delete(manager, name) do
    GenServer.call(manager, {:delete, name})
  end

  # Server Callbacks

  @impl true
  def init(%{
        experiment_id: experiment_id,
        storage_dir: storage_dir,
        max_checkpoints: max_checkpoints
      }) do
    # Ensure storage directory exists
    File.mkdir_p!(storage_dir)

    state = %__MODULE__{
      experiment_id: experiment_id,
      storage_dir: storage_dir,
      max_checkpoints: max_checkpoints,
      checkpoints: []
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:save, step, _metrics, _opts}, _from, state) when step < 1 do
    {:reply, {:error, :invalid_step}, state}
  end

  @impl true
  def handle_call({:save, step, metrics, _opts}, _from, state) do
    timestamp = System.system_time(:microsecond)
    name = "#{state.experiment_id}_step_#{step}_#{timestamp}"
    path = "tinker://#{state.experiment_id}/checkpoints/#{name}"

    checkpoint = %{
      name: name,
      experiment_id: state.experiment_id,
      step: step,
      path: path,
      local_path: nil,
      metrics: metrics,
      created_at: DateTime.utc_now()
    }

    # Emit telemetry
    emit_telemetry(:checkpoint_created, state, checkpoint)

    # Add checkpoint and enforce max limit
    new_checkpoints = [checkpoint | state.checkpoints]

    new_checkpoints =
      if length(new_checkpoints) > state.max_checkpoints do
        # Remove oldest by created_at
        new_checkpoints
        |> Enum.sort_by(& &1.created_at, {:desc, DateTime})
        |> Enum.take(state.max_checkpoints)
      else
        new_checkpoints
      end

    new_state = %{state | checkpoints: new_checkpoints}

    {:reply, {:ok, checkpoint}, new_state}
  end

  @impl true
  def handle_call(:list, _from, state) do
    sorted = Enum.sort_by(state.checkpoints, & &1.step, :desc)
    {:reply, sorted, state}
  end

  @impl true
  def handle_call({:get, name}, _from, state) do
    case find_checkpoint(state.checkpoints, name) do
      nil -> {:reply, {:error, :not_found}, state}
      checkpoint -> {:reply, {:ok, checkpoint}, state}
    end
  end

  @impl true
  def handle_call({:get_best, metric, direction}, _from, state) do
    case state.checkpoints do
      [] ->
        {:reply, {:error, :no_checkpoints}, state}

      checkpoints ->
        # Filter checkpoints that have the metric
        with_metric =
          Enum.filter(checkpoints, fn cp ->
            Map.has_key?(cp.metrics, metric)
          end)

        case with_metric do
          [] ->
            {:reply, {:error, :metric_not_found}, state}

          filtered ->
            best =
              case direction do
                :min -> Enum.min_by(filtered, & &1.metrics[metric])
                :max -> Enum.max_by(filtered, & &1.metrics[metric])
              end

            {:reply, {:ok, best}, state}
        end
    end
  end

  @impl true
  def handle_call({:load_for_sampling, name, _opts}, _from, state) do
    case find_checkpoint(state.checkpoints, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      checkpoint ->
        if checkpoint.local_path && File.exists?(checkpoint.local_path) do
          {:reply, {:ok, %{path: checkpoint.local_path, checkpoint: checkpoint}}, state}
        else
          {:reply, {:error, :not_downloaded}, state}
        end
    end
  end

  @impl true
  def handle_call({:download, name, local_path}, _from, state) do
    case find_checkpoint(state.checkpoints, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      checkpoint ->
        # Update checkpoint with local path
        updated =
          Enum.map(state.checkpoints, fn cp ->
            if cp.name == name do
              %{cp | local_path: local_path}
            else
              cp
            end
          end)

        emit_telemetry(:checkpoint_downloaded, state, checkpoint)

        {:reply, {:ok, local_path}, %{state | checkpoints: updated}}
    end
  end

  @impl true
  def handle_call({:prune, keep_count, metric}, _from, state) do
    # Sort by metric (ascending - keep lowest)
    with_metric =
      state.checkpoints
      |> Enum.filter(&Map.has_key?(&1.metrics, metric))
      |> Enum.sort_by(& &1.metrics[metric])

    without_metric =
      Enum.reject(state.checkpoints, &Map.has_key?(&1.metrics, metric))

    to_keep = Enum.take(with_metric, keep_count)
    to_remove = Enum.drop(with_metric, keep_count)

    # Clean up local files for removed checkpoints
    for checkpoint <- to_remove do
      if checkpoint.local_path && File.exists?(checkpoint.local_path) do
        File.rm(checkpoint.local_path)
      end

      emit_telemetry(:checkpoint_pruned, state, checkpoint)
    end

    new_checkpoints = to_keep ++ without_metric
    removed_count = length(to_remove)

    {:reply, {:ok, removed_count}, %{state | checkpoints: new_checkpoints}}
  end

  @impl true
  def handle_call({:delete, name}, _from, state) do
    case find_checkpoint(state.checkpoints, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      checkpoint ->
        # Remove local file if exists
        if checkpoint.local_path && File.exists?(checkpoint.local_path) do
          File.rm(checkpoint.local_path)
        end

        new_checkpoints = Enum.reject(state.checkpoints, &(&1.name == name))

        emit_telemetry(:checkpoint_deleted, state, checkpoint)

        {:reply, :ok, %{state | checkpoints: new_checkpoints}}
    end
  end

  # Private Functions

  defp find_checkpoint(checkpoints, name) do
    Enum.find(checkpoints, &(&1.name == name))
  end

  defp emit_telemetry(event, state, checkpoint) do
    :telemetry.execute(
      [:crucible, :tinkex, :checkpoint, event],
      %{timestamp: System.system_time(:millisecond)},
      %{
        experiment_id: state.experiment_id,
        checkpoint_name: checkpoint.name,
        step: checkpoint.step
      }
    )
  end
end
