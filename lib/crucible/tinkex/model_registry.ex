defmodule Crucible.Tinkex.ModelRegistry do
  @moduledoc """
  Registry for tracking trained LoRA adapters and their metadata.

  Used for ensemble creation, model versioning, and tracking trained models
  across experiments. Supports filtering, sorting, and export/import for
  persistence.

  ## Examples

      {:ok, registry} = ModelRegistry.start_link()

      # Register a trained model
      :ok = ModelRegistry.register(registry, "scifact-v1", %{
        experiment_id: "exp-123",
        checkpoint_path: "tinker://exp-123/checkpoints/best",
        metrics: %{accuracy: 0.92, loss: 0.25},
        tags: [:production, :scifact]
      })

      # Find models for ensemble
      {:ok, models} = ModelRegistry.find_for_ensemble(registry, %{
        sort_by: :accuracy,
        top_n: 5,
        tags: [:production]
      })

      # Export registry
      :ok = ModelRegistry.export(registry, "/path/to/registry.json")
  """

  use GenServer

  require Logger

  @type model_entry :: %{
          name: String.t(),
          experiment_id: String.t(),
          checkpoint_name: String.t() | nil,
          checkpoint_path: String.t(),
          base_model: String.t() | nil,
          lora_rank: pos_integer() | nil,
          metrics: map(),
          tags: [atom()],
          created_at: DateTime.t()
        }

  defstruct models: %{}

  @required_fields [:experiment_id, :checkpoint_path]

  # Client API

  @doc """
  Starts the model registry.

  ## Options
    * `:name` - GenServer name for registration.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    server_opts = Keyword.take(opts, [:name])
    GenServer.start_link(__MODULE__, %{}, server_opts)
  end

  @doc """
  Registers a model with metadata.

  ## Required metadata fields
    * `:experiment_id` - The experiment that produced this model
    * `:checkpoint_path` - Path to the checkpoint (tinker:// URL)

  ## Optional metadata fields
    * `:checkpoint_name` - Name of the checkpoint
    * `:base_model` - Base model identifier
    * `:lora_rank` - LoRA rank used
    * `:metrics` - Training/evaluation metrics
    * `:tags` - Tags for filtering
  """
  @spec register(GenServer.server(), String.t(), map()) ::
          :ok | {:error, term()}
  def register(registry, name, metadata) do
    GenServer.call(registry, {:register, name, metadata})
  end

  @doc """
  Gets a model by name.
  """
  @spec get(GenServer.server(), String.t()) ::
          {:ok, model_entry()} | {:error, :not_found}
  def get(registry, name) do
    GenServer.call(registry, {:get, name})
  end

  @doc """
  Lists all models with optional filtering.

  ## Options
    * `:experiment_id` - Filter by experiment ID
  """
  @spec list(GenServer.server(), keyword()) :: [model_entry()]
  def list(registry, opts \\ []) do
    GenServer.call(registry, {:list, opts})
  end

  @doc """
  Finds models with all specified tags.
  """
  @spec find_by_tags(GenServer.server(), [atom()]) :: [model_entry()]
  def find_by_tags(registry, tags) do
    GenServer.call(registry, {:find_by_tags, tags})
  end

  @doc """
  Finds models suitable for ensemble based on criteria.

  ## Criteria
    * `:sort_by` - Metric to sort by (e.g., :accuracy, :loss)
    * `:top_n` - Number of models to return (default: 5)
    * `:tags` - Required tags (optional)
  """
  @spec find_for_ensemble(GenServer.server(), map()) ::
          {:ok, [model_entry()]} | {:error, term()}
  def find_for_ensemble(registry, criteria) do
    GenServer.call(registry, {:find_for_ensemble, criteria})
  end

  @doc """
  Updates metrics for a model.
  """
  @spec update_metrics(GenServer.server(), String.t(), map()) ::
          :ok | {:error, :not_found}
  def update_metrics(registry, name, metrics) do
    GenServer.call(registry, {:update_metrics, name, metrics})
  end

  @doc """
  Deletes a model from the registry.
  """
  @spec delete(GenServer.server(), String.t()) :: :ok | {:error, :not_found}
  def delete(registry, name) do
    GenServer.call(registry, {:delete, name})
  end

  @doc """
  Exports the registry to a JSON file.
  """
  @spec export(GenServer.server(), String.t()) :: :ok | {:error, term()}
  def export(registry, path) do
    GenServer.call(registry, {:export, path})
  end

  @doc """
  Imports models from a JSON file.
  """
  @spec import(GenServer.server(), String.t()) :: :ok | {:error, term()}
  def import(registry, path) do
    GenServer.call(registry, {:import, path})
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    {:ok, %__MODULE__{models: %{}}}
  end

  @impl true
  def handle_call({:register, name, metadata}, _from, state) do
    # Validate required fields
    missing = Enum.filter(@required_fields, &(!Map.has_key?(metadata, &1)))

    cond do
      length(missing) > 0 ->
        {:reply, {:error, :missing_required_fields}, state}

      Map.has_key?(state.models, name) ->
        {:reply, {:error, :already_exists}, state}

      true ->
        model = %{
          name: name,
          experiment_id: metadata[:experiment_id] || metadata["experiment_id"],
          checkpoint_name: metadata[:checkpoint_name] || metadata["checkpoint_name"],
          checkpoint_path: metadata[:checkpoint_path] || metadata["checkpoint_path"],
          base_model: metadata[:base_model] || metadata["base_model"],
          lora_rank: metadata[:lora_rank] || metadata["lora_rank"],
          metrics: metadata[:metrics] || metadata["metrics"] || %{},
          tags: normalize_tags(metadata[:tags] || metadata["tags"] || []),
          created_at: DateTime.utc_now()
        }

        emit_telemetry(:model_registered, model)

        {:reply, :ok, %{state | models: Map.put(state.models, name, model)}}
    end
  end

  @impl true
  def handle_call({:get, name}, _from, state) do
    case Map.get(state.models, name) do
      nil -> {:reply, {:error, :not_found}, state}
      model -> {:reply, {:ok, model}, state}
    end
  end

  @impl true
  def handle_call({:list, opts}, _from, state) do
    models = Map.values(state.models)

    filtered =
      case Keyword.get(opts, :experiment_id) do
        nil -> models
        exp_id -> Enum.filter(models, &(&1.experiment_id == exp_id))
      end

    {:reply, filtered, state}
  end

  @impl true
  def handle_call({:find_by_tags, tags}, _from, state) do
    models =
      state.models
      |> Map.values()
      |> Enum.filter(fn model ->
        Enum.all?(tags, &(&1 in model.tags))
      end)

    {:reply, models, state}
  end

  @impl true
  def handle_call({:find_for_ensemble, criteria}, _from, state) do
    sort_by = criteria[:sort_by] || criteria["sort_by"]
    top_n = criteria[:top_n] || criteria["top_n"] || 5
    required_tags = criteria[:tags] || criteria["tags"]

    models =
      state.models
      |> Map.values()
      |> maybe_filter_by_tags(required_tags)
      |> Enum.filter(&Map.has_key?(&1.metrics, sort_by))
      |> Enum.sort_by(& &1.metrics[sort_by], :desc)
      |> Enum.take(top_n)

    {:reply, {:ok, models}, state}
  end

  @impl true
  def handle_call({:update_metrics, name, metrics}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      model ->
        updated = %{model | metrics: Map.merge(model.metrics, metrics)}
        new_models = Map.put(state.models, name, updated)

        emit_telemetry(:model_metrics_updated, updated)

        {:reply, :ok, %{state | models: new_models}}
    end
  end

  @impl true
  def handle_call({:delete, name}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_found}, state}

      model ->
        emit_telemetry(:model_deleted, model)
        {:reply, :ok, %{state | models: Map.delete(state.models, name)}}
    end
  end

  @impl true
  def handle_call({:export, path}, _from, state) do
    data = %{
      "version" => 1,
      "exported_at" => DateTime.to_iso8601(DateTime.utc_now()),
      "models" =>
        state.models
        |> Map.values()
        |> Enum.map(&serialize_model/1)
    }

    case File.write(path, Jason.encode!(data, pretty: true)) do
      :ok -> {:reply, :ok, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:import, path}, _from, state) do
    with {:ok, content} <- File.read(path),
         {:ok, data} <- Jason.decode(content) do
      imported_models =
        data["models"]
        |> Enum.map(&deserialize_model/1)
        |> Map.new(&{&1.name, &1})

      # Merge with existing models
      new_models = Map.merge(state.models, imported_models)

      {:reply, :ok, %{state | models: new_models}}
    else
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  # Private Functions

  defp normalize_tags(tags) when is_list(tags) do
    Enum.map(tags, fn
      tag when is_atom(tag) -> tag
      tag when is_binary(tag) -> String.to_atom(tag)
    end)
  end

  defp normalize_tags(_), do: []

  defp maybe_filter_by_tags(models, nil), do: models

  defp maybe_filter_by_tags(models, tags) do
    Enum.filter(models, fn model ->
      Enum.all?(tags, &(&1 in model.tags))
    end)
  end

  defp serialize_model(model) do
    %{
      "name" => model.name,
      "experiment_id" => model.experiment_id,
      "checkpoint_name" => model.checkpoint_name,
      "checkpoint_path" => model.checkpoint_path,
      "base_model" => model.base_model,
      "lora_rank" => model.lora_rank,
      "metrics" => stringify_keys(model.metrics),
      "tags" => Enum.map(model.tags, &Atom.to_string/1),
      "created_at" => DateTime.to_iso8601(model.created_at)
    }
  end

  defp deserialize_model(data) do
    %{
      name: data["name"],
      experiment_id: data["experiment_id"],
      checkpoint_name: data["checkpoint_name"],
      checkpoint_path: data["checkpoint_path"],
      base_model: data["base_model"],
      lora_rank: data["lora_rank"],
      metrics: atomize_keys(data["metrics"] || %{}),
      tags: Enum.map(data["tags"] || [], &String.to_atom/1),
      created_at: parse_datetime(data["created_at"])
    }
  end

  defp stringify_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} ->
      key = if is_atom(k), do: Atom.to_string(k), else: k
      {key, v}
    end)
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} ->
      key = if is_binary(k), do: String.to_atom(k), else: k
      {key, v}
    end)
  end

  defp parse_datetime(nil), do: DateTime.utc_now()

  defp parse_datetime(str) when is_binary(str) do
    case DateTime.from_iso8601(str) do
      {:ok, dt, _} -> dt
      _ -> DateTime.utc_now()
    end
  end

  defp emit_telemetry(event, model) do
    :telemetry.execute(
      [:crucible, :tinkex, :registry, event],
      %{timestamp: System.system_time(:millisecond)},
      %{
        model_name: model.name,
        experiment_id: model.experiment_id
      }
    )
  end
end
