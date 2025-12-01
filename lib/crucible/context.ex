defmodule Crucible.Context do
  @moduledoc """
  Runtime context threaded through experiment stages.

  Stages mutate the context to add data, telemetry, metrics, or artifacts.

  ## Helper Functions (v0.4.0+)

  This module provides ergonomic helper functions for common context operations:

  ### Metrics Management
  - `put_metric/3` - Add or update a metric
  - `get_metric/3` - Get a metric value
  - `update_metric/3` - Update metric using function
  - `merge_metrics/2` - Merge multiple metrics
  - `has_metric?/2` - Check if metric exists

  ### Output Management
  - `add_output/2` - Add single output
  - `add_outputs/2` - Add multiple outputs

  ### Artifact Management
  - `put_artifact/3` - Store an artifact
  - `get_artifact/3` - Retrieve an artifact
  - `has_artifact?/2` - Check if artifact exists

  ### Assigns Management (Phoenix-style)
  - `assign/2` - Assign single or multiple values
  - `assign/3` - Assign single key-value pair

  ### Query Functions
  - `has_data?/1` - Check if dataset loaded
  - `has_backend_session?/2` - Check backend session exists
  - `get_backend_session/2` - Get backend session

  ### Stage Tracking
  - `mark_stage_complete/2` - Mark stage as completed
  - `stage_completed?/2` - Check if stage completed
  - `completed_stages/1` - List all completed stages

  ## Examples

      # Create a context with required fields
      ctx = %Crucible.Context{
        experiment_id: "exp1",
        run_id: "run1",
        experiment: %CrucibleIR.Experiment{id: "exp1", backend: %CrucibleIR.BackendRef{id: :mock}}
      }

      # Add metrics
      ctx = Crucible.Context.put_metric(ctx, :accuracy, 0.95)
      Crucible.Context.get_metric(ctx, :accuracy)
      # => 0.95

      # Phoenix-style assigns
      ctx = Crucible.Context.assign(ctx, user: "alice", priority: :high)
      ctx.assigns.user
      # => "alice"

      # Track stage completion
      ctx = Crucible.Context.mark_stage_complete(ctx, :data_load)
      Crucible.Context.stage_completed?(ctx, :data_load)
      # => true
  """

  alias CrucibleIR.Experiment

  @type t :: %__MODULE__{
          experiment_id: String.t(),
          run_id: String.t(),
          experiment: Experiment.t(),
          dataset: term() | nil,
          batches: Enumerable.t() | nil,
          examples: list() | nil,
          backend_sessions: %{atom() => term()},
          backend_state: map(),
          outputs: list(),
          metrics: map(),
          artifacts: map(),
          trace: term() | nil,
          telemetry_context: map(),
          assigns: map()
        }

  @enforce_keys [:experiment_id, :run_id, :experiment]
  defstruct [
    :experiment_id,
    :run_id,
    :experiment,
    dataset: nil,
    batches: nil,
    examples: nil,
    backend_sessions: %{},
    backend_state: %{},
    outputs: [],
    metrics: %{},
    artifacts: %{},
    trace: nil,
    telemetry_context: %{},
    assigns: %{}
  ]

  # ============================================================================
  # Metrics Management
  # ============================================================================

  @doc """
  Puts a metric into the context.

  ## Examples

      ctx = put_metric(ctx, :accuracy, 0.95)
      ctx.metrics.accuracy
      # => 0.95
  """
  @spec put_metric(t(), atom(), term()) :: t()
  def put_metric(%__MODULE__{} = ctx, key, value) when is_atom(key) do
    %__MODULE__{ctx | metrics: Map.put(ctx.metrics, key, value)}
  end

  @doc """
  Gets a metric from the context, returning a default if not found.

  ## Examples

      get_metric(ctx, :accuracy)
      # => 0.95

      get_metric(ctx, :missing, :default)
      # => :default
  """
  @spec get_metric(t(), atom(), term()) :: term()
  def get_metric(%__MODULE__{} = ctx, key, default \\ nil) when is_atom(key) do
    Map.get(ctx.metrics, key, default)
  end

  @doc """
  Updates a metric using a function.

  ## Examples

      ctx = put_metric(ctx, :count, 1)
      ctx = update_metric(ctx, :count, &(&1 + 1))
      get_metric(ctx, :count)
      # => 2
  """
  @spec update_metric(t(), atom(), (term() -> term())) :: t()
  def update_metric(%__MODULE__{} = ctx, key, update_fn)
      when is_atom(key) and is_function(update_fn, 1) do
    %__MODULE__{ctx | metrics: Map.update(ctx.metrics, key, nil, update_fn)}
  end

  @doc """
  Merges multiple metrics into the context.

  ## Examples

      ctx = merge_metrics(ctx, %{accuracy: 0.95, loss: 0.05})
      get_metric(ctx, :accuracy)
      # => 0.95
  """
  @spec merge_metrics(t(), map()) :: t()
  def merge_metrics(%__MODULE__{} = ctx, metrics) when is_map(metrics) do
    %__MODULE__{ctx | metrics: Map.merge(ctx.metrics, metrics)}
  end

  @doc """
  Checks if a metric exists in the context.

  ## Examples

      has_metric?(ctx, :accuracy)
      # => true

      has_metric?(ctx, :missing)
      # => false
  """
  @spec has_metric?(t(), atom()) :: boolean()
  def has_metric?(%__MODULE__{} = ctx, key) when is_atom(key) do
    Map.has_key?(ctx.metrics, key)
  end

  # ============================================================================
  # Output Management
  # ============================================================================

  @doc """
  Adds a single output to the context.

  ## Examples

      ctx = add_output(ctx, %{result: "success"})
      length(ctx.outputs)
      # => 1
  """
  @spec add_output(t(), term()) :: t()
  def add_output(%__MODULE__{} = ctx, output) do
    %__MODULE__{ctx | outputs: ctx.outputs ++ [output]}
  end

  @doc """
  Adds multiple outputs to the context.

  ## Examples

      ctx = add_outputs(ctx, [%{result: "a"}, %{result: "b"}])
      length(ctx.outputs)
      # => 2
  """
  @spec add_outputs(t(), list()) :: t()
  def add_outputs(%__MODULE__{} = ctx, outputs) when is_list(outputs) do
    %__MODULE__{ctx | outputs: ctx.outputs ++ outputs}
  end

  # ============================================================================
  # Artifact Management
  # ============================================================================

  @doc """
  Stores an artifact in the context.

  ## Examples

      ctx = put_artifact(ctx, :report, "report.html")
      get_artifact(ctx, :report)
      # => "report.html"
  """
  @spec put_artifact(t(), atom(), term()) :: t()
  def put_artifact(%__MODULE__{} = ctx, key, artifact) when is_atom(key) do
    %__MODULE__{ctx | artifacts: Map.put(ctx.artifacts, key, artifact)}
  end

  @doc """
  Retrieves an artifact from the context.

  ## Examples

      get_artifact(ctx, :report)
      # => "report.html"

      get_artifact(ctx, :missing, :not_found)
      # => :not_found
  """
  @spec get_artifact(t(), atom(), term()) :: term()
  def get_artifact(%__MODULE__{} = ctx, key, default \\ nil) when is_atom(key) do
    Map.get(ctx.artifacts, key, default)
  end

  @doc """
  Checks if an artifact exists in the context.

  ## Examples

      has_artifact?(ctx, :report)
      # => true
  """
  @spec has_artifact?(t(), atom()) :: boolean()
  def has_artifact?(%__MODULE__{} = ctx, key) when is_atom(key) do
    Map.has_key?(ctx.artifacts, key)
  end

  # ============================================================================
  # Assigns Management (Phoenix-style)
  # ============================================================================

  @doc """
  Assigns a single key-value pair to the context assigns.

  ## Examples

      ctx = assign(ctx, :user, "alice")
      ctx.assigns.user
      # => "alice"
  """
  @spec assign(t(), atom(), term()) :: t()
  def assign(%__MODULE__{} = ctx, key, value) when is_atom(key) do
    %__MODULE__{ctx | assigns: Map.put(ctx.assigns, key, value)}
  end

  @doc """
  Assigns multiple values to the context assigns.

  ## Examples

      ctx = assign(ctx, user: "alice", priority: :high)
      ctx.assigns.user
      # => "alice"
  """
  @spec assign(t(), keyword() | map()) :: t()
  def assign(%__MODULE__{} = ctx, assigns) when is_list(assigns) or is_map(assigns) do
    assigns_map = if is_list(assigns), do: Map.new(assigns), else: assigns
    %__MODULE__{ctx | assigns: Map.merge(ctx.assigns, assigns_map)}
  end

  # ============================================================================
  # Query Functions
  # ============================================================================

  @doc """
  Checks if the context has loaded data (dataset and examples).

  ## Examples

      has_data?(ctx)
      # => true
  """
  @spec has_data?(t()) :: boolean()
  def has_data?(%__MODULE__{dataset: dataset, examples: examples}) do
    not is_nil(dataset) and not is_nil(examples) and examples != []
  end

  @doc """
  Checks if a backend session exists for the given backend ID.

  ## Examples

      has_backend_session?(ctx, :tinkex)
      # => true
  """
  @spec has_backend_session?(t(), atom()) :: boolean()
  def has_backend_session?(%__MODULE__{} = ctx, backend_id) when is_atom(backend_id) do
    Enum.any?(ctx.backend_sessions, fn {{bid, _exp_id}, _session} -> bid == backend_id end)
  end

  @doc """
  Gets a backend session for the given backend ID.

  Returns the first matching session if multiple exist.

  ## Examples

      get_backend_session(ctx, :tinkex)
      # => #PID<0.123.0>
  """
  @spec get_backend_session(t(), atom()) :: term() | nil
  def get_backend_session(%__MODULE__{} = ctx, backend_id) when is_atom(backend_id) do
    ctx.backend_sessions
    |> Enum.find_value(fn
      {{^backend_id, _exp_id}, session} -> session
      _ -> nil
    end)
  end

  # ============================================================================
  # Stage Tracking
  # ============================================================================

  @doc """
  Marks a stage as completed in the context.

  ## Examples

      ctx = mark_stage_complete(ctx, :data_load)
      stage_completed?(ctx, :data_load)
      # => true
  """
  @spec mark_stage_complete(t(), atom()) :: t()
  def mark_stage_complete(%__MODULE__{} = ctx, stage_name) when is_atom(stage_name) do
    completed = Map.get(ctx.assigns, :completed_stages, [])
    updated_completed = if stage_name in completed, do: completed, else: completed ++ [stage_name]
    assign(ctx, :completed_stages, updated_completed)
  end

  @doc """
  Checks if a stage has been completed.

  ## Examples

      stage_completed?(ctx, :data_load)
      # => true

      stage_completed?(ctx, :not_run)
      # => false
  """
  @spec stage_completed?(t(), atom()) :: boolean()
  def stage_completed?(%__MODULE__{} = ctx, stage_name) when is_atom(stage_name) do
    completed = Map.get(ctx.assigns, :completed_stages, [])
    stage_name in completed
  end

  @doc """
  Returns a list of all completed stages.

  ## Examples

      completed_stages(ctx)
      # => [:data_load, :backend_call, :bench]
  """
  @spec completed_stages(t()) :: [atom()]
  def completed_stages(%__MODULE__{} = ctx) do
    Map.get(ctx.assigns, :completed_stages, [])
  end
end
