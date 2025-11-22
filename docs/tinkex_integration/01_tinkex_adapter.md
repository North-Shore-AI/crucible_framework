# Tinkex Adapter Design

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

The Tinkex adapter (`Crucible.Tinkex`) wraps the Tinkex SDK to provide experiment orchestration, checkpoint management, and telemetry integration for the Crucible Framework.

## Module Structure

```
lib/crucible/
  tinkex.ex                    # High-level API, Adapter implementation
  tinkex/
    config.ex                  # Configuration struct
    session.ex                 # Session management
    checkpoint_manager.ex      # Checkpoint storage & retrieval
    telemetry_bridge.ex        # Tinkex -> Crucible telemetry
    model_registry.ex          # Track trained models/adapters
```

## Adapter Behaviour Implementation

```elixir
defmodule Crucible.Tinkex do
  @behaviour Crucible.Lora.Adapter

  # Required callbacks
  @impl true
  def generate_id/0
  @impl true
  def create_experiment/1
  @impl true
  def batch_dataset/2
  @impl true
  def format_training_data/2
  @impl true
  def calculate_metrics/1
  @impl true
  def validate_quality/2
  @impl true
  def sampling_params/1
  @impl true
  def checkpoint_name/2

  # Extended API (not in behaviour)
  def start_session/2
  def train/3
  def evaluate/3
  def infer/3
  def save_checkpoint/2
  def load_checkpoint/2
end
```

## Session Management

### Session Lifecycle

```elixir
defmodule Crucible.Tinkex.Session do
  @moduledoc """
  Manages the lifecycle of a Tinkex training session.
  """

  use GenServer

  defstruct [
    :experiment_id,
    :service_client,
    :training_client,
    :sampling_client,
    :config,
    :checkpoints,
    :metrics_buffer,
    :started_at,
    :status
  ]

  @type status :: :initializing | :ready | :training | :evaluating | :completed | :failed

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  @impl true
  def init(opts) do
    experiment = Keyword.fetch!(opts, :experiment)
    config = experiment.config

    # Start Tinkex clients
    {:ok, service} = start_service_client(config)
    {:ok, training} = start_training_client(config, service)

    state = %__MODULE__{
      experiment_id: experiment.id,
      service_client: service,
      training_client: training,
      config: config,
      checkpoints: [],
      metrics_buffer: [],
      started_at: DateTime.utc_now(),
      status: :ready
    }

    # Emit session start telemetry
    emit_telemetry(:session_start, state)

    {:ok, state}
  end

  # Training operations
  def handle_call({:forward_backward, data, loss_fn, opts}, from, state) do
    state = %{state | status: :training}

    # Delegate to TrainingClient with telemetry wrapping
    case Tinkex.TrainingClient.forward_backward(
      state.training_client, data, loss_fn, opts
    ) do
      {:ok, task} ->
        # Monitor the task and buffer results
        spawn_link(fn ->
          result = Task.await(task, :infinity)
          GenServer.cast(self(), {:training_result, from, result})
        end)
        {:noreply, state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  def handle_call({:optim_step, adam_params, opts}, _from, state) do
    case Tinkex.TrainingClient.optim_step(
      state.training_client, adam_params, opts
    ) do
      {:ok, task} ->
        result = Task.await(task, :infinity)
        update_metrics(state, result)
        {:reply, result, state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  # Checkpoint operations
  def handle_call({:save_checkpoint, step}, _from, state) do
    checkpoint_name = Crucible.Tinkex.checkpoint_name(state.experiment_id, step)

    case Tinkex.TrainingClient.save_weights_for_sampler(
      state.training_client,
      path: "tinker://#{state.experiment_id}/checkpoints/#{checkpoint_name}"
    ) do
      {:ok, task} ->
        result = Task.await(task, :infinity)
        checkpoint = %{
          name: checkpoint_name,
          step: step,
          path: result["path"],
          created_at: DateTime.utc_now()
        }
        new_state = %{state | checkpoints: [checkpoint | state.checkpoints]}

        emit_telemetry(:checkpoint_saved, new_state, checkpoint)
        {:reply, {:ok, checkpoint}, new_state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  defp emit_telemetry(event, state, metadata \\ %{}) do
    :telemetry.execute(
      [:crucible, :tinkex, event],
      %{timestamp: System.system_time(:millisecond)},
      Map.merge(metadata, %{
        experiment_id: state.experiment_id,
        status: state.status
      })
    )
  end
end
```

## Crucible Harness Integration

### Experiment Orchestration

```elixir
defmodule Crucible.Tinkex.HarnessIntegration do
  @moduledoc """
  Integrates Tinkex training into crucible_harness experiments.
  """

  alias Crucible.Harness.Experiment

  @doc """
  Registers Tinkex training as a harness experiment step.
  """
  def register_training_step(experiment, opts) do
    Experiment.add_step(experiment, :tinkex_training, fn ctx ->
      session = ctx.session
      dataset = ctx.dataset

      # Run training loop
      results = train_loop(session, dataset, opts)

      # Return results for harness analysis
      {:ok, %{
        metrics: Crucible.Tinkex.calculate_metrics(results),
        checkpoints: session.checkpoints,
        trace_id: ctx.trace_id
      }}
    end)
  end

  defp train_loop(session, dataset, opts) do
    epochs = Keyword.get(opts, :epochs, 1)
    batch_size = Keyword.get(opts, :batch_size, 8)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 100)
    adam_params = Keyword.get(opts, :adam_params, default_adam_params())

    batches = Crucible.Tinkex.batch_dataset(dataset, batch_size)

    for epoch <- 1..epochs,
        {batch, batch_idx} <- Enum.with_index(batches) do
      step = (epoch - 1) * length(batches) + batch_idx + 1

      # Format data for Tinkex
      training_data = Crucible.Tinkex.format_training_data(batch, opts)

      # Forward-backward pass
      {:ok, task} = Tinkex.TrainingClient.forward_backward(
        session.training_client,
        training_data,
        loss_fn(opts),
        loss_fn_config: opts[:loss_fn_config]
      )
      {:ok, fb_result} = Task.await(task, :infinity)

      # Optimizer step
      {:ok, task} = Tinkex.TrainingClient.optim_step(
        session.training_client,
        adam_params
      )
      {:ok, optim_result} = Task.await(task, :infinity)

      # Checkpoint if needed
      if rem(step, checkpoint_every) == 0 do
        GenServer.call(session, {:save_checkpoint, step})
      end

      # Return step metrics
      %{
        step: step,
        epoch: epoch,
        loss: fb_result.loss,
        grad_norm: optim_result.grad_norm,
        timestamp: DateTime.utc_now()
      }
    end
  end
end
```

## Checkpoint Management

### Storage Backend

```elixir
defmodule Crucible.Tinkex.CheckpointManager do
  @moduledoc """
  Manages checkpoint storage, retrieval, and versioning.
  """

  use GenServer

  defstruct [
    :experiment_id,
    :storage_backend,  # :local | :s3 | :tinkex
    :checkpoints,
    :max_checkpoints
  ]

  @doc """
  Lists all checkpoints for an experiment.
  """
  def list_checkpoints(manager) do
    GenServer.call(manager, :list_checkpoints)
  end

  @doc """
  Gets the best checkpoint by a specific metric.
  """
  def get_best_checkpoint(manager, metric, direction \\ :min) do
    GenServer.call(manager, {:get_best, metric, direction})
  end

  @doc """
  Loads a checkpoint into a sampling client.
  """
  def load_for_sampling(manager, checkpoint_name, opts \\ []) do
    GenServer.call(manager, {:load_for_sampling, checkpoint_name, opts})
  end

  @impl true
  def handle_call({:load_for_sampling, name, opts}, _from, state) do
    checkpoint = find_checkpoint(state.checkpoints, name)

    case checkpoint do
      nil ->
        {:reply, {:error, :not_found}, state}

      %{path: path} ->
        # Create sampling client from checkpoint
        case Tinkex.TrainingClient.create_sampling_client_async(
          state.training_client,
          path,
          opts
        ) do
          task ->
            result = Task.await(task, :infinity)
            {:reply, result, state}
        end
    end
  end

  @doc """
  Prunes old checkpoints keeping only the best N.
  """
  def prune_checkpoints(manager, keep_count, metric) do
    GenServer.call(manager, {:prune, keep_count, metric})
  end
end
```

## Telemetry Bridge

### Event Translation

```elixir
defmodule Crucible.Tinkex.TelemetryBridge do
  @moduledoc """
  Translates Tinkex telemetry events to Crucible telemetry format.
  """

  def attach_handlers do
    # Attach to Tinkex events
    :telemetry.attach_many(
      "crucible-tinkex-bridge",
      [
        [:tinkex, :request, :start],
        [:tinkex, :request, :stop],
        [:tinkex, :request, :exception]
      ],
      &handle_event/4,
      nil
    )
  end

  def handle_event([:tinkex, :request, event], measurements, metadata, _config) do
    # Translate to Crucible format
    crucible_event = translate_event(event, metadata)

    :telemetry.execute(
      [:crucible, :tinkex, crucible_event],
      enrich_measurements(measurements),
      enrich_metadata(metadata)
    )
  end

  defp translate_event(:start, %{tinker_request_type: "ForwardBackward"}),
    do: :forward_backward_start
  defp translate_event(:stop, %{tinker_request_type: "ForwardBackward"}),
    do: :forward_backward_stop
  defp translate_event(:start, %{tinker_request_type: "OptimStep"}),
    do: :optim_step_start
  defp translate_event(:stop, %{tinker_request_type: "OptimStep"}),
    do: :optim_step_stop
  defp translate_event(event, _), do: event

  defp enrich_measurements(measurements) do
    Map.merge(measurements, %{
      crucible_timestamp: System.system_time(:millisecond)
    })
  end

  defp enrich_metadata(metadata) do
    Map.merge(metadata, %{
      adapter: :tinkex,
      crucible_version: CrucibleFramework.version()
    })
  end
end
```

## Model Registry

```elixir
defmodule Crucible.Tinkex.ModelRegistry do
  @moduledoc """
  Tracks trained adapters and their metadata for ensemble creation.
  """

  use GenServer

  defstruct models: %{}

  @type model_entry :: %{
    name: String.t(),
    experiment_id: String.t(),
    checkpoint_path: String.t(),
    metrics: map(),
    created_at: DateTime.t(),
    tags: [String.t()]
  }

  def register_model(registry, name, metadata) do
    GenServer.call(registry, {:register, name, metadata})
  end

  def get_model(registry, name) do
    GenServer.call(registry, {:get, name})
  end

  def list_models(registry, opts \\ []) do
    GenServer.call(registry, {:list, opts})
  end

  @doc """
  Finds models suitable for ensemble based on criteria.
  """
  def find_for_ensemble(registry, criteria) do
    GenServer.call(registry, {:find_for_ensemble, criteria})
  end

  @impl true
  def handle_call({:find_for_ensemble, criteria}, _from, state) do
    models = state.models
    |> Map.values()
    |> Enum.filter(fn model ->
      matches_criteria?(model, criteria)
    end)
    |> Enum.sort_by(& &1.metrics[criteria[:sort_by]], :desc)
    |> Enum.take(criteria[:top_n] || 5)

    {:reply, {:ok, models}, state}
  end
end
```

## Integration with crucible_telemetry

```elixir
# Event storage configuration
config :crucible_telemetry,
  storage: :ets,
  handlers: [
    # Training events
    {[:crucible, :tinkex, :forward_backward_stop], :training_step},
    {[:crucible, :tinkex, :optim_step_stop], :optimizer_step},
    {[:crucible, :tinkex, :checkpoint_saved], :checkpoint},

    # Session events
    {[:crucible, :tinkex, :session_start], :session},
    {[:crucible, :tinkex, :session_end], :session}
  ]

# Querying training data
Crucible.Telemetry.query(
  experiment_id: "abc123",
  event_type: :training_step,
  from: ~U[2025-11-21 00:00:00Z],
  to: ~U[2025-11-21 23:59:59Z]
)
```

## Usage Example

```elixir
# Complete training workflow
alias Crucible.{Lora, Tinkex, Harness}

# 1. Create experiment
{:ok, experiment} = Lora.create_experiment(
  name: "SciFact Fine-tuning v2",
  config: %{
    api_key: System.get_env("TINKEX_API_KEY"),
    base_model: "llama-3-8b",
    lora_config: %{rank: 16, alpha: 32}
  }
)

# 2. Load dataset via crucible_datasets
{:ok, dataset} = Crucible.Datasets.load(:scifact, split: :train)

# 3. Start training session
{:ok, session} = Tinkex.start_session(experiment)

# 4. Run training with harness orchestration
{:ok, results} = Harness.run(experiment, fn ctx ->
  Tinkex.HarnessIntegration.register_training_step(ctx,
    epochs: 3,
    batch_size: 8,
    checkpoint_every: 100,
    loss_fn: :cross_entropy
  )
end)

# 5. Analyze results
{:ok, analysis} = Crucible.Bench.analyze(results.metrics,
  tests: [:normality, :trend],
  visualize: true
)

# 6. Register best model
{:ok, _} = Tinkex.ModelRegistry.register_model(
  session.model_registry,
  "scifact-v2",
  %{
    checkpoint_path: results.best_checkpoint.path,
    metrics: results.final_metrics
  }
)
```

## Error Handling

```elixir
defmodule Crucible.Tinkex.Error do
  @moduledoc """
  Wraps Tinkex errors with Crucible context.
  """

  defstruct [:type, :message, :tinkex_error, :context]

  def wrap(%Tinkex.Error{} = error, context) do
    %__MODULE__{
      type: translate_type(error.type),
      message: error.message,
      tinkex_error: error,
      context: context
    }
  end

  defp translate_type(:rate_limited), do: :rate_limit
  defp translate_type(:validation), do: :validation
  defp translate_type(:request_failed), do: :network
  defp translate_type(other), do: other
end
```

## Next Steps

- **02_lora_training_interface.md**: Detailed LoRA training abstractions
- **03_ensemble_ml_integration.md**: ML-aware ensemble strategies
