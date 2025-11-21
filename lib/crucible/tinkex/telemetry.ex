defmodule Crucible.Tinkex.Telemetry do
  @moduledoc """
  Telemetry integration for Tinkex training events.

  This module provides telemetry handlers that capture Tinkex HTTP requests,
  training steps, and evaluation events for integration with Crucible's
  experiment tracking infrastructure.

  ## Events

  The following telemetry events are emitted:

  - `[:crucible, :tinkex, :training, :step]` - Training step completed
  - `[:crucible, :tinkex, :training, :epoch]` - Epoch completed
  - `[:crucible, :tinkex, :evaluation, :complete]` - Evaluation completed
  - `[:crucible, :tinkex, :checkpoint, :saved]` - Checkpoint saved

  ## Usage

      # Attach handlers at application start
      Crucible.Tinkex.Telemetry.attach()

      # Or attach with experiment context
      Crucible.Tinkex.Telemetry.attach(experiment_id: "exp-123")

      # Emit training events
      Crucible.Tinkex.Telemetry.emit_training_step(%{
        step: 100,
        loss: 0.5,
        citation_invalid_rate: 0.0
      })

  """

  require Logger

  @events [
    [:crucible, :tinkex, :training, :step],
    [:crucible, :tinkex, :training, :epoch],
    [:crucible, :tinkex, :evaluation, :complete],
    [:crucible, :tinkex, :checkpoint, :saved],
    [:crucible, :tinkex, :http, :request]
  ]

  @doc """
  Attaches telemetry handlers for Tinkex events.

  ## Options

  - `:experiment_id` - ID of the current experiment for context
  - `:handler_id` - Custom handler ID (default: "crucible-tinkex-handler")

  ## Examples

      iex> :ok = Crucible.Tinkex.Telemetry.attach()
      :ok

      iex> :ok = Crucible.Tinkex.Telemetry.attach(experiment_id: "exp-123")
      :ok
  """
  @spec attach(keyword()) :: :ok
  def attach(opts \\ []) do
    handler_id = Keyword.get(opts, :handler_id, "crucible-tinkex-handler")
    config = %{experiment_id: Keyword.get(opts, :experiment_id)}

    :telemetry.attach_many(handler_id, @events, &__MODULE__.handle_event/4, config)

    :ok
  end

  @doc """
  Detaches telemetry handlers.

  ## Examples

      iex> Crucible.Tinkex.Telemetry.detach()
      :ok
  """
  @spec detach(String.t()) :: :ok | {:error, :not_found}
  def detach(handler_id \\ "crucible-tinkex-handler") do
    :telemetry.detach(handler_id)
  end

  @doc """
  Emits a training step event.

  ## Examples

      iex> Crucible.Tinkex.Telemetry.emit_training_step(%{step: 1, loss: 1.0})
      :ok
  """
  @spec emit_training_step(map()) :: :ok
  def emit_training_step(metrics) when is_map(metrics) do
    :telemetry.execute(
      [:crucible, :tinkex, :training, :step],
      %{
        loss: Map.get(metrics, :loss),
        citation_invalid_rate: Map.get(metrics, :citation_invalid_rate, 0.0)
      },
      %{
        step: Map.get(metrics, :step),
        epoch: Map.get(metrics, :epoch),
        timestamp: DateTime.utc_now()
      }
    )

    :ok
  end

  @doc """
  Emits an epoch completion event.

  ## Examples

      iex> Crucible.Tinkex.Telemetry.emit_epoch_complete(%{epoch: 1, mean_loss: 0.8})
      :ok
  """
  @spec emit_epoch_complete(map()) :: :ok
  def emit_epoch_complete(metrics) when is_map(metrics) do
    :telemetry.execute(
      [:crucible, :tinkex, :training, :epoch],
      %{
        mean_loss: Map.get(metrics, :mean_loss),
        steps: Map.get(metrics, :steps, 0)
      },
      %{
        epoch: Map.get(metrics, :epoch),
        timestamp: DateTime.utc_now()
      }
    )

    :ok
  end

  @doc """
  Emits an evaluation completion event.

  ## Examples

      iex> Crucible.Tinkex.Telemetry.emit_evaluation_complete(%{adapter_name: "v1", metrics: %{}})
      :ok
  """
  @spec emit_evaluation_complete(map()) :: :ok
  def emit_evaluation_complete(evaluation) when is_map(evaluation) do
    :telemetry.execute(
      [:crucible, :tinkex, :evaluation, :complete],
      evaluation.metrics || %{},
      %{
        adapter_name: Map.get(evaluation, :adapter_name),
        samples: Map.get(evaluation, :samples, 0),
        timestamp: DateTime.utc_now()
      }
    )

    :ok
  end

  @doc """
  Emits a checkpoint saved event.

  ## Examples

      iex> Crucible.Tinkex.Telemetry.emit_checkpoint_saved(%{name: "step_100", step: 100})
      :ok
  """
  @spec emit_checkpoint_saved(map()) :: :ok
  def emit_checkpoint_saved(checkpoint) when is_map(checkpoint) do
    :telemetry.execute(
      [:crucible, :tinkex, :checkpoint, :saved],
      %{step: Map.get(checkpoint, :step)},
      %{
        name: Map.get(checkpoint, :name),
        timestamp: DateTime.utc_now()
      }
    )

    :ok
  end

  # Event handlers

  @doc false
  def handle_event([:crucible, :tinkex, :training, :step], measurements, metadata, config) do
    log_event(:training_step, measurements, metadata, config)
  end

  def handle_event([:crucible, :tinkex, :training, :epoch], measurements, metadata, config) do
    log_event(:epoch_complete, measurements, metadata, config)
  end

  def handle_event([:crucible, :tinkex, :evaluation, :complete], measurements, metadata, config) do
    log_event(:evaluation_complete, measurements, metadata, config)
  end

  def handle_event([:crucible, :tinkex, :checkpoint, :saved], measurements, metadata, config) do
    log_event(:checkpoint_saved, measurements, metadata, config)
  end

  def handle_event([:crucible, :tinkex, :http, :request], measurements, metadata, config) do
    log_event(:http_request, measurements, metadata, config)
  end

  defp log_event(event_type, measurements, metadata, config) do
    Logger.debug(fn ->
      experiment_id = config[:experiment_id] || "unknown"

      "[Crucible.Tinkex] #{event_type} " <>
        "experiment=#{experiment_id} " <>
        "measurements=#{inspect(measurements)} " <>
        "metadata=#{inspect(Map.drop(metadata, [:timestamp]))}"
    end)
  end

  @doc """
  Returns the list of telemetry events this module handles.

  ## Examples

      iex> events = Crucible.Tinkex.Telemetry.events()
      iex> length(events)
      5
  """
  @spec events() :: [[atom()]]
  def events, do: @events
end
