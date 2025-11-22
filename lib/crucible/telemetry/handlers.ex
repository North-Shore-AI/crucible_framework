defmodule Crucible.Telemetry.Handlers do
  @moduledoc """
  Sets up telemetry handlers for all Crucible events.

  This module attaches handlers to the standard telemetry events emitted by
  Crucible components and routes them to the appropriate collectors.

  ## Events Handled

  ### Training
  - `[:crucible, :training, :start]`
  - `[:crucible, :training, :stop]`
  - `[:crucible, :training, :exception]`
  - `[:crucible, :training, :step]`
  - `[:crucible, :training, :epoch]`
  - `[:crucible, :training, :checkpoint]`

  ### Inference
  - `[:crucible, :inference, :start]`
  - `[:crucible, :inference, :stop]`

  ### Ensemble
  - `[:crucible, :ensemble, :infer]`
  - `[:crucible, :ensemble, :vote]`

  ### Harness
  - `[:crucible, :harness, :stage_start]`
  - `[:crucible, :harness, :stage_stop]`
  - `[:crucible, :harness, :stage_error]`

  ### Hedging
  - `[:crucible, :hedging, :dispatch]`
  - `[:crucible, :hedging, :winner]`

  ## Usage

      # Attach all handlers at application startup
      Crucible.Telemetry.Handlers.attach_handlers()

      # With custom configuration
      Crucible.Telemetry.Handlers.attach_handlers(%{
        callback: fn event, measurements, metadata ->
          Logger.info("Event: \#{inspect(event)}")
        end
      })

      # Detach handlers
      Crucible.Telemetry.Handlers.detach_handlers()
  """

  require Logger

  @handler_prefix "crucible-telemetry-"

  @doc """
  Attaches telemetry handlers for all Crucible events.

  ## Options

  - `:callback` - Custom callback function for all events
  - `:collector` - PID of the collector to route to
  - `:filters` - List of event categories to handle (e.g., [:training, :inference])
  """
  @spec attach_handlers(map()) :: :ok
  def attach_handlers(config \\ %{}) do
    events =
      all_events()
      |> maybe_filter_events(config[:filters])

    # Group events by category for separate handlers
    events
    |> Enum.group_by(&get_category/1)
    |> Enum.each(fn {category, category_events} ->
      handler_id = "#{@handler_prefix}#{category}"

      :telemetry.attach_many(
        handler_id,
        category_events,
        &dispatch_event/4,
        config
      )
    end)

    :ok
  end

  @doc """
  Detaches all Crucible telemetry handlers.
  """
  @spec detach_handlers() :: :ok
  def detach_handlers do
    # Detach all handler categories
    [:training, :inference, :ensemble, :harness, :hedging, :tinkex]
    |> Enum.each(fn category ->
      handler_id = "#{@handler_prefix}#{category}"
      :telemetry.detach(handler_id)
    end)

    :ok
  end

  @doc """
  Returns the complete list of telemetry events handled.
  """
  @spec all_events() :: [[atom()]]
  def all_events do
    training_events() ++
      inference_events() ++
      ensemble_events() ++
      harness_events() ++
      hedging_events() ++
      tinkex_events()
  end

  # Event dispatch

  defp dispatch_event(event, measurements, metadata, config) do
    # Custom callback takes precedence
    if callback = config[:callback] do
      callback.(event, measurements, metadata)
    end

    # Route to appropriate handler based on event category
    case get_category(event) do
      :training -> handle_training_event(event, measurements, metadata, config)
      :inference -> handle_inference_event(event, measurements, metadata, config)
      :ensemble -> handle_ensemble_event(event, measurements, metadata, config)
      :harness -> handle_harness_event(event, measurements, metadata, config)
      :hedging -> handle_hedging_event(event, measurements, metadata, config)
      :tinkex -> handle_tinkex_event(event, measurements, metadata, config)
      _ -> :ok
    end
  end

  @doc """
  Handles training-related telemetry events.
  """
  @spec handle_training_event(list(), map(), map(), map()) :: :ok
  def handle_training_event(event, measurements, metadata, _config) do
    case event do
      [:crucible, :training, :step] ->
        # Record to collector if available
        if collector = get_collector(metadata) do
          Crucible.Telemetry.MLMetrics.record_training_step(
            collector,
            metadata[:step] || 0,
            metadata[:epoch] || 1,
            measurements[:loss] || 0.0,
            measurements[:grad_norm] || 0.0
          )
        end

      [:crucible, :training, :checkpoint] ->
        if collector = get_collector(metadata) do
          Crucible.Telemetry.MLMetrics.record_checkpoint(
            collector,
            metadata[:step] || 0,
            Map.take(measurements, [:loss, :accuracy, :val_loss])
          )
        end

      [:crucible, :training, :start] ->
        Logger.debug("Training started: #{inspect(metadata[:experiment_id])}")

      [:crucible, :training, :stop] ->
        Logger.debug("Training stopped: #{inspect(metadata[:experiment_id])}")

      [:crucible, :training, :exception] ->
        Logger.error("Training exception: #{inspect(metadata)}")

      [:crucible, :training, :epoch] ->
        Logger.debug("Epoch #{metadata[:epoch]} completed")

      _ ->
        :ok
    end

    :ok
  end

  @doc """
  Handles inference-related telemetry events.
  """
  @spec handle_inference_event(list(), map(), map(), map()) :: :ok
  def handle_inference_event(event, measurements, metadata, _config) do
    case event do
      [:crucible, :inference, :stop] ->
        if collector = get_collector(metadata) do
          latency = measurements[:duration] |> convert_duration()
          tokens = measurements[:tokens] || 0
          model = metadata[:model] || "unknown"

          Crucible.Telemetry.MLMetrics.record_inference(
            collector,
            model,
            latency,
            tokens
          )
        end

      [:crucible, :inference, :start] ->
        :ok

      _ ->
        :ok
    end

    :ok
  end

  @doc """
  Handles ensemble-related telemetry events.
  """
  @spec handle_ensemble_event(list(), map(), map(), map()) :: :ok
  def handle_ensemble_event(event, measurements, metadata, _config) do
    case event do
      [:crucible, :ensemble, :vote] ->
        if collector = get_collector(metadata) do
          Crucible.Telemetry.MLMetrics.record(
            collector,
            :ensemble,
            %{
              latency: convert_duration(measurements[:duration]),
              model_count: measurements[:model_count]
            },
            %{
              strategy: metadata[:strategy],
              experiment_id: metadata[:experiment_id]
            }
          )
        end

      [:crucible, :ensemble, :infer] ->
        if collector = get_collector(metadata) do
          Crucible.Telemetry.MLMetrics.record(
            collector,
            :ensemble,
            measurements,
            metadata
          )
        end

      _ ->
        :ok
    end

    :ok
  end

  @doc """
  Handles harness-related telemetry events.
  """
  @spec handle_harness_event(list(), map(), map(), map()) :: :ok
  def handle_harness_event(event, measurements, metadata, _config) do
    case event do
      [:crucible, :harness, :stage_start] ->
        if tracker = get_tracker(metadata) do
          Crucible.Telemetry.ExperimentTracker.start_stage(
            tracker,
            metadata[:experiment_id],
            metadata[:stage]
          )
        end

      [:crucible, :harness, :stage_stop] ->
        if tracker = get_tracker(metadata) do
          Crucible.Telemetry.ExperimentTracker.end_stage(
            tracker,
            metadata[:experiment_id],
            metadata[:stage],
            Map.take(measurements, [:duration, :result])
          )
        end

      [:crucible, :harness, :stage_error] ->
        Logger.error("Stage error: #{inspect(metadata[:stage])} - #{inspect(metadata[:error])}")

      _ ->
        :ok
    end

    :ok
  end

  @doc """
  Handles hedging-related telemetry events.
  """
  @spec handle_hedging_event(list(), map(), map(), map()) :: :ok
  def handle_hedging_event(event, measurements, metadata, _config) do
    case event do
      [:crucible, :hedging, :winner] ->
        if collector = get_collector(metadata) do
          Crucible.Telemetry.MLMetrics.record(
            collector,
            :hedging,
            %{
              latency: convert_duration(measurements[:duration]),
              winner_index: metadata[:winner_index]
            },
            %{
              strategy: metadata[:strategy],
              experiment_id: metadata[:experiment_id]
            }
          )
        end

      [:crucible, :hedging, :dispatch] ->
        :ok

      _ ->
        :ok
    end

    :ok
  end

  # Tinkex-specific events
  defp handle_tinkex_event(_event, _measurements, _metadata, _config) do
    # These are handled by the TelemetryBridge
    :ok
  end

  # Event lists by category

  defp training_events do
    [
      [:crucible, :training, :start],
      [:crucible, :training, :stop],
      [:crucible, :training, :exception],
      [:crucible, :training, :step],
      [:crucible, :training, :epoch],
      [:crucible, :training, :checkpoint]
    ]
  end

  defp inference_events do
    [
      [:crucible, :inference, :start],
      [:crucible, :inference, :stop]
    ]
  end

  defp ensemble_events do
    [
      [:crucible, :ensemble, :infer],
      [:crucible, :ensemble, :vote]
    ]
  end

  defp harness_events do
    [
      [:crucible, :harness, :stage_start],
      [:crucible, :harness, :stage_stop],
      [:crucible, :harness, :stage_error]
    ]
  end

  defp hedging_events do
    [
      [:crucible, :hedging, :dispatch],
      [:crucible, :hedging, :winner]
    ]
  end

  defp tinkex_events do
    [
      [:crucible, :tinkex, :forward_backward_start],
      [:crucible, :tinkex, :forward_backward_stop],
      [:crucible, :tinkex, :optim_step_start],
      [:crucible, :tinkex, :optim_step_stop],
      [:crucible, :tinkex, :sample_start],
      [:crucible, :tinkex, :sample_stop],
      [:crucible, :tinkex, :session_start],
      [:crucible, :tinkex, :checkpoint_saved]
    ]
  end

  # Helper functions

  defp get_category([:crucible, category | _]), do: category
  defp get_category(_), do: :unknown

  defp maybe_filter_events(events, nil), do: events

  defp maybe_filter_events(events, filters) do
    Enum.filter(events, fn event ->
      get_category(event) in filters
    end)
  end

  defp get_collector(%{experiment_id: experiment_id}) do
    case Registry.lookup(Crucible.Telemetry.Registry, {:collector, experiment_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  rescue
    _ -> nil
  end

  defp get_collector(_), do: nil

  defp get_tracker(%{experiment_id: experiment_id}) do
    case Registry.lookup(Crucible.Telemetry.Registry, {:tracker, experiment_id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  rescue
    _ -> nil
  end

  defp get_tracker(_), do: nil

  defp convert_duration(nil), do: 0

  defp convert_duration(duration) when is_integer(duration) do
    # Convert from native time to milliseconds
    System.convert_time_unit(duration, :native, :millisecond)
  end

  defp convert_duration(duration), do: duration
end
