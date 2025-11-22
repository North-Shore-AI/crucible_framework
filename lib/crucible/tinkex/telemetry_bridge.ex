defmodule Crucible.Tinkex.TelemetryBridge do
  @moduledoc """
  Translates Tinkex telemetry events to Crucible telemetry format.

  This module attaches handlers to Tinkex telemetry events and translates
  them into the Crucible telemetry namespace, enriching them with
  experiment context and adapter metadata.

  ## Event Translation

  Tinkex events are translated as follows:

  - `[:tinkex, :request, :start]` with ForwardBackward -> `[:crucible, :tinkex, :forward_backward_start]`
  - `[:tinkex, :request, :stop]` with ForwardBackward -> `[:crucible, :tinkex, :forward_backward_stop]`
  - `[:tinkex, :request, :start]` with OptimStep -> `[:crucible, :tinkex, :optim_step_start]`
  - `[:tinkex, :request, :stop]` with OptimStep -> `[:crucible, :tinkex, :optim_step_stop]`
  - `[:tinkex, :request, :start]` with Sample -> `[:crucible, :tinkex, :sample_start]`
  - `[:tinkex, :request, :stop]` with Sample -> `[:crucible, :tinkex, :sample_stop]`
  - `[:tinkex, :request, :exception]` -> `[:crucible, :tinkex, :request_exception]`

  ## Usage

      # Attach handlers at application startup
      Crucible.Tinkex.TelemetryBridge.attach_handlers()

      # Detach when no longer needed
      Crucible.Tinkex.TelemetryBridge.detach_handlers()

  ## Configuration

  The bridge can be configured with an experiment context:

      Crucible.Tinkex.TelemetryBridge.attach_handlers(
        experiment_id: "exp-123",
        extra_metadata: %{adapter: :tinkex}
      )
  """

  @handler_id "crucible-tinkex-bridge"

  @doc """
  Attaches telemetry handlers to translate Tinkex events.

  ## Options

  - `:experiment_id` - The current experiment ID to include in metadata
  - `:extra_metadata` - Additional metadata to include with all events

  ## Examples

      Crucible.Tinkex.TelemetryBridge.attach_handlers()

      Crucible.Tinkex.TelemetryBridge.attach_handlers(
        experiment_id: "exp-123"
      )
  """
  @spec attach_handlers(keyword()) :: :ok | {:error, :already_exists}
  def attach_handlers(opts \\ []) do
    config = %{
      experiment_id: Keyword.get(opts, :experiment_id),
      extra_metadata: Keyword.get(opts, :extra_metadata, %{})
    }

    :telemetry.attach_many(
      @handler_id,
      [
        [:tinkex, :request, :start],
        [:tinkex, :request, :stop],
        [:tinkex, :request, :exception]
      ],
      &handle_event/4,
      config
    )
  end

  @doc """
  Detaches the telemetry handlers.
  """
  @spec detach_handlers() :: :ok | {:error, :not_found}
  def detach_handlers do
    :telemetry.detach(@handler_id)
  end

  @doc """
  Handles incoming Tinkex telemetry events and translates them.
  """
  @spec handle_event(list(), map(), map(), map()) :: :ok
  def handle_event(event, measurements, metadata, config) do
    crucible_event = translate_event(event, metadata)

    :telemetry.execute(
      crucible_event,
      enrich_measurements(measurements),
      enrich_metadata(metadata, config)
    )
  end

  # Translate Tinkex events to Crucible events
  defp translate_event([:tinkex, :request, :start], %{tinker_request_type: "ForwardBackward"}) do
    [:crucible, :tinkex, :forward_backward_start]
  end

  defp translate_event([:tinkex, :request, :stop], %{tinker_request_type: "ForwardBackward"}) do
    [:crucible, :tinkex, :forward_backward_stop]
  end

  defp translate_event([:tinkex, :request, :start], %{tinker_request_type: "OptimStep"}) do
    [:crucible, :tinkex, :optim_step_start]
  end

  defp translate_event([:tinkex, :request, :stop], %{tinker_request_type: "OptimStep"}) do
    [:crucible, :tinkex, :optim_step_stop]
  end

  defp translate_event([:tinkex, :request, :start], %{tinker_request_type: "Sample"}) do
    [:crucible, :tinkex, :sample_start]
  end

  defp translate_event([:tinkex, :request, :stop], %{tinker_request_type: "Sample"}) do
    [:crucible, :tinkex, :sample_stop]
  end

  defp translate_event([:tinkex, :request, :start], %{
         tinker_request_type: "SaveWeightsForSampler"
       }) do
    [:crucible, :tinkex, :save_weights_start]
  end

  defp translate_event([:tinkex, :request, :stop], %{tinker_request_type: "SaveWeightsForSampler"}) do
    [:crucible, :tinkex, :save_weights_stop]
  end

  defp translate_event([:tinkex, :request, :exception], _metadata) do
    [:crucible, :tinkex, :request_exception]
  end

  defp translate_event([:tinkex, :request, event], _metadata) do
    [:crucible, :tinkex, event]
  end

  # Enrich measurements with Crucible-specific data
  defp enrich_measurements(measurements) do
    Map.merge(measurements, %{
      crucible_timestamp: System.system_time(:millisecond)
    })
  end

  # Enrich metadata with Crucible context
  defp enrich_metadata(metadata, config) do
    base_metadata = %{
      adapter: :tinkex,
      crucible_version: crucible_version()
    }

    experiment_metadata =
      if config.experiment_id do
        %{experiment_id: config.experiment_id}
      else
        %{}
      end

    metadata
    |> Map.merge(base_metadata)
    |> Map.merge(experiment_metadata)
    |> Map.merge(config.extra_metadata)
  end

  defp crucible_version do
    case :application.get_key(:crucible_framework, :vsn) do
      {:ok, vsn} -> to_string(vsn)
      :undefined -> "unknown"
    end
  end
end
