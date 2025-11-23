defmodule Crucible.Thinker.Telemetry do
  @moduledoc """
  Telemetry handlers for thinker experiment tracking.

  Attaches handlers to capture all thinker events and optionally
  forward them to crucible_telemetry for persistent storage.
  """

  require Logger

  @events [
    # Harness events
    [:crucible, :thinker, :harness, :start],
    [:crucible, :thinker, :harness, :dataset_loaded],
    [:crucible, :thinker, :harness, :training_complete],
    [:crucible, :thinker, :harness, :evaluation_complete],
    [:crucible, :thinker, :harness, :complete],
    [:crucible, :thinker, :harness, :failed],

    # Validation events
    [:crucible, :thinker, :validation, :complete],
    [:crucible, :thinker, :validation, :entailment, :bumblebee],
    [:crucible, :thinker, :validation, :similarity, :bumblebee],

    # Antagonist events
    [:crucible, :thinker, :antagonist, :complete],

    # Training events
    [:crucible, :thinker, :training, :start],
    [:crucible, :thinker, :training, :progress],
    [:crucible, :thinker, :training, :complete]
  ]

  @doc """
  Returns all thinker telemetry event names.
  """
  @spec events() :: [[atom()]]
  def events, do: @events

  @doc """
  Attaches all thinker telemetry handlers.

  ## Options

  - `:log_level` - Log level for events (:debug, :info, :warning). Default: :debug
  - `:forward_to_research` - Forward to crucible_telemetry Research store. Default: false
  """
  @spec attach(keyword()) :: :ok
  def attach(opts \\ []) do
    log_level = Keyword.get(opts, :log_level, :debug)
    forward = Keyword.get(opts, :forward_to_research, false)

    config = %{
      log_level: log_level,
      forward_to_research: forward
    }

    :telemetry.attach_many(
      "crucible-thinker-telemetry",
      @events,
      &handle_event/4,
      config
    )

    :ok
  end

  @doc """
  Detaches all thinker telemetry handlers.
  """
  @spec detach() :: :ok | {:error, :not_found}
  def detach do
    :telemetry.detach("crucible-thinker-telemetry")
  end

  @doc """
  Handles telemetry events.
  """
  def handle_event(event, measurements, metadata, config) do
    # Log the event
    log_event(event, measurements, metadata, config.log_level)

    # Optionally forward to crucible_telemetry
    if config.forward_to_research do
      forward_to_research(event, measurements, metadata)
    end
  end

  defp log_event(event, measurements, metadata, level) do
    event_name = Enum.join(event, ".")

    message =
      case event do
        [:crucible, :thinker, :harness, :start] ->
          "Experiment started: #{metadata.name} (#{metadata.experiment_id})"

        [:crucible, :thinker, :harness, :dataset_loaded] ->
          "Dataset loaded: #{metadata.count} samples"

        [:crucible, :thinker, :harness, :training_complete] ->
          "Training complete: loss=#{metadata.final_loss}"

        [:crucible, :thinker, :harness, :evaluation_complete] ->
          "Evaluation complete: #{metadata.sample_count} samples"

        [:crucible, :thinker, :harness, :complete] ->
          passed = if metadata.quality_check.passed, do: "PASSED", else: "FAILED"

          details =
            metadata.quality_check.details
            |> Enum.map(fn d ->
              status = if d.passed, do: "✓", else: "✗"

              "  #{status} #{d.metric}: #{Float.round(d.actual * 100, 1)}% (threshold: #{Float.round(d.threshold * 100, 1)}%)"
            end)
            |> Enum.join("\n")

          "Experiment complete: #{passed}\n[quality]\n#{details}"

        [:crucible, :thinker, :harness, :failed] ->
          "Experiment failed: #{metadata.error}"

        [:crucible, :thinker, :validation, :complete] ->
          "Validation: #{metadata.claim_count} claims, schema=#{measurements.schema_compliance}"

        [:crucible, :thinker, :antagonist, :complete] ->
          "Antagonist: #{measurements.total_claims} claims, #{measurements.total_issues} issues"

        _ ->
          "#{event_name}: #{inspect(measurements)}"
      end

    case level do
      :debug -> Logger.debug(message)
      :info -> Logger.info(message)
      :warning -> Logger.warning(message)
      _ -> Logger.debug(message)
    end
  end

  defp forward_to_research(event, measurements, metadata) do
    # Check if crucible_telemetry Research module is available
    if Code.ensure_loaded?(Crucible.Telemetry.Research) do
      Crucible.Telemetry.Research.capture(%{
        event: event,
        measurements: measurements,
        metadata: metadata,
        timestamp: DateTime.utc_now()
      })
    end
  end
end
