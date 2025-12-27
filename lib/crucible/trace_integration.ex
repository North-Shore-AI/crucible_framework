defmodule Crucible.TraceIntegration do
  @moduledoc """
  Integration module for crucible_trace within the experiment pipeline.

  This module provides helpers for managing trace chains throughout the
  experiment lifecycle, including:

  - Creating and initializing trace chains
  - Emitting stage transition events
  - Capturing decision points and alternatives
  - Persisting and exporting traces

  ## Usage

  Traces are automatically managed when enabled in the experiment configuration.
  Each stage can emit trace events to capture its decision-making process.

  ## Event Types

  The following event types are emitted by the pipeline:

  - `:stage_start` - When a stage begins execution
  - `:stage_complete` - When a stage completes successfully
  - `:stage_failed` - When a stage encounters an error
  - `:hypothesis_formed` - When forming a hypothesis about approach
  - `:pattern_applied` - When applying a known pattern
  - `:alternative_considered` - When considering alternatives
  - `:decision_made` - When making a final decision

  ## Example

      # In a stage implementation
      ctx = TraceIntegration.emit_event(
        ctx,
        :hypothesis_formed,
        "Use ensemble for improved accuracy",
        "Multiple models can provide better consensus",
        alternatives: ["Single model", "Sequential cascade"],
        confidence: 0.85
      )
  """

  alias Crucible.Context
  alias CrucibleTrace

  require Logger

  @compile {:no_warn_undefined, CrucibleTrace}
  @compile {:no_warn_undefined, CrucibleTrace.Chain}

  if Code.ensure_loaded?(CrucibleTrace) do
    @type trace_chain :: CrucibleTrace.Chain.t()
    @type trace_event :: CrucibleTrace.Event.t()
  else
    @type trace_chain :: term()
    @type trace_event :: term()
  end

  @doc """
  Initialize a new trace chain for an experiment.

  Creates a new trace chain and adds it to the context.
  """
  @spec init_trace(Context.t(), String.t()) :: Context.t()
  def init_trace(%Context{} = ctx, experiment_name) do
    if trace_available?() do
      metadata = %{
        experiment_id: ctx.experiment_id,
        run_id: ctx.run_id
      }

      chain = CrucibleTrace.new_chain(experiment_name, metadata: metadata)

      %Context{ctx | trace: chain}
    else
      warn_missing_trace()
      ctx
    end
  end

  @doc """
  Emit a trace event to the current chain.

  If tracing is not enabled (trace is nil), returns the context unchanged.

  ## Parameters

  - `ctx` - The current context
  - `type` - Event type atom
  - `decision` - What was decided
  - `reasoning` - Why this decision was made
  - `opts` - Additional event options

  ## Options

  - `:alternatives` - List of alternatives considered
  - `:confidence` - Confidence level (0.0-1.0)
  - `:metadata` - Additional metadata map
  - `:code_section` - Related code section
  - `:spec_reference` - Related specification reference
  """
  @spec emit_event(Context.t(), atom(), String.t(), String.t(), keyword()) :: Context.t()
  def emit_event(ctx, type, decision, reasoning),
    do: emit_event(ctx, type, decision, reasoning, [])

  def emit_event(%Context{trace: nil} = ctx, _type, _decision, _reasoning, _opts) do
    # Tracing not enabled
    ctx
  end

  def emit_event(%Context{trace: chain} = ctx, type, decision, reasoning, opts)
      when not is_nil(chain) do
    if trace_available?() do
      event = CrucibleTrace.create_event(type, decision, reasoning, opts)
      new_chain = CrucibleTrace.add_event(chain, event)
      %Context{ctx | trace: new_chain}
    else
      ctx
    end
  end

  def emit_event(ctx, _, _, _, _), do: ctx

  @doc """
  Emit a stage start event.

  Captures the beginning of a pipeline stage execution.
  """
  @spec emit_stage_start(Context.t(), atom(), map()) :: Context.t()
  def emit_stage_start(ctx, stage_name, stage_opts \\ %{}) do
    emit_event(
      ctx,
      :stage_start,
      "Starting stage: #{stage_name}",
      "Pipeline executing stage #{stage_name}",
      metadata: %{
        stage: stage_name,
        options: stage_opts,
        timestamp: DateTime.utc_now()
      }
    )
  end

  @doc """
  Emit a stage completion event.

  Captures successful completion of a pipeline stage.
  """
  @spec emit_stage_complete(Context.t(), atom(), map()) :: Context.t()
  def emit_stage_complete(ctx, stage_name, results \\ %{}) do
    emit_event(
      ctx,
      :stage_complete,
      "Completed stage: #{stage_name}",
      "Stage #{stage_name} completed successfully",
      metadata: %{
        stage: stage_name,
        results: summarize_results(results),
        timestamp: DateTime.utc_now()
      }
    )
  end

  @doc """
  Emit a stage failure event.

  Captures when a pipeline stage encounters an error.
  """
  @spec emit_stage_failed(Context.t(), atom(), term()) :: Context.t()
  def emit_stage_failed(ctx, stage_name, error) do
    emit_event(
      ctx,
      :stage_failed,
      "Failed stage: #{stage_name}",
      "Stage #{stage_name} failed with error: #{inspect(error)}",
      metadata: %{
        stage: stage_name,
        error: inspect(error),
        timestamp: DateTime.utc_now()
      },
      confidence: 0.0
    )
  end

  @doc """
  Emit a decision event with alternatives.

  Captures a decision point where alternatives were considered.
  """
  @spec emit_decision(Context.t(), String.t(), String.t(), [String.t()], float()) :: Context.t()
  def emit_decision(ctx, decision, reasoning, alternatives, confidence) do
    emit_event(
      ctx,
      :decision_made,
      decision,
      reasoning,
      alternatives: alternatives,
      confidence: confidence
    )
  end

  @doc """
  Export the trace chain to JSON.

  Returns the JSON representation of the trace chain, or nil if tracing is not enabled.
  """
  @spec export_json(Context.t()) :: String.t() | nil
  def export_json(%Context{trace: nil}), do: nil

  def export_json(%Context{trace: chain}) when not is_nil(chain) do
    if trace_available?() do
      case CrucibleTrace.export(chain, :json) do
        {:ok, json} -> json
        _ -> nil
      end
    else
      nil
    end
  end

  def export_json(_), do: nil

  @doc """
  Generate an HTML visualization of the trace.

  Returns the HTML content for visualizing the trace chain, or nil if tracing is not enabled.
  """
  @spec export_html(Context.t()) :: String.t() | nil
  def export_html(%Context{trace: nil}), do: nil

  def export_html(%Context{trace: chain}) when not is_nil(chain) do
    if trace_available?() do
      CrucibleTrace.visualize(chain)
    else
      nil
    end
  end

  def export_html(_), do: nil

  @doc """
  Save the trace to disk.

  Persists the trace chain to the filesystem.
  """
  @spec save_trace(Context.t(), String.t()) :: :ok | {:error, term()}
  def save_trace(%Context{trace: nil}, _path), do: {:error, :no_trace}

  def save_trace(%Context{trace: chain}, path) when not is_nil(chain) do
    if trace_available?() do
      with :ok <- File.mkdir_p(Path.dirname(path)),
           {:ok, json} <- CrucibleTrace.export(chain, :json) do
        File.write(path, json)
      end
    else
      {:error, {:missing_dependency, :crucible_trace}}
    end
  end

  def save_trace(_, _), do: {:error, :invalid_context}

  @doc """
  Load a trace from disk.

  Loads a previously saved trace chain from the filesystem.
  """
  @spec load_trace(String.t()) ::
          {:ok, trace_chain()} | {:error, {:missing_dependency, :crucible_trace} | term()}
  def load_trace(path) do
    if trace_available?() do
      with {:ok, content} <- File.read(path),
           {:ok, map} <- Jason.decode(content) do
        {:ok, CrucibleTrace.Chain.from_map(map)}
      end
    else
      {:error, {:missing_dependency, :crucible_trace}}
    end
  end

  @doc """
  Check if tracing is enabled for the context.
  """
  @spec tracing_enabled?(Context.t()) :: boolean()
  def tracing_enabled?(%Context{trace: nil}), do: false
  def tracing_enabled?(%Context{trace: _}), do: true
  def tracing_enabled?(_), do: false

  @doc """
  Get event count from the trace chain.
  """
  @spec event_count(Context.t()) :: non_neg_integer()
  def event_count(%Context{trace: nil}), do: 0

  def event_count(%Context{trace: chain}) when not is_nil(chain) do
    length(chain.events)
  end

  def event_count(_), do: 0

  @doc """
  Filter trace events by type.

  Returns all events of a specific type from the trace chain.
  """
  @spec filter_events(Context.t(), atom()) :: [trace_event()]
  def filter_events(%Context{trace: nil}, _type), do: []

  def filter_events(%Context{trace: chain}, type) when not is_nil(chain) do
    Enum.filter(chain.events, fn event -> event.type == type end)
  end

  def filter_events(_, _), do: []

  @doc """
  Get the most recent event from the trace.
  """
  @spec last_event(Context.t()) :: trace_event() | nil
  def last_event(%Context{trace: nil}), do: nil

  def last_event(%Context{trace: chain}) when not is_nil(chain) do
    List.last(chain.events)
  end

  def last_event(_), do: nil

  @doc """
  Calculate confidence statistics from the trace.

  Returns statistics about confidence levels across all decisions.
  """
  @spec confidence_stats(Context.t()) :: map()
  def confidence_stats(%Context{trace: nil}), do: %{}

  def confidence_stats(%Context{trace: chain}) when not is_nil(chain) do
    confidences =
      chain.events
      |> Enum.map(fn e -> e.confidence end)
      |> Enum.filter(&(not is_nil(&1)))

    if confidences == [] do
      %{count: 0}
    else
      count = length(confidences)
      mean = Float.round(Enum.sum(confidences) / count, 4)

      %{
        count: count,
        mean: mean,
        min: Enum.min(confidences),
        max: Enum.max(confidences)
      }
    end
  end

  def confidence_stats(_), do: %{}

  # Private functions

  defp summarize_results(results) when is_map(results) do
    # Summarize results to avoid huge metadata
    results
    |> Enum.take(10)
    |> Map.new()
  end

  defp summarize_results(results) when is_list(results) do
    %{
      count: length(results),
      sample: Enum.take(results, 3)
    }
  end

  defp summarize_results(results), do: results

  defp trace_available? do
    Code.ensure_loaded?(CrucibleTrace)
  end

  defp warn_missing_trace do
    Logger.warning(
      "crucible_trace dependency not available; tracing disabled. Add {:crucible_trace, \"~> 0.3.0\"} or set enable_trace: false"
    )
  end
end
