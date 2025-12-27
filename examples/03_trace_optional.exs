#!/usr/bin/env elixir

# Optional trace example. Runs with or without crucible_trace installed.

alias Crucible.Context
alias Crucible.TraceIntegration
alias CrucibleFramework
alias CrucibleIR.{BackendRef, Experiment, OutputSpec, StageDef}

defmodule Example.DecisionStage do
  @behaviour Crucible.Stage

  @impl true
  def run(%Context{} = ctx, _opts) do
    ctx =
      TraceIntegration.emit_event(
        ctx,
        :decision_made,
        "Select fallback strategy",
        "Demo decision with two options",
        alternatives: ["fast_path", "safe_path"],
        confidence: 0.72
      )

    ctx = Context.put_metric(ctx, :decision, %{choice: :safe_path, confidence: 0.72})

    {:ok, ctx}
  end

  @impl true
  def describe(_opts) do
    %{
      name: :decision_stage,
      description: "Emits a trace event and records a decision metric",
      required: [],
      optional: [],
      types: %{}
    }
  end
end

IO.puts("== Optional trace example ==")

trace_available? = Code.ensure_loaded?(CrucibleTrace)

if trace_available? do
  IO.puts("crucible_trace detected; enabling trace integration.")
else
  IO.puts("crucible_trace not installed; running without tracing.")
  IO.puts("Add {:crucible_trace, \"~> 0.3.0\"} to deps to enable.")
end

experiment = %Experiment{
  id: "trace_optional_demo",
  description: "Trace integration demo with optional dependency",
  backend: %BackendRef{id: :noop},
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :decision_stage, module: Example.DecisionStage},
    %StageDef{name: :report}
  ],
  outputs: [
    %OutputSpec{name: :summary, formats: [:markdown], sink: :stdout}
  ]
}

case CrucibleFramework.run(experiment, persist: false, enable_trace: trace_available?) do
  {:ok, ctx} ->
    IO.puts("Pipeline complete.")
    IO.puts("Tracing enabled? #{TraceIntegration.tracing_enabled?(ctx)}")
    IO.puts("Trace event count: #{TraceIntegration.event_count(ctx)}")

    case TraceIntegration.last_event(ctx) do
      nil ->
        IO.puts("No trace events available.")

      event ->
        IO.puts("Last event type: #{event.type}")
    end

    case TraceIntegration.export_json(ctx) do
      nil ->
        IO.puts("Trace export not available.")

      json ->
        IO.puts("Trace JSON bytes: #{byte_size(json)}")
    end

  {:error, {stage, reason, _ctx}} ->
    IO.puts("Pipeline failed at #{inspect(stage)}: #{inspect(reason)}")
    System.halt(1)
end
