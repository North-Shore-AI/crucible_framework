#!/usr/bin/env elixir

# Core pipeline example using only built-in stages.

alias Crucible.Context
alias CrucibleFramework
alias CrucibleIR.{BackendRef, Experiment, OutputSpec, StageDef}

defmodule Example.SeedExamples do
  @behaviour Crucible.Stage

  @impl true
  def run(%Context{} = ctx, _opts) do
    examples = [
      %{input: "What is 2 + 2?", output: "4"},
      %{input: "Summarize: N-Queens", output: "Constraint satisfaction problem."},
      %{input: "Translate 'hello' to Spanish", output: "hola"}
    ]

    ctx =
      ctx
      |> Context.assign(:examples, examples)
      |> Context.put_metric(:examples_loaded, length(examples))

    {:ok, ctx}
  end

  @impl true
  def describe(_opts) do
    %{
      name: :seed_examples,
      description: "Seeds demo examples into context.assigns",
      required: [],
      optional: [],
      types: %{}
    }
  end
end

IO.puts("== Core pipeline example ==")

experiment = %Experiment{
  id: "core_pipeline_demo",
  description: "Basic pipeline with built-in stages",
  backend: %BackendRef{id: :noop},
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :seed_examples, module: Example.SeedExamples},
    %StageDef{name: :data_checks, options: %{required_fields: [:input, :output]}},
    %StageDef{name: :guardrails, options: %{fail_on_violation: false}},
    %StageDef{name: :report}
  ],
  outputs: [
    %OutputSpec{name: :summary, formats: [:markdown], sink: :stdout}
  ]
}

case CrucibleFramework.run(experiment, persist: false) do
  {:ok, ctx} ->
    IO.puts("Pipeline complete.")
    IO.inspect(ctx.metrics, label: "metrics")
    IO.puts("Completed stages: #{inspect(Context.completed_stages(ctx))}")

  {:error, {stage, reason, _ctx}} ->
    IO.puts("Pipeline failed at #{inspect(stage)}: #{inspect(reason)}")
    System.halt(1)
end
