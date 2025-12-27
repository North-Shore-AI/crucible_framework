#!/usr/bin/env elixir

# Optional bench example. Runs with or without crucible_bench installed.

alias Crucible.Context
alias CrucibleFramework
alias CrucibleIR.{BackendRef, Experiment, OutputSpec, StageDef}

defmodule Example.SeedBenchMetrics do
  @behaviour Crucible.Stage

  @impl true
  def run(%Context{} = ctx, _opts) do
    baseline = [0.71, 0.69, 0.73, 0.72, 0.70, 0.68, 0.74, 0.69]
    treatment = [0.78, 0.76, 0.81, 0.79, 0.77, 0.75, 0.80, 0.78]

    ctx = Context.merge_metrics(ctx, %{baseline: baseline, treatment: treatment})

    {:ok, ctx}
  end

  @impl true
  def describe(_opts) do
    %{
      name: :seed_metrics,
      description: "Seeds baseline/treatment metrics for bench",
      required: [],
      optional: [],
      types: %{}
    }
  end
end

IO.puts("== Optional bench example ==")

bench_available? = Code.ensure_loaded?(CrucibleBench)

if bench_available? do
  IO.puts("crucible_bench detected; running bench stage.")
else
  IO.puts("crucible_bench not installed; running without bench stage.")
  IO.puts("Add {:crucible_bench, \"~> 0.4.0\"} to deps to enable.")
end

bench_stage =
  if bench_available? do
    [
      %StageDef{
        name: :bench,
        options: %{data_source: :metrics, tests: [:ttest], alpha: 0.05}
      }
    ]
  else
    []
  end

pipeline =
  [
    %StageDef{name: :validate},
    %StageDef{name: :seed_metrics, module: Example.SeedBenchMetrics}
  ] ++
    bench_stage ++
    [
      %StageDef{name: :report}
    ]

experiment = %Experiment{
  id: "bench_optional_demo",
  description: "Bench stage demo with optional dependency",
  backend: %BackendRef{id: :noop},
  pipeline: pipeline,
  outputs: [
    %OutputSpec{name: :summary, formats: [:markdown], sink: :stdout}
  ]
}

case CrucibleFramework.run(experiment, persist: false) do
  {:ok, ctx} ->
    IO.puts("Pipeline complete.")

    case Map.get(ctx.metrics, :bench) do
      nil ->
        IO.puts("Bench results not available.")

      bench ->
        IO.inspect(bench, label: "bench metrics")
    end

  {:error, {stage, reason, _ctx}} ->
    IO.puts("Pipeline failed at #{inspect(stage)}: #{inspect(reason)}")
    System.halt(1)
end
