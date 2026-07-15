#!/usr/bin/env elixir

# Plan-driven pipeline example using Jido.Plan + Crucible.PlanAdapter.

alias Crucible.PlanAdapter
alias CrucibleFramework
alias CrucibleIR.{BackendRef, Experiment}

if Code.ensure_loaded?(Jido.Plan) do
  if Code.ensure_loaded?(Jido.Action) do
    defmodule Example.Actions.Echo do
      use Jido.Action,
        name: "echo",
        description: "Echoes params back to the caller"

      @impl true
      def run(params, _context), do: {:ok, params}
    end
  else
    defmodule Example.Actions.Echo do
      def run(params, _context), do: {:ok, params}
    end
  end

  IO.puts("== Plan pipeline example ==")

  plan =
    Jido.Plan.new()
    |> Jido.Plan.add(:hello, {Example.Actions.Echo, %{message: "hello"}})
    |> Jido.Plan.add(:world, {Example.Actions.Echo, %{message: "world"}},
      depends_on: :hello
    )

  {:ok, stage_defs} = PlanAdapter.to_stage_defs(plan)

  experiment = %Experiment{
    id: "plan_pipeline_demo",
    backend: %BackendRef{id: :noop},
    pipeline: stage_defs
  }

  case CrucibleFramework.run(experiment, persist: false, enable_lineage: true) do
    {:ok, ctx} ->
      IO.inspect(ctx.assigns.plan_results, label: "plan_results")
      IO.puts("Lineage spans: #{length(ctx.assigns[:lineage_spans] || [])}")

    {:error, {stage, reason, _ctx}} ->
      IO.puts("Pipeline failed at #{inspect(stage)}: #{inspect(reason)}")
      System.halt(1)
  end
else
  IO.puts("Jido.Plan not available; add {:jido_action, \"~> 1.0\"} to deps to run.")
end
