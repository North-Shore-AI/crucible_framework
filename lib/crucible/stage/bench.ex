defmodule Crucible.Stage.Bench do
  @moduledoc """
  Lightweight benchmarking stage that summarizes metrics collected so far.

  This is a placeholder for crucible_bench integration and keeps the IR
  contract intact.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @impl true
  def run(%Context{} = ctx, opts) do
    backend_metrics = Map.get(ctx.metrics, :backend, %{})
    guardrail_metrics = Map.get(ctx.metrics, :guardrails, %{})

    summary = %{
      backend: backend_metrics,
      guardrails: guardrail_metrics,
      alpha: opts |> Map.get(:alpha, 0.05),
      tests: opts |> Map.get(:tests, [])
    }

    {:ok, %Context{ctx | metrics: Map.put(ctx.metrics, :bench, summary)}}
  end
end
