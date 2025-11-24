defmodule Crucible.Stage.Guardrails do
  @moduledoc """
  Applies guardrail checks to examples before backend calls.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.Stage.Guardrails.Noop

  @impl true
  def run(%Context{examples: nil} = ctx, _opts), do: {:ok, ctx}

  def run(%Context{examples: examples} = ctx, opts) do
    adapter = adapter(opts)

    with {:ok, violations} <- adapter.scan(examples, opts) do
      metrics = %{
        total: length(examples),
        violations: length(violations)
      }

      new_ctx =
        %Context{
          ctx
          | metrics: Map.put(ctx.metrics, :guardrails, metrics),
            assigns: Map.put(ctx.assigns, :guardrail_violations, violations)
        }

      if violations != [] and Map.get(opts, :fail_on_violation, false) do
        {:error, {:guardrail_violation, violations}}
      else
        {:ok, new_ctx}
      end
    end
  end

  defp adapter(opts) do
    mod =
      cond do
        is_map(opts) -> Map.get(opts, :adapter)
        true -> Keyword.get(opts, :adapter)
      end

    mod ||
      Application.get_env(:crucible_framework, :guardrail_adapter, Noop)
  end
end
