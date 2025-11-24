defmodule Crucible.Stage.DataChecks do
  @moduledoc """
  Lightweight data validation stage.

  This is intentionally minimal; heavier validation can be plugged in by
  providing a custom checker module that returns a list of issues.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @impl true
  def run(%Context{examples: nil} = ctx, _opts), do: {:ok, ctx}

  def run(%Context{examples: examples} = ctx, opts) do
    required_fields = Map.get(opts, :required_fields, [:input, :output])
    checker = Map.get(opts, :checker)

    issues =
      cond do
        checker && function_exported?(checker, :run, 2) ->
          checker.run(examples, opts)

        true ->
          built_in_checks(examples, required_fields)
      end

    metrics = %{
      total: length(examples),
      missing_required: length(issues),
      required_fields: required_fields
    }

    if Map.get(opts, :fail_fast, false) and issues != [] do
      {:error, {:data_checks_failed, issues}}
    else
      {:ok,
       %Context{
         ctx
         | metrics: Map.put(ctx.metrics, :data_checks, metrics),
           assigns: Map.put(ctx.assigns, :data_issues, issues)
       }}
    end
  end

  defp built_in_checks(examples, required_fields) do
    Enum.reduce(examples, [], fn example, acc ->
      missing =
        required_fields
        |> Enum.filter(fn field -> Map.get(example, field) in [nil, ""] end)

      case missing do
        [] -> acc
        fields -> [%{example: example, missing: fields} | acc]
      end
    end)
  end
end
