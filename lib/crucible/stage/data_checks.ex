defmodule Crucible.Stage.DataChecks do
  @moduledoc """
  Lightweight data validation stage.

  This stage validates data stored in `context.assigns[:examples]`.
  Domain-specific stages should load data into assigns before running this stage.

  This is intentionally minimal; heavier validation can be plugged in by
  providing a custom checker module that returns a list of issues.

  ## Configuration

      %StageDef{
        name: :data_checks,
        options: %{
          required_fields: [:input, :output],  # Fields required in each example
          fail_fast: false,                     # Fail immediately on issues
          checker: MyApp.CustomChecker          # Optional custom checker module
        }
      }
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @impl true
  def run(%Context{assigns: %{examples: nil}} = ctx, _opts), do: {:ok, ctx}

  def run(%Context{assigns: %{examples: examples}} = ctx, opts) when is_list(examples) do
    required_fields = Map.get(opts, :required_fields, [:input, :output])
    checker = Map.get(opts, :checker)

    issues =
      if checker && function_exported?(checker, :run, 2) do
        checker.run(examples, opts)
      else
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
       ctx
       |> Context.put_metric(:data_checks, metrics)
       |> Context.assign(:data_issues, issues)}
    end
  end

  def run(%Context{} = ctx, _opts) do
    # No examples in assigns - nothing to check
    {:ok, ctx}
  end

  @impl true
  def describe(opts) do
    %{
      stage: :data_checks,
      description: "Lightweight data validation",
      required_fields: Map.get(opts, :required_fields, [:input, :output])
    }
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
