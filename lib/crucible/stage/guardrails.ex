defmodule Crucible.Stage.Guardrails do
  @moduledoc """
  Applies guardrail checks to examples before processing.

  This stage scans examples stored in `context.assigns[:examples]` using
  a configured guardrail adapter. Domain-specific stages should load data
  into assigns before running this stage.

  ## Configuration

      %StageDef{
        name: :guardrails,
        options: %{
          adapter: MyApp.GuardrailAdapter,  # Optional, uses config default
          fail_on_violation: false          # Fail on any violation
        }
      }

  ## Adapter Behaviour

  Adapters must implement `Crucible.Stage.Guardrails.Adapter`:

      defmodule MyAdapter do
        @behaviour Crucible.Stage.Guardrails.Adapter

        @impl true
        def scan(examples, opts) do
          # Return {:ok, violations} or {:error, reason}
        end
      end
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.Stage.Guardrails.Noop

  @impl true
  def run(%Context{assigns: %{examples: nil}} = ctx, _opts), do: {:ok, ctx}

  def run(%Context{assigns: %{examples: examples}} = ctx, opts) when is_list(examples) do
    adapter = adapter(opts)

    with {:ok, violations} <- adapter.scan(examples, opts) do
      metrics = %{
        total: length(examples),
        violations: length(violations)
      }

      new_ctx =
        ctx
        |> Context.put_metric(:guardrails, metrics)
        |> Context.assign(:guardrail_violations, violations)

      if violations != [] and Map.get(opts, :fail_on_violation, false) do
        {:error, {:guardrail_violation, violations}}
      else
        {:ok, new_ctx}
      end
    end
  end

  def run(%Context{} = ctx, _opts) do
    # No examples in assigns - nothing to check
    {:ok, ctx}
  end

  @impl true
  def describe(_opts) do
    %{
      name: :guardrails,
      description: "Applies safety guardrail checks to examples via configurable adapter",
      required: [],
      optional: [:adapter, :fail_on_violation],
      types: %{
        adapter: :module,
        fail_on_violation: :boolean
      }
    }
  end

  defp adapter(opts) do
    mod =
      cond do
        is_map(opts) -> Map.get(opts, :adapter)
        is_list(opts) -> Keyword.get(opts, :adapter)
        true -> nil
      end

    mod ||
      Application.get_env(:crucible_framework, :guardrail_adapter, Noop)
  end
end
