defmodule Crucible.Stage.GuardrailsTest do
  use ExUnit.Case, async: true

  import Mox

  alias Crucible.Context
  alias Crucible.Stage.Guardrails

  setup :set_mox_from_context
  setup :verify_on_exit!

  test "passes through when no violations" do
    Crucible.GuardrailMock
    |> expect(:scan, fn _examples, _opts -> {:ok, []} end)

    ctx = %Context{
      experiment_id: "exp",
      run_id: "run",
      experiment: %CrucibleIR.Experiment{
        id: "exp",
        backend: %CrucibleIR.BackendRef{id: :tinkex},
        pipeline: []
      },
      examples: [%{input: "x", output: "y"}]
    }

    assert {:ok, new_ctx} = Guardrails.run(ctx, %{adapter: Crucible.GuardrailMock})
    assert new_ctx.metrics.guardrails.violations == 0
  end

  test "fails fast on violations when configured" do
    Crucible.GuardrailMock
    |> expect(:scan, fn _examples, _opts -> {:ok, [%{issue: :pii}]} end)

    ctx = %Context{
      experiment_id: "exp",
      run_id: "run",
      experiment: %CrucibleIR.Experiment{
        id: "exp",
        backend: %CrucibleIR.BackendRef{id: :tinkex},
        pipeline: []
      },
      examples: [%{input: "x", output: "y"}]
    }

    assert {:error, {:guardrail_violation, _}} =
             Guardrails.run(ctx, %{adapter: Crucible.GuardrailMock, fail_on_violation: true})
  end
end
