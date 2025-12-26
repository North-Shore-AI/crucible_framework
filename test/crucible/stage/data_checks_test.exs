defmodule Crucible.Stage.DataChecksTest do
  use ExUnit.Case, async: true

  alias Crucible.Context
  alias Crucible.Stage.DataChecks

  test "flags missing required fields" do
    ctx =
      %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %CrucibleIR.Experiment{
          id: "exp",
          backend: %CrucibleIR.BackendRef{id: :tinkex},
          pipeline: []
        }
      }
      |> Context.assign(:examples, [%{input: "a"}, %{input: "b", output: "c"}])

    {:ok, new_ctx} = DataChecks.run(ctx, %{required_fields: [:input, :output]})

    assert new_ctx.metrics.data_checks.missing_required == 1
    assert length(new_ctx.assigns.data_issues) == 1
  end

  test "fails fast when configured" do
    ctx =
      %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %CrucibleIR.Experiment{
          id: "exp",
          backend: %CrucibleIR.BackendRef{id: :tinkex},
          pipeline: []
        }
      }
      |> Context.assign(:examples, [%{input: "a"}])

    assert {:error, {:data_checks_failed, _}} =
             DataChecks.run(ctx, %{required_fields: [:output], fail_fast: true})
  end

  test "passes when no examples in assigns" do
    ctx = %Context{
      experiment_id: "exp",
      run_id: "run",
      experiment: %CrucibleIR.Experiment{
        id: "exp",
        backend: %CrucibleIR.BackendRef{id: :tinkex},
        pipeline: []
      }
    }

    {:ok, new_ctx} = DataChecks.run(ctx, %{})
    assert new_ctx == ctx
  end
end
