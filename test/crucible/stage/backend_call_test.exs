defmodule Crucible.Stage.BackendCallTest do
  use ExUnit.Case, async: true

  import Mox

  alias Crucible.Context
  alias Crucible.IR.{BackendRef, Experiment}
  alias Crucible.Stage.BackendCall

  setup :set_mox_from_context
  setup :verify_on_exit!

  setup do
    Application.put_env(:crucible_framework, :backends, %{mock: Crucible.BackendMock})
    :ok
  end

  test "runs training batches and collects metrics" do
    expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :mock, options: %{}},
      pipeline: []
    }

    expect(Crucible.BackendMock, :start_session, fn :state, ^experiment -> {:ok, :session} end)

    expect(Crucible.BackendMock, :train_step, fn :session, _batch ->
      {:ok, %{loss: 0.1, batch_size: 1, metrics: %{}}}
    end)

    expect(Crucible.BackendMock, :save_checkpoint, fn :session, 1 -> {:ok, :checkpoint} end)
    expect(Crucible.BackendMock, :create_sampler, fn :session, :checkpoint -> {:ok, :sampler} end)
    expect(Crucible.BackendMock, :sample, fn :sampler, "prompt", _opts -> {:ok, ["ok"]} end)

    ctx = %Context{
      experiment_id: experiment.id,
      run_id: "run",
      experiment: experiment,
      batches: [[%{input: "a", output: "b"}]],
      outputs: []
    }

    assert {:ok, new_ctx} =
             BackendCall.run(ctx, %{
               mode: :train,
               sample_prompts: ["prompt"],
               create_sampler?: true
             })

    assert new_ctx.metrics.backend.mean_loss == 0.1
    assert new_ctx.assigns.checkpoint_ref == :checkpoint
    assert new_ctx.outputs == [%{prompt: "prompt", responses: ["ok"]}]
  end
end
