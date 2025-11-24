defmodule Crucible.Stage.DataLoadTest do
  use ExUnit.Case, async: true

  alias Crucible.Context
  alias Crucible.IR.{BackendRef, DatasetRef, Experiment}
  alias Crucible.Stage.DataLoad

  defmodule FakeProvider do
    @behaviour Crucible.Data.Provider
    def load(_ref, _opts), do: {:ok, [%{input: "a", output: "b"}, %{input: "c", output: "d"}]}
  end

  test "loads data from in-memory provider and batches it" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :tinkex},
      dataset: %DatasetRef{name: "mem", options: %{data: [%{input: "x", output: "y"}]}},
      pipeline: []
    }

    ctx = %Context{experiment_id: experiment.id, run_id: "r1", experiment: experiment}

    assert {:ok, new_ctx} = DataLoad.run(ctx, %{batch_size: 1})
    assert length(new_ctx.examples) == 1
    assert length(new_ctx.batches) == 1
    assert new_ctx.metrics.data_load.batch_size == 1
  end

  test "respects limit and custom provider" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :tinkex},
      dataset: %DatasetRef{name: "custom", provider: FakeProvider},
      pipeline: []
    }

    ctx = %Context{experiment_id: experiment.id, run_id: "r2", experiment: experiment}

    assert {:ok, new_ctx} = DataLoad.run(ctx, %{limit: 1, batch_size: 2})
    assert length(new_ctx.examples) == 1
    assert new_ctx.batches == [[%{input: "a", output: "b"}]]
  end

  test "maps fields into input/output keys" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :tinkex},
      dataset: %DatasetRef{name: "mem", options: %{data: [%{prompt: "p", completion: "c"}]}},
      pipeline: []
    }

    ctx = %Context{experiment_id: experiment.id, run_id: "r3", experiment: experiment}

    assert {:ok, new_ctx} =
             DataLoad.run(ctx, %{input_key: :prompt, output_key: :completion, batch_size: 1})

    assert [%{input: "p", output: "c"}] = new_ctx.examples
  end
end
