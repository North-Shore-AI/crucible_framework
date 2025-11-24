defmodule Crucible.Pipeline.RunnerTest do
  use ExUnit.Case, async: true

  alias Crucible.IR.{BackendRef, Experiment, StageDef}
  alias Crucible.Pipeline.Runner

  defmodule StageOne do
    @behaviour Crucible.Stage
    alias Crucible.Context
    @impl true
    def run(%Context{} = ctx, _opts),
      do: {:ok, %Context{ctx | assigns: Map.put(ctx.assigns, :one, true)}}
  end

  defmodule StageFail do
    @behaviour Crucible.Stage
    alias Crucible.Context
    @impl true
    def run(%Context{} = _ctx, _opts), do: {:error, :nope}
  end

  setup do
    Application.put_env(:crucible_framework, :stage_registry, %{one: StageOne, fail: StageFail})
    :ok
  end

  test "runs configured stages in order" do
    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :tinkex},
      pipeline: [%StageDef{name: :one}]
    }

    assert {:ok, ctx} = Runner.run(experiment, persist: false)
    assert ctx.assigns.one
  end

  test "halts on stage errors" do
    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :tinkex},
      pipeline: [%StageDef{name: :fail}, %StageDef{name: :one}]
    }

    assert {:error, {:fail, :nope, _ctx}} = Runner.run(experiment, persist: false)
  end
end
