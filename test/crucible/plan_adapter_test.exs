defmodule Crucible.PlanAdapterTest do
  use ExUnit.Case, async: true

  alias Crucible.PlanAdapter
  alias CrucibleIR.StageDef

  defmodule ExampleAction do
    def run(_params, _context), do: {:ok, %{ok: true}}
  end

  defmodule ExampleStage do
    @behaviour Crucible.Stage

    @impl true
    def run(ctx, _opts), do: {:ok, ctx}

    @impl true
    def describe(_opts) do
      %{
        name: :example,
        description: "Example stage",
        required: [],
        optional: [],
        types: %{}
      }
    end
  end

  defp plan_fixture do
    %{
      id: "plan-1",
      context: %{tenant_id: "tenant-1"},
      steps: %{
        fetch: %{
          id: "step-fetch",
          name: :fetch,
          instruction: %{
            action: ExampleAction,
            params: %{url: "https://example.test"},
            context: %{source: "api"},
            opts: [timeout: 5000]
          },
          depends_on: []
        },
        transform: %{
          id: "step-transform",
          name: :transform,
          instruction: %{
            action: ExampleAction,
            params: %{format: "json"},
            context: %{},
            opts: []
          },
          depends_on: [:fetch]
        },
        save: %{
          id: "step-save",
          name: :save,
          instruction: %{
            action: ExampleAction,
            params: %{},
            context: %{},
            opts: []
          },
          depends_on: [:transform]
        }
      }
    }
  end

  test "compiles a plan into ordered StageDefs" do
    {:ok, stage_defs} = PlanAdapter.to_stage_defs(plan_fixture())

    assert Enum.map(stage_defs, & &1.name) == [:fetch, :transform, :save]
    assert Enum.all?(stage_defs, &match?(%StageDef{}, &1))
    assert Enum.all?(stage_defs, &(&1.module == Crucible.Stage.PlanStep))
  end

  test "propagates plan metadata into stage options" do
    {:ok, [first | _]} = PlanAdapter.to_stage_defs(plan_fixture())

    assert first.options.plan_id == "plan-1"
    assert first.options.step_id == "step-fetch"
    assert first.options.step_name == :fetch
    assert first.options.depends_on == []
    assert first.options.plan_context == %{tenant_id: "tenant-1"}
    assert first.options.action == ExampleAction
    assert first.options.params == %{url: "https://example.test"}
    assert first.options.context == %{source: "api"}
    assert first.options.exec_opts == [timeout: 5000]
  end

  test "errors on missing dependencies" do
    plan = plan_fixture()

    steps =
      Map.put(plan.steps, :broken, %{
        id: "step-broken",
        name: :broken,
        instruction: %{action: ExampleAction, params: %{}, context: %{}, opts: []},
        depends_on: [:missing]
      })

    plan = %{plan | steps: steps}

    assert {:error, {:missing_dependency, :missing, :broken}} = PlanAdapter.to_stage_defs(plan)
  end

  test "errors on cycles" do
    plan = plan_fixture()
    steps = Map.update!(plan.steps, :fetch, &Map.put(&1, :depends_on, [:save]))
    plan = %{plan | steps: steps}

    assert {:error, {:cycle, cycle}} = PlanAdapter.to_stage_defs(plan)
    assert Enum.sort(cycle) == Enum.sort([:fetch, :transform, :save])
  end

  test "allows custom stage module" do
    {:ok, [first | _]} = PlanAdapter.to_stage_defs(plan_fixture(), stage_module: ExampleStage)

    assert first.module == ExampleStage
  end
end
