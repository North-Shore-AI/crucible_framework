defmodule Crucible.IR.ExperimentTest do
  use ExUnit.Case, async: true

  alias CrucibleIR.{
    BackendRef,
    DatasetRef,
    Experiment,
    StageDef
  }

  test "builds minimal experiment struct" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :tinkex},
      pipeline: [%StageDef{name: :data_load}]
    }

    # CrucibleIR.Experiment has nil defaults for optional fields
    assert experiment.reliability == nil
    assert experiment.tags == nil
    assert experiment.dataset == nil
  end

  test "supports dataset refs and metadata" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :nx},
      dataset: %DatasetRef{name: "scifact", split: :train},
      pipeline: [%StageDef{name: :data_load}],
      metadata: %{owner: "lab"}
    }

    assert experiment.dataset.name == "scifact"
    assert experiment.metadata[:owner] == "lab"
  end
end
