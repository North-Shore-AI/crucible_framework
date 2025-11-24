defmodule Crucible.IR.ExperimentTest do
  use ExUnit.Case, async: true

  alias Crucible.IR.{
    BackendRef,
    DatasetRef,
    Experiment,
    ReliabilityConfig,
    StageDef
  }

  test "builds minimal experiment struct" do
    experiment = %Experiment{
      id: "demo",
      backend: %BackendRef{id: :tinkex},
      pipeline: [%StageDef{name: :data_load}]
    }

    assert experiment.reliability == %ReliabilityConfig{}
    assert experiment.tags == []
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
