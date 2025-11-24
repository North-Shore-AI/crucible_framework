defmodule CrucibleFramework.PersistenceTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Crucible.IR.{BackendRef, Experiment, StageDef}
  alias CrucibleFramework.Persistence
  alias CrucibleFramework.Persistence.{ArtifactRecord, ExperimentRecord}
  alias CrucibleFramework.Repo

  setup do
    if Application.get_env(:crucible_framework, :enable_repo, false) do
      :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
      :ok
    else
      {:skip, "CRUCIBLE_DB_ENABLED not set; skipping persistence integration test"}
    end
  end

  test "upserts experiment and creates run records" do
    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :mock},
      pipeline: [%StageDef{name: :data_load}]
    }

    assert {:ok, run} = Persistence.start_run(experiment)
    assert run.experiment_id == "exp"
    assert Repo.aggregate(ExperimentRecord, :count, :id) == 1

    assert {:ok, finished} = Persistence.finish_run(run, "completed", %{metrics: %{loss: 0.1}})
    assert finished.status == "completed"
    assert finished.metrics["loss"] == 0.1
  end

  test "records artifacts" do
    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :mock},
      pipeline: [%StageDef{name: :data_load}]
    }

    {:ok, run} = Persistence.start_run(experiment)

    {:ok, artifact} =
      Persistence.record_artifact(run, %{
        name: :report,
        type: "file",
        location: "/tmp/report.md",
        format: "markdown"
      })

    assert %ArtifactRecord{} = artifact
    assert Repo.aggregate(ArtifactRecord, :count, :id) == 1
  end
end
