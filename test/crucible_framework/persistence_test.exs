defmodule CrucibleFramework.PersistenceTest do
  use ExUnit.Case, async: false
  @moduletag :integration
  @repo_enabled Application.compile_env(:crucible_framework, :enable_repo, false)

  if not @repo_enabled do
    @moduletag skip: "CRUCIBLE_DB_ENABLED is not true; skipping persistence integration test"
  end

  alias CrucibleFramework.Persistence
  alias CrucibleFramework.Persistence.{ArtifactRecord, ExperimentRecord}
  alias CrucibleFramework.Repo
  alias CrucibleIR.{BackendRef, Experiment, StageDef}
  alias Ecto.Adapters.SQL.Sandbox

  setup do
    :ok = Sandbox.checkout(Repo)
    :ok
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
