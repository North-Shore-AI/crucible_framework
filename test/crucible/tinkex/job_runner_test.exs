defmodule Crucible.Tinkex.JobRunnerTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.Job
  alias Crucible.Tinkex.JobRunner
  alias Crucible.Tinkex.TelemetryBroker

  setup do
    Application.put_env(:crucible_tinkex, :runner_mode, :simulate)
    TelemetryBroker.ensure_started()
    :ok
  end

  test "writes manifest and emits simulated telemetry" do
    {:ok, job} =
      Job.new(%{
        "name" => "sim",
        "dataset_manifest" => "file://dataset",
        "hyperparams" => %{}
      })

    TelemetryBroker.subscribe(job.id)
    artifacts_path = job.artifacts_path
    File.rm_rf!(artifacts_path)

    assert :ok = JobRunner.submit(job)

    assert File.exists?(Path.join(artifacts_path, "manifest.json"))
    job_id = job.id
    assert_receive {:crucible_tinkex_event, ^job_id, %{event: :training_step}}, 200
  end
end
