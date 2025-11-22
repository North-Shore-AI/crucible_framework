defmodule Crucible.Tinkex.JobRunnerTest do
  use Supertester.ExUnitFoundation, isolation: :full_isolation

  import Supertester.OTPHelpers

  alias Crucible.Tinkex.Job
  alias Crucible.Tinkex.JobRunner
  alias Crucible.Tinkex.TelemetryBroker

  setup do
    Application.put_env(:crucible_framework, :runner_mode, :simulate)
    :ok = TelemetryBroker.ensure_started()
    :ok = wait_for_genserver_sync(TelemetryBroker)
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
