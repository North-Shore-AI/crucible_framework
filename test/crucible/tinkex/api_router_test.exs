defmodule Crucible.Tinkex.APIRouterTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.API.Router
  alias Crucible.Tinkex.TelemetryBroker

  setup do
    Application.put_env(:crucible_tinkex, :api_tokens, ["test-token"])

    Application.put_env(
      :crucible_tinkex,
      :job_submit_fun,
      &Crucible.Tinkex.JobQueue.noop_submit/1
    )

    :ok
  end

  test "submits and fetches a job manifest" do
    headers = [{"authorization", "Bearer test-token"}]

    {:ok, resp} =
      Router.submit(%{
        headers: headers,
        params: %{
          "name" => "cns-run",
          "dataset_manifest" => "s3://bucket/dataset.jsonl",
          "hyperparams" => %{"lr" => 0.0002}
        }
      })

    assert resp.status == :queued
    assert is_binary(resp.job_id)
    assert is_binary(resp.stream_token)

    {:ok, job} = Router.fetch(%{headers: headers}, resp.job_id)
    assert job.job_id == resp.job_id
    assert job.spec.dataset_manifest == "s3://bucket/dataset.jsonl"
  end

  test "streams telemetry with a valid stream token" do
    headers = [{"authorization", "Bearer test-token"}]

    {:ok, resp} =
      Router.submit(%{
        headers: headers,
        params: %{"dataset_manifest" => "local"}
      })

    stream_headers = headers ++ [{"x-stream-token", resp.stream_token}]
    {:ok, stream} = Router.stream(%{headers: stream_headers}, resp.job_id)

    stream.subscribe.()
    job_id = resp.job_id

    TelemetryBroker.broadcast(job_id, %{
      event: :training_step,
      measurements: %{loss: 1.0},
      metadata: %{step: 1}
    })

    assert_receive {:crucible_tinkex_event, ^job_id, %{event: :training_step}}, 200
  end

  test "rejects unauthorized requests" do
    assert {:error, :unauthorized} =
             Router.submit(%{headers: [], params: %{"dataset_manifest" => "d"}})
  end
end
