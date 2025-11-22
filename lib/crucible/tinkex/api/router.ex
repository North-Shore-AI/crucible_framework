defmodule Crucible.Tinkex.API.Router do
  @moduledoc """
  REST/WebSocket scaffolding for the Crucible Tinkex overlay.

  This module defines controller-like functions that mirror the CNS blueprint
  endpoints:

    - POST /v1/jobs
    - GET  /v1/jobs/:id
    - GET  /v1/jobs/:id/stream (SSE/WebSocket)
    - POST /v1/jobs/:id/cancel

  The functions are framework-agnostic so they can be wired into Phoenix or
  any Plug-compatible router without pulling UI logic into the core library.
  """

  alias Crucible.Tinkex.API.Auth
  alias Crucible.Tinkex.Job
  alias Crucible.Tinkex.JobQueue
  alias Crucible.Tinkex.JobStore
  alias Crucible.Tinkex.TelemetryBroker

  @type request :: %{params: map(), headers: [{String.t(), String.t()}]}

  @doc """
  Handles job submission (POST /v1/jobs).
  """
  @spec submit(request()) ::
          {:ok, map()} | {:error, :unauthorized} | {:error, :invalid_request}
  def submit(%{params: params, headers: headers}) do
    with {:ok, actor} <- Auth.verify(headers),
         {:ok, job} <- Job.new(params),
         {:ok, _} <- JobQueue.enqueue(job) do
      token = TelemetryBroker.issue_stream_token(job.id, actor: actor)

      {:ok,
       %{
         job_id: job.id,
         status: :queued,
         stream_token: token,
         artifacts_path: job.artifacts_path
       }}
    else
      {:error, :unauthorized} = error -> error
      _ -> {:error, :invalid_request}
    end
  end

  @doc """
  Fetches job status and latest metadata (GET /v1/jobs/:id).
  """
  @spec fetch(request(), String.t()) :: {:ok, map()} | {:error, :unauthorized | :not_found}
  def fetch(%{headers: headers}, job_id) do
    with {:ok, _} <- Auth.verify(headers),
         {:ok, job} <- JobStore.get(job_id) do
      {:ok,
       %{
         job_id: job.id,
         status: job.status,
         spec: job.spec,
         artifacts_path: job.artifacts_path,
         inserted_at: job.inserted_at,
         updated_at: job.updated_at,
         error: job.error
       }}
    else
      {:error, :unauthorized} = error -> error
      {:error, :not_found} -> {:error, :not_found}
    end
  end

  @doc """
  Issues a stream subscription for telemetry (GET /v1/jobs/:id/stream).

  Returns a function that can be used by WebSocket/SSE handlers to receive
  messages via mailbox or enumerator.
  """
  @spec stream(request(), String.t()) ::
          {:ok, %{subscribe: (-> :ok)}} | {:error, :unauthorized | :not_found}
  def stream(%{headers: headers}, job_id) do
    with {:ok, _actor} <- Auth.verify(headers),
         {:ok, _job} <- JobStore.get(job_id),
         {:ok, token} <- extract_stream_token(headers),
         :ok <- TelemetryBroker.verify_stream_token(token, job_id) do
      {:ok, %{subscribe: fn -> TelemetryBroker.subscribe(job_id) end}}
    else
      {:error, :unauthorized} = error -> error
      {:error, :not_found} -> {:error, :not_found}
      _ -> {:error, :unauthorized}
    end
  end

  @doc """
  Cancels a running job (POST /v1/jobs/:id/cancel).
  """
  @spec cancel(request(), String.t()) :: :ok | {:error, :unauthorized | :not_found}
  def cancel(%{headers: headers}, job_id) do
    with {:ok, _} <- Auth.verify(headers),
         :ok <- JobQueue.cancel(job_id) do
      :ok
    else
      {:error, :unauthorized} = error -> error
      {:error, :not_found} -> {:error, :not_found}
    end
  end

  defp extract_stream_token(headers) do
    case Enum.find(headers, fn {k, _} -> String.downcase(k) == "x-stream-token" end) do
      {_, token} when byte_size(token) > 0 -> {:ok, token}
      _ -> {:error, :missing}
    end
  end
end
