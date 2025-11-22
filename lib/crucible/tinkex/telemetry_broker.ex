defmodule Crucible.Tinkex.TelemetryBroker do
  @moduledoc """
  Broadcasts Crucible Tinkex telemetry events to in-process subscribers.

  This broker is the bridge for the CNS-facing WebSocket/SSE surfaces.
  It mirrors `:telemetry` events without requiring UI clients to attach
  directly to BEAM instrumentation, enabling the single-owner credential
  boundary to hold while CNS surfaces remain consumers only.
  """

  use GenServer

  @registry __MODULE__.Registry

  @type job_id :: String.t()
  @type event :: %{event: atom(), measurements: map(), metadata: map()}

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @spec ensure_started() :: :ok | {:error, term()}
  def ensure_started do
    case Process.whereis(__MODULE__) do
      nil -> GenServer.start(__MODULE__, [], name: __MODULE__)
      _ -> :ok
    end

    :ok
  end

  @doc """
  Subscribes the calling process to telemetry events for a job.
  """
  @spec subscribe(job_id()) :: :ok
  def subscribe(job_id) do
    ensure_started()
    {:ok, _} = Registry.register(@registry, job_id, %{})
    :ok
  end

  @doc """
  Issues a short-lived stream token for WebSocket/SSE clients.
  """
  @spec issue_stream_token(job_id(), keyword()) :: String.t()
  def issue_stream_token(job_id, opts \\ []) do
    actor = Keyword.get(opts, :actor, "system")
    Base.encode64("#{job_id}:#{actor}:#{System.system_time()}")
  end

  @doc """
  Verifies that a stream token belongs to the provided job.
  """
  @spec verify_stream_token(String.t(), job_id()) :: :ok | {:error, :unauthorized}
  def verify_stream_token(token, job_id) do
    with {:ok, decoded} <- Base.decode64(token),
         [decoded_job_id | _rest] <- String.split(decoded, ":"),
         true <- decoded_job_id == job_id do
      :ok
    else
      _ -> {:error, :unauthorized}
    end
  end

  @doc """
  Broadcasts a telemetry payload to all subscribers of a job.
  """
  @spec broadcast(job_id(), event()) :: :ok
  def broadcast(job_id, event) do
    ensure_started()

    @registry
    |> Registry.lookup(job_id)
    |> Enum.each(fn {pid, _} ->
      send(pid, {:crucible_tinkex_event, job_id, event})
    end)

    :ok
  end

  @impl true
  def handle_call(:__supertester_sync__, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def init(_opts) do
    Registry.start_link(keys: :duplicate, name: @registry)
    {:ok, %{}}
  end
end
