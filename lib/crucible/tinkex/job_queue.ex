defmodule Crucible.Tinkex.JobQueue do
  @moduledoc """
  Minimal queue and scheduler for Tinkex jobs submitted via the REST/WebSocket
  API surface.

  The queue enforces concurrency limits, persists job status via
  `Crucible.Tinkex.JobStore`, and delegates job execution to a configurable
  callback so the Tinkex SDK remains encapsulated inside the crucible
  application.
  """

  use GenServer

  alias Crucible.Tinkex.Job
  alias Crucible.Tinkex.JobStore
  alias Crucible.Tinkex.JobRunner
  alias Crucible.Tinkex.TelemetryBroker

  @type submit_fun :: (Job.t() -> :ok | {:error, term()})

  @default_concurrency 2

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @spec ensure_started(keyword()) :: :ok | {:error, term()}
  def ensure_started(opts \\ []) do
    case Process.whereis(__MODULE__) do
      nil -> start_link(opts)
      _pid -> :ok
    end

    :ok
  end

  @doc """
  Adds a job to the queue and returns the persisted manifest.
  """
  @spec enqueue(Job.t()) :: {:ok, Job.t()}
  def enqueue(%Job{} = job) do
    ensure_started()
    GenServer.call(__MODULE__, {:enqueue, job})
  end

  @doc """
  Requests cancellation of a job.
  """
  @spec cancel(String.t()) :: :ok | {:error, :not_found}
  def cancel(job_id) do
    ensure_started()
    GenServer.call(__MODULE__, {:cancel, job_id})
  end

  @impl true
  def init(opts) do
    JobStore.ensure_started()
    TelemetryBroker.ensure_started()

    state = %{
      queue: :queue.new(),
      running: %{},
      concurrency: opts[:concurrency] || @default_concurrency,
      submit_fun:
        opts[:submit_fun] ||
          Application.get_env(:crucible_tinkex, :job_submit_fun, &JobRunner.submit/1)
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:enqueue, job}, _from, state) do
    JobStore.put(Job.with_status(job, :queued))
    new_state = maybe_start_next(enqueue_job(job, state))
    {:reply, {:ok, job}, new_state}
  end

  def handle_call({:cancel, job_id}, _from, state) do
    case JobStore.update_status(job_id, :canceled) do
      {:ok, _job} ->
        TelemetryBroker.broadcast(job_id, %{
          event: :canceled,
          measurements: %{},
          metadata: %{job_id: job_id}
        })

        {:reply, :ok,
         %{
           state
           | queue: drop_job(job_id, state.queue),
             running: Map.delete(state.running, job_id)
         }}

      {:error, :not_found} ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_info({:job_finished, job_id, result}, state) do
    status = if match?({:ok, _}, result), do: :completed, else: :failed

    _ = JobStore.update_status(job_id, status, error: error_from(result))

    TelemetryBroker.broadcast(job_id, %{
      event: :completed,
      measurements: %{},
      metadata: %{job_id: job_id, status: status}
    })

    new_state = %{state | running: Map.delete(state.running, job_id)}
    {:noreply, maybe_start_next(new_state)}
  end

  defp enqueue_job(job, state) do
    %{state | queue: :queue.in(job, state.queue)}
  end

  defp maybe_start_next(state) do
    if map_size(state.running) < state.concurrency do
      case :queue.out(state.queue) do
        {{:value, job}, q2} ->
          start_job(job, %{state | queue: q2})

        {:empty, _} ->
          state
      end
    else
      state
    end
  end

  defp drop_job(job_id, queue) do
    queue
    |> :queue.to_list()
    |> Enum.reject(&(&1.id == job_id))
    |> :queue.from_list()
  end

  defp start_job(%Job{} = job, state) do
    running = Map.put(state.running, job.id, job)
    JobStore.update_status(job.id, :running)

    TelemetryBroker.broadcast(job.id, %{
      event: :started,
      measurements: %{},
      metadata: %{job_id: job.id}
    })

    # Delegate execution without surfacing credentials to callers.
    server = self()

    Task.start(fn ->
      result = state.submit_fun.(job)
      send(server, {:job_finished, job.id, result})
    end)

    %{state | running: running}
  end

  @doc false
  def noop_submit(_job), do: :ok

  defp error_from({:error, reason}), do: inspect(reason)
  defp error_from(_), do: nil
end
