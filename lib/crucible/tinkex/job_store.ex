defmodule Crucible.Tinkex.JobStore do
  @moduledoc """
  In-memory registry for Tinkex jobs submitted through the API layer.

  Backed by ETS to keep the scaffolding dependency-light while supporting
  concurrent reads/writes from the REST and WebSocket surfaces. This store
  persists job manifests and their latest status updates for UI consumption.
  """

  use GenServer

  alias Crucible.Tinkex.Job

  @table :crucible_tinkex_jobs

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @spec ensure_started() :: :ok | {:error, term()}
  def ensure_started do
    case Process.whereis(__MODULE__) do
      nil -> start_link([])
      _pid -> :ok
    end

    :ok
  end

  @doc """
  Stores or updates a job manifest.
  """
  @spec put(Job.t()) :: :ok
  def put(%Job{} = job) do
    ensure_started()
    true = :ets.insert(@table, {job.id, job})
    :ok
  end

  @doc """
  Fetches a job by ID.
  """
  @spec get(String.t()) :: {:ok, Job.t()} | {:error, :not_found}
  def get(job_id) do
    ensure_started()

    case :ets.lookup(@table, job_id) do
      [{^job_id, job}] -> {:ok, job}
      _ -> {:error, :not_found}
    end
  end

  @doc """
  Returns all jobs currently stored.
  """
  @spec all() :: [Job.t()]
  def all do
    ensure_started()
    @table |> :ets.tab2list() |> Enum.map(fn {_id, job} -> job end)
  end

  @doc """
  Updates a job status and writes it back.
  """
  @spec update_status(String.t(), Job.status(), keyword()) ::
          {:ok, Job.t()} | {:error, :not_found}
  def update_status(job_id, status, opts \\ []) do
    with {:ok, job} <- get(job_id) do
      updated = Job.with_status(job, status, opts)
      put(updated)
      {:ok, updated}
    end
  end

  @impl true
  def init(_opts) do
    table =
      :ets.new(@table, [
        :named_table,
        :public,
        :set,
        read_concurrency: true,
        write_concurrency: true
      ])

    {:ok, %{table: table}}
  end
end
