defmodule Crucible.Telemetry.Research do
  @moduledoc """
  Research-grade event capture for experiment tracking.

  Stores telemetry events for later analysis, export, and visualization.
  Events are stored in ETS for fast access during experiments.
  """

  use GenServer

  @table_name :crucible_research_events

  # Client API

  @doc """
  Starts the Research event store.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Captures a telemetry event for research storage.

  ## Example

      Crucible.Telemetry.Research.capture(%{
        event: [:crucible, :thinker, :harness, :complete],
        measurements: %{duration: 1500},
        metadata: %{experiment_id: "abc123"},
        timestamp: DateTime.utc_now()
      })

  """
  @spec capture(map()) :: :ok
  def capture(event_data) do
    ensure_started()

    entry = %{
      id: generate_id(),
      event: event_data.event,
      measurements: event_data.measurements,
      metadata: event_data.metadata,
      timestamp: event_data[:timestamp] || DateTime.utc_now(),
      captured_at: System.monotonic_time(:millisecond)
    }

    :ets.insert(@table_name, {entry.id, entry})
    :ok
  end

  @doc """
  Retrieves all captured events.
  """
  @spec all() :: [map()]
  def all do
    ensure_started()

    @table_name
    |> :ets.tab2list()
    |> Enum.map(fn {_id, event} -> event end)
    |> Enum.sort_by(& &1.captured_at)
  end

  @doc """
  Retrieves events for a specific experiment.
  """
  @spec for_experiment(String.t()) :: [map()]
  def for_experiment(experiment_id) do
    all()
    |> Enum.filter(fn event ->
      get_in(event, [:metadata, :experiment_id]) == experiment_id
    end)
  end

  @doc """
  Retrieves events matching a specific event name pattern.
  """
  @spec by_event([atom()]) :: [map()]
  def by_event(event_name) do
    all()
    |> Enum.filter(fn event -> event.event == event_name end)
  end

  @doc """
  Clears all captured events.
  """
  @spec clear() :: :ok
  def clear do
    ensure_started()
    :ets.delete_all_objects(@table_name)
    :ok
  end

  @doc """
  Returns the count of captured events.
  """
  @spec count() :: non_neg_integer()
  def count do
    ensure_started()
    :ets.info(@table_name, :size)
  end

  @doc """
  Exports events to JSON Lines format.
  """
  @spec export_jsonl(String.t()) :: :ok | {:error, term()}
  def export_jsonl(path) do
    events = all()

    content =
      events
      |> Enum.map(&Jason.encode!/1)
      |> Enum.join("\n")

    File.write(path, content)
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    table = :ets.new(@table_name, [:set, :public, :named_table])
    {:ok, %{table: table}}
  end

  # Private functions

  defp ensure_started do
    case :ets.whereis(@table_name) do
      :undefined ->
        # Table doesn't exist, try to start the GenServer
        case GenServer.whereis(__MODULE__) do
          nil ->
            # Start under a simple supervisor or directly
            :ets.new(@table_name, [:set, :public, :named_table])

          _pid ->
            :ok
        end

      _tid ->
        :ok
    end
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
