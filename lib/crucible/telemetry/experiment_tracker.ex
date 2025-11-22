defmodule Crucible.Telemetry.ExperimentTracker do
  @moduledoc """
  Tracks experiment runs, stages, and results.

  This module provides comprehensive experiment lifecycle tracking including
  start/end times, stage management, event timelines, and export capabilities.

  ## Usage

      {:ok, tracker} = ExperimentTracker.start_link([])

      # Start an experiment
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{name: "Test"})

      # Track stages
      :ok = ExperimentTracker.start_stage(tracker, "exp-1", :training)
      :ok = ExperimentTracker.end_stage(tracker, "exp-1", :training, %{steps: 100})

      # End experiment
      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :completed, %{accuracy: 0.95})

      # Export results
      :ok = ExperimentTracker.export_csv(tracker, "exp-1", "/path/to/export.csv")
  """

  use GenServer

  @type experiment_status :: :pending | :running | :completed | :failed

  defstruct [
    :experiments,
    :current_experiment,
    :storage
  ]

  # Client API

  @doc """
  Starts the experiment tracker.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    name = Keyword.get(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Starts tracking a new experiment.
  """
  @spec start_experiment(GenServer.server(), String.t(), map()) :: :ok
  def start_experiment(tracker, experiment_id, metadata) do
    GenServer.call(tracker, {:start_experiment, experiment_id, metadata})
  end

  @doc """
  Ends an experiment with final status and results.
  """
  @spec end_experiment(GenServer.server(), String.t(), experiment_status(), map()) :: :ok
  def end_experiment(tracker, experiment_id, status, results) do
    GenServer.call(tracker, {:end_experiment, experiment_id, status, results})
  end

  @doc """
  Starts a stage within an experiment.
  """
  @spec start_stage(GenServer.server(), String.t(), atom()) :: :ok
  def start_stage(tracker, experiment_id, stage_name) do
    GenServer.call(tracker, {:start_stage, experiment_id, stage_name})
  end

  @doc """
  Ends a stage with results.
  """
  @spec end_stage(GenServer.server(), String.t(), atom(), map()) :: :ok
  def end_stage(tracker, experiment_id, stage_name, results) do
    GenServer.call(tracker, {:end_stage, experiment_id, stage_name, results})
  end

  @doc """
  Adds an event to the experiment timeline.
  """
  @spec add_event(GenServer.server(), String.t(), map()) :: :ok
  def add_event(tracker, experiment_id, event) do
    GenServer.call(tracker, {:add_event, experiment_id, event})
  end

  @doc """
  Gets an experiment by ID.
  """
  @spec get_experiment(GenServer.server(), String.t()) :: map() | nil
  def get_experiment(tracker, experiment_id) do
    GenServer.call(tracker, {:get_experiment, experiment_id})
  end

  @doc """
  Lists all experiments, optionally filtered.

  ## Options

  - `:status` - Filter by experiment status
  """
  @spec list_experiments(GenServer.server(), keyword()) :: [map()]
  def list_experiments(tracker, opts \\ []) do
    GenServer.call(tracker, {:list_experiments, opts})
  end

  @doc """
  Gets the chronological event timeline for an experiment.
  """
  @spec get_timeline(GenServer.server(), String.t()) :: [map()]
  def get_timeline(tracker, experiment_id) do
    GenServer.call(tracker, {:get_timeline, experiment_id})
  end

  @doc """
  Saves tracker state to a file.
  """
  @spec save_to_file(GenServer.server(), String.t()) :: :ok
  def save_to_file(tracker, path) do
    GenServer.call(tracker, {:save_to_file, path})
  end

  @doc """
  Loads tracker state from a file.
  """
  @spec load_from_file(GenServer.server(), String.t()) :: :ok
  def load_from_file(tracker, path) do
    GenServer.call(tracker, {:load_from_file, path})
  end

  @doc """
  Exports experiment metrics to CSV format.
  """
  @spec export_csv(GenServer.server(), String.t(), String.t()) :: :ok
  def export_csv(tracker, experiment_id, path) do
    GenServer.call(tracker, {:export_csv, experiment_id, path})
  end

  @doc """
  Exports experiment metrics to JSON Lines format.
  """
  @spec export_jsonl(GenServer.server(), String.t(), String.t()) :: :ok
  def export_jsonl(tracker, experiment_id, path) do
    GenServer.call(tracker, {:export_jsonl, experiment_id, path})
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      experiments: %{},
      current_experiment: nil,
      storage: :memory
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:start_experiment, experiment_id, metadata}, _from, state) do
    experiment = %{
      id: experiment_id,
      metadata: metadata,
      status: :running,
      started_at: DateTime.utc_now(),
      ended_at: nil,
      results: nil,
      active_stages: [],
      completed_stages: [],
      timeline: [
        %{type: :experiment_start, timestamp: DateTime.utc_now()}
      ]
    }

    new_experiments = Map.put(state.experiments, experiment_id, experiment)
    {:reply, :ok, %{state | experiments: new_experiments, current_experiment: experiment_id}}
  end

  @impl true
  def handle_call({:end_experiment, experiment_id, status, results}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        ended_at = DateTime.utc_now()
        event = %{type: :experiment_end, status: status, timestamp: ended_at}

        updated = %{
          experiment
          | status: status,
            ended_at: ended_at,
            results: results,
            timeline: experiment.timeline ++ [event]
        }

        new_experiments = Map.put(state.experiments, experiment_id, updated)
        {:reply, :ok, %{state | experiments: new_experiments}}
    end
  end

  @impl true
  def handle_call({:start_stage, experiment_id, stage_name}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        event = %{type: :stage_start, name: stage_name, timestamp: DateTime.utc_now()}

        updated = %{
          experiment
          | active_stages: [stage_name | experiment.active_stages],
            timeline: experiment.timeline ++ [event]
        }

        new_experiments = Map.put(state.experiments, experiment_id, updated)
        {:reply, :ok, %{state | experiments: new_experiments}}
    end
  end

  @impl true
  def handle_call({:end_stage, experiment_id, stage_name, results}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        ended_at = DateTime.utc_now()
        event = %{type: :stage_end, name: stage_name, timestamp: ended_at}

        stage_record = %{
          name: stage_name,
          results: results,
          ended_at: ended_at
        }

        updated = %{
          experiment
          | active_stages: List.delete(experiment.active_stages, stage_name),
            completed_stages: [stage_record | experiment.completed_stages],
            timeline: experiment.timeline ++ [event]
        }

        new_experiments = Map.put(state.experiments, experiment_id, updated)
        {:reply, :ok, %{state | experiments: new_experiments}}
    end
  end

  @impl true
  def handle_call({:add_event, experiment_id, event}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        timestamped_event = Map.put_new(event, :timestamp, DateTime.utc_now())

        updated = %{experiment | timeline: experiment.timeline ++ [timestamped_event]}

        new_experiments = Map.put(state.experiments, experiment_id, updated)
        {:reply, :ok, %{state | experiments: new_experiments}}
    end
  end

  @impl true
  def handle_call({:get_experiment, experiment_id}, _from, state) do
    experiment = Map.get(state.experiments, experiment_id)
    {:reply, experiment, state}
  end

  @impl true
  def handle_call({:list_experiments, opts}, _from, state) do
    experiments =
      state.experiments
      |> Map.values()
      |> filter_by_status(opts[:status])
      |> Enum.sort_by(& &1.started_at, {:desc, DateTime})

    {:reply, experiments, state}
  end

  @impl true
  def handle_call({:get_timeline, experiment_id}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, [], state}

      experiment ->
        timeline = Enum.sort_by(experiment.timeline, & &1.timestamp, DateTime)
        {:reply, timeline, state}
    end
  end

  @impl true
  def handle_call({:save_to_file, path}, _from, state) do
    binary = :erlang.term_to_binary(state.experiments)
    File.write!(path, binary)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:load_from_file, path}, _from, state) do
    binary = File.read!(path)
    experiments = :erlang.binary_to_term(binary)
    {:reply, :ok, %{state | experiments: experiments}}
  end

  @impl true
  def handle_call({:export_csv, experiment_id, path}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        csv_content = build_csv(experiment)
        File.write!(path, csv_content)
        {:reply, :ok, state}
    end
  end

  @impl true
  def handle_call({:export_jsonl, experiment_id, path}, _from, state) do
    case Map.get(state.experiments, experiment_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      experiment ->
        jsonl_content = build_jsonl(experiment)
        File.write!(path, jsonl_content)
        {:reply, :ok, state}
    end
  end

  # Private helpers

  defp filter_by_status(experiments, nil), do: experiments

  defp filter_by_status(experiments, status) do
    Enum.filter(experiments, &(&1.status == status))
  end

  defp build_csv(experiment) do
    # Get metric events
    metric_events =
      experiment.timeline
      |> Enum.filter(&(&1.type == :metric))

    if metric_events == [] do
      # Export timeline instead
      headers = "timestamp,type,details\n"

      rows =
        experiment.timeline
        |> Enum.map(fn event ->
          ts = DateTime.to_iso8601(event.timestamp)
          type = event.type
          details = event |> Map.drop([:timestamp, :type]) |> inspect()
          "#{ts},#{type},\"#{details}\""
        end)
        |> Enum.join("\n")

      headers <> rows
    else
      # Get all keys from metrics
      all_keys =
        metric_events
        |> Enum.flat_map(&Map.keys/1)
        |> Enum.uniq()
        |> Enum.reject(&(&1 == :timestamp))

      headers = ["timestamp" | Enum.map(all_keys, &to_string/1)] |> Enum.join(",")

      rows =
        metric_events
        |> Enum.map(fn event ->
          ts = DateTime.to_iso8601(event.timestamp)

          values =
            Enum.map(all_keys, fn key ->
              case Map.get(event, key) do
                nil -> ""
                val -> to_string(val)
              end
            end)

          [ts | values] |> Enum.join(",")
        end)
        |> Enum.join("\n")

      headers <> "\n" <> rows
    end
  end

  defp build_jsonl(experiment) do
    experiment.timeline
    |> Enum.map(fn event ->
      event
      |> Map.update!(:timestamp, &DateTime.to_iso8601/1)
      |> Jason.encode!()
    end)
    |> Enum.join("\n")
  end
end
