defmodule Crucible.Telemetry.ExperimentTrackerTest do
  use ExUnit.Case, async: true

  alias Crucible.Telemetry.ExperimentTracker

  setup do
    {:ok, tracker} = ExperimentTracker.start_link([])
    %{tracker: tracker}
  end

  describe "start/end_experiment" do
    test "tracks experiment lifecycle", %{tracker: tracker} do
      metadata = %{name: "Test Experiment", model: "llama-3"}
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", metadata)

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert experiment.id == "exp-1"
      assert experiment.status == :running
      assert experiment.metadata.name == "Test Experiment"

      results = %{final_loss: 0.5, accuracy: 0.95}
      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :completed, results)

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert experiment.status == :completed
      assert experiment.results.final_loss == 0.5
    end

    test "records timestamps", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert %DateTime{} = experiment.started_at

      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :completed, %{})

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert %DateTime{} = experiment.ended_at
      assert DateTime.compare(experiment.ended_at, experiment.started_at) in [:gt, :eq]
    end

    test "handles failed experiments", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :failed, %{error: "OOM"})

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert experiment.status == :failed
      assert experiment.results.error == "OOM"
    end
  end

  describe "start/end_stage" do
    test "tracks stage within experiment", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.start_stage(tracker, "exp-1", :training)

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert :training in experiment.active_stages

      :ok = ExperimentTracker.end_stage(tracker, "exp-1", :training, %{steps: 100})

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      refute :training in experiment.active_stages
      assert hd(experiment.completed_stages).name == :training
      assert hd(experiment.completed_stages).results.steps == 100
    end

    test "tracks multiple stages", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.start_stage(tracker, "exp-1", :data_prep)
      :ok = ExperimentTracker.end_stage(tracker, "exp-1", :data_prep, %{rows: 1000})
      :ok = ExperimentTracker.start_stage(tracker, "exp-1", :training)
      :ok = ExperimentTracker.end_stage(tracker, "exp-1", :training, %{steps: 500})

      experiment = ExperimentTracker.get_experiment(tracker, "exp-1")
      assert length(experiment.completed_stages) == 2
    end
  end

  describe "add_event/3" do
    test "adds events to experiment timeline", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      event = %{type: :checkpoint, step: 100, path: "/tmp/ckpt"}
      :ok = ExperimentTracker.add_event(tracker, "exp-1", event)

      timeline = ExperimentTracker.get_timeline(tracker, "exp-1")
      assert length(timeline) >= 1
      checkpoint_events = Enum.filter(timeline, &(&1.type == :checkpoint))
      assert length(checkpoint_events) == 1
    end
  end

  describe "list_experiments/2" do
    test "lists all experiments", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.start_experiment(tracker, "exp-2", %{})

      experiments = ExperimentTracker.list_experiments(tracker)
      assert length(experiments) == 2
    end

    test "filters by status", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.start_experiment(tracker, "exp-2", %{})
      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :completed, %{})

      running = ExperimentTracker.list_experiments(tracker, status: :running)
      assert length(running) == 1
      assert hd(running).id == "exp-2"
    end
  end

  describe "get_timeline/2" do
    test "returns chronological event timeline", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.start_stage(tracker, "exp-1", :training)
      :ok = ExperimentTracker.add_event(tracker, "exp-1", %{type: :metric, value: 1.0})
      :ok = ExperimentTracker.end_stage(tracker, "exp-1", :training, %{})

      timeline = ExperimentTracker.get_timeline(tracker, "exp-1")
      assert length(timeline) >= 3

      # Verify chronological order
      timestamps = Enum.map(timeline, & &1.timestamp)
      assert timestamps == Enum.sort(timestamps, DateTime)
    end
  end

  describe "export_csv/3" do
    test "exports metrics to CSV", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.add_event(tracker, "exp-1", %{type: :metric, step: 1, loss: 2.0})
      :ok = ExperimentTracker.add_event(tracker, "exp-1", %{type: :metric, step: 2, loss: 1.5})

      path = "/tmp/test_export_#{System.unique_integer([:positive])}.csv"
      :ok = ExperimentTracker.export_csv(tracker, "exp-1", path)

      assert File.exists?(path)
      content = File.read!(path)
      assert String.contains?(content, "step")
      assert String.contains?(content, "loss")
      File.rm!(path)
    end
  end

  describe "export_jsonl/3" do
    test "exports metrics to JSONL", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{})
      :ok = ExperimentTracker.add_event(tracker, "exp-1", %{type: :metric, step: 1, loss: 2.0})

      path = "/tmp/test_export_#{System.unique_integer([:positive])}.jsonl"
      :ok = ExperimentTracker.export_jsonl(tracker, "exp-1", path)

      assert File.exists?(path)
      content = File.read!(path)
      assert String.contains?(content, "loss")
      File.rm!(path)
    end
  end

  describe "save_to_file/2 and load_from_file/2" do
    test "persists and restores tracker state", %{tracker: tracker} do
      :ok = ExperimentTracker.start_experiment(tracker, "exp-1", %{name: "Test"})
      :ok = ExperimentTracker.end_experiment(tracker, "exp-1", :completed, %{})

      path = "/tmp/tracker_state_#{System.unique_integer([:positive])}.bin"
      :ok = ExperimentTracker.save_to_file(tracker, path)

      {:ok, new_tracker} = ExperimentTracker.start_link([])
      :ok = ExperimentTracker.load_from_file(new_tracker, path)

      experiment = ExperimentTracker.get_experiment(new_tracker, "exp-1")
      assert experiment.metadata.name == "Test"
      assert experiment.status == :completed

      File.rm!(path)
    end
  end
end
