defmodule Crucible.Telemetry.HandlersTest do
  use ExUnit.Case, async: false

  alias Crucible.Telemetry.{Handlers, MLMetrics}

  setup do
    # Start fresh collector for each test
    {:ok, collector} = MLMetrics.start_link(experiment_id: "handler-test")

    on_exit(fn ->
      Handlers.detach_handlers()
    end)

    %{collector: collector}
  end

  describe "attach_handlers/1" do
    test "attaches to all Crucible events" do
      :ok = Handlers.attach_handlers()

      events = Handlers.all_events()
      assert length(events) > 0

      # Verify handlers are attached
      handlers = :telemetry.list_handlers([:crucible, :training, :step])
      assert length(handlers) > 0
    end

    test "routes to appropriate collector", %{collector: collector} do
      :ok = Handlers.attach_handlers()

      # Emit a training event
      :telemetry.execute(
        [:crucible, :training, :step],
        %{loss: 2.0, grad_norm: 0.1, duration: 100},
        %{experiment_id: "handler-test", step: 1, epoch: 1}
      )

      # Give time for async processing
      Process.sleep(50)

      # Verify it was recorded - the collector is registered for "handler-test"
      metrics = MLMetrics.get_metrics(collector)
      # Should have at least 1 metric if routing works
      assert length(metrics) >= 1
    end

    test "accepts custom config" do
      config = %{
        collector: self(),
        filters: [:training]
      }

      :ok = Handlers.attach_handlers(config)

      # Should only handle training events
      handlers = :telemetry.list_handlers([:crucible, :training, :step])
      assert length(handlers) > 0
    end
  end

  describe "detach_handlers/0" do
    test "removes all handlers" do
      :ok = Handlers.attach_handlers()
      :ok = Handlers.detach_handlers()

      handlers = :telemetry.list_handlers([:crucible, :training, :step])

      crucible_handlers =
        Enum.filter(handlers, fn h ->
          String.starts_with?(to_string(h.id), "crucible-")
        end)

      assert crucible_handlers == []
    end
  end

  describe "handle_training_event/4" do
    test "processes training start/stop events" do
      test_pid = self()

      config = %{
        callback: fn event, measurements, metadata ->
          send(test_pid, {:event, event, measurements, metadata})
        end
      }

      :ok = Handlers.attach_handlers(config)

      # Emit start event
      :telemetry.execute(
        [:crucible, :training, :start],
        %{system_time: System.system_time()},
        %{experiment_id: "test", config: %{}}
      )

      assert_receive {:event, [:crucible, :training, :start], _, _}, 100
    end
  end

  describe "handle_inference_event/4" do
    test "processes inference events" do
      test_pid = self()

      config = %{
        callback: fn event, measurements, metadata ->
          send(test_pid, {:event, event, measurements, metadata})
        end
      }

      :ok = Handlers.attach_handlers(config)

      :telemetry.execute(
        [:crucible, :inference, :stop],
        %{duration: 150_000_000, tokens: 100},
        %{experiment_id: "test", model: "model-a"}
      )

      assert_receive {:event, [:crucible, :inference, :stop], measurements, _}, 100
      assert measurements.duration == 150_000_000
    end
  end

  describe "handle_ensemble_event/4" do
    test "processes ensemble voting events" do
      test_pid = self()

      config = %{
        callback: fn event, measurements, metadata ->
          send(test_pid, {:event, event, measurements, metadata})
        end
      }

      :ok = Handlers.attach_handlers(config)

      :telemetry.execute(
        [:crucible, :ensemble, :vote],
        %{duration: 200_000_000, model_count: 3},
        %{experiment_id: "test", strategy: :majority}
      )

      assert_receive {:event, [:crucible, :ensemble, :vote], _, metadata}, 100
      assert metadata.strategy == :majority
    end
  end

  describe "handle_hedging_event/4" do
    test "processes hedging dispatch/winner events" do
      test_pid = self()

      config = %{
        callback: fn event, measurements, metadata ->
          send(test_pid, {:event, event, measurements, metadata})
        end
      }

      :ok = Handlers.attach_handlers(config)

      :telemetry.execute(
        [:crucible, :hedging, :winner],
        %{duration: 100_000_000},
        %{experiment_id: "test", winner_index: 0, strategy: :percentile}
      )

      assert_receive {:event, [:crucible, :hedging, :winner], _, metadata}, 100
      assert metadata.winner_index == 0
    end
  end

  describe "handle_harness_event/4" do
    test "processes harness stage events" do
      test_pid = self()

      config = %{
        callback: fn event, measurements, metadata ->
          send(test_pid, {:event, event, measurements, metadata})
        end
      }

      :ok = Handlers.attach_handlers(config)

      :telemetry.execute(
        [:crucible, :harness, :stage_start],
        %{system_time: System.system_time()},
        %{experiment_id: "test", stage: :training}
      )

      assert_receive {:event, [:crucible, :harness, :stage_start], _, metadata}, 100
      assert metadata.stage == :training
    end
  end

  describe "all_events/0" do
    test "returns complete list of handled events" do
      events = Handlers.all_events()

      assert is_list(events)
      assert length(events) > 0

      # Check for key event categories
      event_prefixes = events |> Enum.map(&hd/1) |> Enum.uniq()
      assert :crucible in event_prefixes

      # Check specific events exist
      assert [:crucible, :training, :step] in events

      assert [:crucible, :inference, :start] in events or
               [:crucible, :inference, :stop] in events

      assert [:crucible, :ensemble, :vote] in events or
               [:crucible, :ensemble, :infer] in events
    end
  end
end
