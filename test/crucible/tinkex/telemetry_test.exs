defmodule Crucible.Tinkex.TelemetryTest do
  use ExUnit.Case, async: false

  alias Crucible.Tinkex.Telemetry

  def capture_handler(event, measurements, metadata, test_pid) do
    send(test_pid, {:event, event, measurements, metadata})
  end

  setup do
    # Ensure handlers are detached before each test
    Telemetry.detach()
    :ok
  end

  describe "attach/1" do
    test "attaches handlers successfully" do
      assert :ok = Telemetry.attach()
    end

    test "attaches with experiment context" do
      assert :ok = Telemetry.attach(experiment_id: "exp-123")
    end

    test "attaches with custom handler id" do
      assert :ok = Telemetry.attach(handler_id: "custom-handler")
      assert :ok = Telemetry.detach("custom-handler")
    end
  end

  describe "detach/1" do
    test "detaches handlers successfully" do
      :ok = Telemetry.attach()
      assert :ok = Telemetry.detach()
    end

    test "returns error for non-existent handler" do
      assert {:error, :not_found} = Telemetry.detach("non-existent")
    end
  end

  describe "emit_training_step/1" do
    test "emits training step event" do
      :ok = Telemetry.attach()

      assert :ok =
               Telemetry.emit_training_step(%{
                 step: 100,
                 epoch: 2,
                 loss: 0.5,
                 citation_invalid_rate: 0.0
               })
    end
  end

  describe "emit_epoch_complete/1" do
    test "emits epoch completion event" do
      :ok = Telemetry.attach()

      assert :ok =
               Telemetry.emit_epoch_complete(%{
                 epoch: 1,
                 mean_loss: 0.8,
                 steps: 64
               })
    end
  end

  describe "emit_evaluation_complete/1" do
    test "emits evaluation completion event" do
      :ok = Telemetry.attach()

      assert :ok =
               Telemetry.emit_evaluation_complete(%{
                 adapter_name: "v1",
                 samples: 50,
                 metrics: %{schema_compliance: 0.96}
               })
    end
  end

  describe "emit_checkpoint_saved/1" do
    test "emits checkpoint saved event" do
      :ok = Telemetry.attach()

      assert :ok =
               Telemetry.emit_checkpoint_saved(%{
                 name: "step_100",
                 step: 100
               })
    end
  end

  describe "events/0" do
    test "returns list of handled events" do
      events = Telemetry.events()

      assert is_list(events)
      assert length(events) == 5
      assert [:crucible, :tinkex, :training, :step] in events
    end
  end

  describe "event capture" do
    test "captures events with measurements" do
      test_pid = self()

      :telemetry.attach(
        "test-capture",
        [:crucible, :tinkex, :training, :step],
        &__MODULE__.capture_handler/4,
        test_pid
      )

      Telemetry.emit_training_step(%{step: 1, loss: 1.0, citation_invalid_rate: 0.1})

      assert_receive {:event, [:crucible, :tinkex, :training, :step], measurements, metadata}
      assert measurements.loss == 1.0
      assert measurements.citation_invalid_rate == 0.1
      assert metadata.step == 1

      :telemetry.detach("test-capture")
    end
  end
end
