defmodule Crucible.Thinker.TelemetryTest do
  use ExUnit.Case, async: false

  alias Crucible.Thinker.Telemetry

  describe "events/0" do
    test "returns list of event names" do
      events = Telemetry.events()

      assert is_list(events)
      assert length(events) > 0
      assert Enum.all?(events, &is_list/1)
    end

    test "includes harness events" do
      events = Telemetry.events()

      assert [:crucible, :thinker, :harness, :start] in events
      assert [:crucible, :thinker, :harness, :complete] in events
    end

    test "includes validation events" do
      events = Telemetry.events()

      assert [:crucible, :thinker, :validation, :complete] in events
    end

    test "includes antagonist events" do
      events = Telemetry.events()

      assert [:crucible, :thinker, :antagonist, :complete] in events
    end
  end

  describe "attach/1 and detach/0" do
    test "attaches and detaches handlers" do
      assert :ok = Telemetry.attach()
      assert :ok = Telemetry.detach()
    end

    test "can attach with custom log level" do
      assert :ok = Telemetry.attach(log_level: :info)
      assert :ok = Telemetry.detach()
    end
  end

  describe "handle_event/4" do
    test "handles harness start event" do
      config = %{log_level: :debug, forward_to_research: false}

      # Should not raise
      Telemetry.handle_event(
        [:crucible, :thinker, :harness, :start],
        %{timestamp: 0},
        %{experiment_id: "test", name: "test-exp"},
        config
      )
    end

    test "handles validation complete event" do
      config = %{log_level: :debug, forward_to_research: false}

      Telemetry.handle_event(
        [:crucible, :thinker, :validation, :complete],
        %{
          schema_compliance: 1.0,
          citation_accuracy: 1.0,
          mean_entailment: 0.8,
          mean_similarity: 0.9
        },
        %{claim_count: 5},
        config
      )
    end
  end
end
