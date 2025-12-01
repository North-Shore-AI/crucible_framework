defmodule Crucible.TraceIntegrationTest do
  use ExUnit.Case, async: true

  alias Crucible.{Context, TraceIntegration}
  alias CrucibleIR.Experiment
  alias CrucibleTrace

  describe "init_trace/2" do
    test "initializes a new trace chain in the context" do
      ctx = %Context{
        experiment_id: "exp-123",
        run_id: "run-456",
        experiment: %Experiment{id: "exp-123", backend: nil, pipeline: []},
        trace: nil
      }

      new_ctx = TraceIntegration.init_trace(ctx, "Test Experiment")

      assert new_ctx.trace != nil
      assert new_ctx.trace.name == "Test Experiment"
      assert new_ctx.trace.metadata.experiment_id == "exp-123"
      assert new_ctx.trace.metadata.run_id == "run-456"
    end
  end

  describe "emit_event/5" do
    setup do
      ctx = %Context{
        experiment_id: "exp-123",
        run_id: "run-456",
        experiment: %Experiment{id: "exp-123", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      {:ok, ctx: ctx}
    end

    test "adds an event to the trace chain", %{ctx: ctx} do
      initial_event_count = length(ctx.trace.events)

      new_ctx =
        TraceIntegration.emit_event(
          ctx,
          :hypothesis_formed,
          "Use caching for performance",
          "Repeated computations detected",
          alternatives: ["Direct computation", "Memoization"],
          confidence: 0.8
        )

      assert length(new_ctx.trace.events) == initial_event_count + 1

      last_event = List.last(new_ctx.trace.events)
      assert last_event.type == :hypothesis_formed
      assert last_event.decision == "Use caching for performance"
      assert last_event.reasoning == "Repeated computations detected"
      assert last_event.alternatives == ["Direct computation", "Memoization"]
      assert last_event.confidence == 0.8
    end

    test "returns context unchanged when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      new_ctx =
        TraceIntegration.emit_event(
          ctx,
          :decision_made,
          "Decision",
          "Reasoning"
        )

      assert new_ctx == ctx
    end
  end

  describe "emit_stage_start/3" do
    test "emits a stage start event" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      new_ctx = TraceIntegration.emit_stage_start(ctx, :data_load, %{source: "file"})

      last_event = List.last(new_ctx.trace.events)
      assert last_event.type == :stage_start
      assert last_event.decision =~ "data_load"
      assert last_event.metadata.stage == :data_load
      assert last_event.metadata.options == %{source: "file"}
    end
  end

  describe "emit_stage_complete/3" do
    test "emits a stage completion event" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      results = %{processed: 100, errors: 0}
      new_ctx = TraceIntegration.emit_stage_complete(ctx, :data_load, results)

      last_event = List.last(new_ctx.trace.events)
      assert last_event.type == :stage_complete
      assert last_event.decision =~ "data_load"
      assert last_event.metadata.stage == :data_load
      assert last_event.metadata.results == results
    end
  end

  describe "emit_stage_failed/3" do
    test "emits a stage failure event" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      error = {:file_not_found, "data.csv"}
      new_ctx = TraceIntegration.emit_stage_failed(ctx, :data_load, error)

      last_event = List.last(new_ctx.trace.events)
      assert last_event.type == :stage_failed
      assert last_event.decision =~ "data_load"
      assert last_event.reasoning =~ "file_not_found"
      assert last_event.confidence == 0.0
    end
  end

  describe "emit_decision/5" do
    test "emits a decision event with alternatives" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      new_ctx =
        TraceIntegration.emit_decision(
          ctx,
          "Use GenServer for state",
          "Need fault tolerance and supervision",
          ["Agent", "ETS", "Process dictionary"],
          0.9
        )

      last_event = List.last(new_ctx.trace.events)
      assert last_event.type == :decision_made
      assert last_event.decision == "Use GenServer for state"
      assert last_event.alternatives == ["Agent", "ETS", "Process dictionary"]
      assert last_event.confidence == 0.9
    end
  end

  describe "export_json/1" do
    test "exports trace chain to JSON" do
      chain = CrucibleTrace.new_chain("test")

      event =
        CrucibleTrace.create_event(
          :hypothesis_formed,
          "Test decision",
          "Test reasoning"
        )

      chain = CrucibleTrace.add_event(chain, event)

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      json = TraceIntegration.export_json(ctx)
      assert is_binary(json)

      # Parse to verify it's valid JSON
      assert {:ok, decoded} = Jason.decode(json)
      assert decoded["name"] == "test"
      assert length(decoded["events"]) == 1
    end

    test "returns nil when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.export_json(ctx) == nil
    end
  end

  describe "export_html/1" do
    test "generates HTML visualization" do
      chain = CrucibleTrace.new_chain("test")

      event =
        CrucibleTrace.create_event(
          :pattern_applied,
          "Apply supervisor pattern",
          "Need fault tolerance"
        )

      chain = CrucibleTrace.add_event(chain, event)

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      html = TraceIntegration.export_html(ctx)
      assert is_binary(html)
      assert html =~ "<html"
      assert html =~ "test"
    end

    test "returns nil when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.export_html(ctx) == nil
    end
  end

  describe "tracing_enabled?/1" do
    test "returns true when trace is present" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: CrucibleTrace.new_chain("test")
      }

      assert TraceIntegration.tracing_enabled?(ctx) == true
    end

    test "returns false when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.tracing_enabled?(ctx) == false
    end
  end

  describe "event_count/1" do
    test "returns the number of events in the trace" do
      chain = CrucibleTrace.new_chain("test")

      chain =
        chain
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:hypothesis_formed, "D1", "R1"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:pattern_applied, "D2", "R2"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:decision_made, "D3", "R3"))

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      assert TraceIntegration.event_count(ctx) == 3
    end

    test "returns 0 when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.event_count(ctx) == 0
    end
  end

  describe "filter_events/2" do
    test "filters events by type" do
      chain = CrucibleTrace.new_chain("test")

      chain =
        chain
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:hypothesis_formed, "H1", "R1"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:pattern_applied, "P1", "R2"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:hypothesis_formed, "H2", "R3"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:decision_made, "D1", "R4"))

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      hypotheses = TraceIntegration.filter_events(ctx, :hypothesis_formed)
      assert length(hypotheses) == 2
      assert Enum.all?(hypotheses, fn e -> e.type == :hypothesis_formed end)
    end

    test "returns empty list when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.filter_events(ctx, :any_type) == []
    end
  end

  describe "last_event/1" do
    test "returns the most recent event" do
      chain = CrucibleTrace.new_chain("test")

      chain =
        chain
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:hypothesis_formed, "First", "R1"))
        |> CrucibleTrace.add_event(CrucibleTrace.create_event(:decision_made, "Last", "R2"))

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      last = TraceIntegration.last_event(ctx)
      assert last.type == :decision_made
      assert last.decision == "Last"
    end

    test "returns nil when trace is nil or empty" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.last_event(ctx) == nil
    end
  end

  describe "confidence_stats/1" do
    test "calculates confidence statistics" do
      chain = CrucibleTrace.new_chain("test")

      chain =
        chain
        |> CrucibleTrace.add_event(
          CrucibleTrace.create_event(:hypothesis_formed, "H1", "R1", confidence: 0.8)
        )
        |> CrucibleTrace.add_event(
          CrucibleTrace.create_event(:pattern_applied, "P1", "R2", confidence: 0.9)
        )
        |> CrucibleTrace.add_event(
          CrucibleTrace.create_event(:decision_made, "D1", "R3", confidence: 0.7)
        )

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      stats = TraceIntegration.confidence_stats(ctx)
      assert stats.count == 3
      assert stats.mean == 0.8
      assert stats.min == 0.7
      assert stats.max == 0.9
    end

    test "returns empty stats when no confidence values" do
      chain = CrucibleTrace.new_chain("test")

      chain =
        CrucibleTrace.add_event(
          chain,
          CrucibleTrace.create_event(:hypothesis_formed, "H1", "R1")
          # No confidence specified
        )

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      stats = TraceIntegration.confidence_stats(ctx)
      assert stats.count == 1
      assert stats.mean == 1.0
    end

    test "returns empty map when trace is nil" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert TraceIntegration.confidence_stats(ctx) == %{}
    end
  end

  describe "save_trace/2 and load_trace/1" do
    @tag :tmp_dir
    test "saves and loads trace from disk", %{tmp_dir: tmp_dir} do
      chain = CrucibleTrace.new_chain("test")

      chain =
        CrucibleTrace.add_event(
          chain,
          CrucibleTrace.create_event(:hypothesis_formed, "Save test", "Testing save/load")
        )

      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: chain
      }

      path = Path.join(tmp_dir, "test_trace.json")

      # Save trace
      assert :ok = TraceIntegration.save_trace(ctx, path)
      assert File.exists?(path)

      # Load trace
      assert {:ok, loaded_chain} = TraceIntegration.load_trace(path)
      assert loaded_chain.name == "test"
      assert length(loaded_chain.events) == 1
      assert List.first(loaded_chain.events).decision == "Save test"
    end

    test "returns error when trying to save nil trace" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      assert {:error, :no_trace} = TraceIntegration.save_trace(ctx, "/tmp/trace.json")
    end
  end

  describe "integration with pipeline" do
    test "trace events flow through pipeline stages" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{id: "exp", backend: nil, pipeline: []},
        trace: nil
      }

      # Initialize trace
      ctx = TraceIntegration.init_trace(ctx, "Pipeline Test")

      # Simulate pipeline flow
      ctx = TraceIntegration.emit_stage_start(ctx, :data_load, %{})
      ctx = TraceIntegration.emit_stage_complete(ctx, :data_load, %{records: 100})

      ctx = TraceIntegration.emit_stage_start(ctx, :backend_call, %{})

      ctx =
        TraceIntegration.emit_event(
          ctx,
          :decision_made,
          "Use ensemble for reliability",
          "Multiple models provide better consensus",
          alternatives: ["Single model"],
          confidence: 0.85
        )

      ctx = TraceIntegration.emit_stage_complete(ctx, :backend_call, %{})

      ctx = TraceIntegration.emit_stage_start(ctx, :bench, %{})
      ctx = TraceIntegration.emit_stage_complete(ctx, :bench, %{p_value: 0.03})

      # Verify trace captured the flow
      assert TraceIntegration.event_count(ctx) == 7

      stage_starts = TraceIntegration.filter_events(ctx, :stage_start)
      assert length(stage_starts) == 3

      stage_completes = TraceIntegration.filter_events(ctx, :stage_complete)
      assert length(stage_completes) == 3

      decisions = TraceIntegration.filter_events(ctx, :decision_made)
      assert length(decisions) == 1
    end
  end
end
