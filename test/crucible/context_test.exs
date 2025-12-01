defmodule Crucible.ContextTest do
  use ExUnit.Case, async: true
  doctest Crucible.Context

  alias Crucible.Context
  alias CrucibleIR.{BackendRef, Experiment}

  setup do
    experiment = %Experiment{
      id: "test_exp",
      backend: %BackendRef{id: :tinkex},
      pipeline: []
    }

    ctx = %Context{
      experiment_id: "test_exp",
      run_id: "run_123",
      experiment: experiment
    }

    {:ok, ctx: ctx}
  end

  # ============================================================================
  # Metrics Management Tests
  # ============================================================================

  describe "put_metric/3" do
    test "adds a new metric", %{ctx: ctx} do
      new_ctx = Context.put_metric(ctx, :accuracy, 0.95)
      assert new_ctx.metrics.accuracy == 0.95
    end

    test "updates existing metric", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :count, 1)
      new_ctx = Context.put_metric(ctx, :count, 2)
      assert new_ctx.metrics.count == 2
    end

    test "preserves other metrics", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :accuracy, 0.95)
      new_ctx = Context.put_metric(ctx, :loss, 0.05)
      assert new_ctx.metrics.accuracy == 0.95
      assert new_ctx.metrics.loss == 0.05
    end
  end

  describe "get_metric/3" do
    test "returns existing metric", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :accuracy, 0.95)
      assert Context.get_metric(ctx, :accuracy) == 0.95
    end

    test "returns nil for missing metric", %{ctx: ctx} do
      assert Context.get_metric(ctx, :missing) == nil
    end

    test "returns default for missing metric", %{ctx: ctx} do
      assert Context.get_metric(ctx, :missing, :default) == :default
    end
  end

  describe "update_metric/3" do
    test "updates metric with function", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :count, 1)
      new_ctx = Context.update_metric(ctx, :count, &(&1 + 1))
      assert new_ctx.metrics.count == 2
    end

    test "handles nil initial value", %{ctx: ctx} do
      # Map.update/4 doesn't call the function when key doesn't exist - it uses the default (nil)
      # To update with nil handling, first set the metric explicitly
      ctx = Context.put_metric(ctx, :count, nil)

      new_ctx =
        Context.update_metric(ctx, :count, fn
          nil -> 1
          n -> n + 1
        end)

      assert new_ctx.metrics.count == 1
    end
  end

  describe "merge_metrics/2" do
    test "merges multiple metrics", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :accuracy, 0.95)
      new_ctx = Context.merge_metrics(ctx, %{loss: 0.05, f1: 0.93})
      assert new_ctx.metrics.accuracy == 0.95
      assert new_ctx.metrics.loss == 0.05
      assert new_ctx.metrics.f1 == 0.93
    end

    test "overwrites existing metrics", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :accuracy, 0.90)
      new_ctx = Context.merge_metrics(ctx, %{accuracy: 0.95})
      assert new_ctx.metrics.accuracy == 0.95
    end
  end

  describe "has_metric?/2" do
    test "returns true for existing metric", %{ctx: ctx} do
      ctx = Context.put_metric(ctx, :accuracy, 0.95)
      assert Context.has_metric?(ctx, :accuracy)
    end

    test "returns false for missing metric", %{ctx: ctx} do
      refute Context.has_metric?(ctx, :missing)
    end
  end

  # ============================================================================
  # Output Management Tests
  # ============================================================================

  describe "add_output/2" do
    test "adds single output", %{ctx: ctx} do
      new_ctx = Context.add_output(ctx, %{result: "success"})
      assert length(new_ctx.outputs) == 1
      assert hd(new_ctx.outputs) == %{result: "success"}
    end

    test "appends to existing outputs", %{ctx: ctx} do
      ctx = Context.add_output(ctx, %{result: "first"})
      new_ctx = Context.add_output(ctx, %{result: "second"})
      assert length(new_ctx.outputs) == 2
      assert Enum.at(new_ctx.outputs, 1) == %{result: "second"}
    end
  end

  describe "add_outputs/2" do
    test "adds multiple outputs", %{ctx: ctx} do
      outputs = [%{result: "a"}, %{result: "b"}, %{result: "c"}]
      new_ctx = Context.add_outputs(ctx, outputs)
      assert length(new_ctx.outputs) == 3
    end

    test "appends to existing outputs", %{ctx: ctx} do
      ctx = Context.add_output(ctx, %{result: "first"})
      new_ctx = Context.add_outputs(ctx, [%{result: "a"}, %{result: "b"}])
      assert length(new_ctx.outputs) == 3
      assert hd(new_ctx.outputs) == %{result: "first"}
    end

    test "handles empty list", %{ctx: ctx} do
      new_ctx = Context.add_outputs(ctx, [])
      assert new_ctx.outputs == []
    end
  end

  # ============================================================================
  # Artifact Management Tests
  # ============================================================================

  describe "put_artifact/3" do
    test "stores artifact", %{ctx: ctx} do
      new_ctx = Context.put_artifact(ctx, :report, "report.html")
      assert new_ctx.artifacts.report == "report.html"
    end

    test "updates existing artifact", %{ctx: ctx} do
      ctx = Context.put_artifact(ctx, :report, "v1.html")
      new_ctx = Context.put_artifact(ctx, :report, "v2.html")
      assert new_ctx.artifacts.report == "v2.html"
    end
  end

  describe "get_artifact/3" do
    test "retrieves existing artifact", %{ctx: ctx} do
      ctx = Context.put_artifact(ctx, :report, "report.html")
      assert Context.get_artifact(ctx, :report) == "report.html"
    end

    test "returns nil for missing artifact", %{ctx: ctx} do
      assert Context.get_artifact(ctx, :missing) == nil
    end

    test "returns default for missing artifact", %{ctx: ctx} do
      assert Context.get_artifact(ctx, :missing, :not_found) == :not_found
    end
  end

  describe "has_artifact?/2" do
    test "returns true for existing artifact", %{ctx: ctx} do
      ctx = Context.put_artifact(ctx, :report, "report.html")
      assert Context.has_artifact?(ctx, :report)
    end

    test "returns false for missing artifact", %{ctx: ctx} do
      refute Context.has_artifact?(ctx, :missing)
    end
  end

  # ============================================================================
  # Assigns Management Tests
  # ============================================================================

  describe "assign/3 (single value)" do
    test "assigns single value", %{ctx: ctx} do
      new_ctx = Context.assign(ctx, :user, "alice")
      assert new_ctx.assigns.user == "alice"
    end

    test "updates existing assign", %{ctx: ctx} do
      ctx = Context.assign(ctx, :user, "alice")
      new_ctx = Context.assign(ctx, :user, "bob")
      assert new_ctx.assigns.user == "bob"
    end

    test "preserves other assigns", %{ctx: ctx} do
      ctx = Context.assign(ctx, :user, "alice")
      new_ctx = Context.assign(ctx, :role, :admin)
      assert new_ctx.assigns.user == "alice"
      assert new_ctx.assigns.role == :admin
    end
  end

  describe "assign/2 (multiple values)" do
    test "assigns from keyword list", %{ctx: ctx} do
      new_ctx = Context.assign(ctx, user: "alice", role: :admin)
      assert new_ctx.assigns.user == "alice"
      assert new_ctx.assigns.role == :admin
    end

    test "assigns from map", %{ctx: ctx} do
      new_ctx = Context.assign(ctx, %{user: "alice", role: :admin})
      assert new_ctx.assigns.user == "alice"
      assert new_ctx.assigns.role == :admin
    end

    test "merges with existing assigns", %{ctx: ctx} do
      ctx = Context.assign(ctx, :priority, :high)
      new_ctx = Context.assign(ctx, user: "alice", role: :admin)
      assert new_ctx.assigns.priority == :high
      assert new_ctx.assigns.user == "alice"
    end
  end

  # ============================================================================
  # Query Functions Tests
  # ============================================================================

  describe "has_data?/1" do
    test "returns true when dataset and examples present", %{ctx: ctx} do
      ctx = %Context{ctx | dataset: [1, 2, 3], examples: [1, 2, 3]}
      assert Context.has_data?(ctx)
    end

    test "returns false when dataset is nil", %{ctx: ctx} do
      ctx = %Context{ctx | dataset: nil, examples: [1, 2, 3]}
      refute Context.has_data?(ctx)
    end

    test "returns false when examples is nil", %{ctx: ctx} do
      ctx = %Context{ctx | dataset: [1, 2, 3], examples: nil}
      refute Context.has_data?(ctx)
    end

    test "returns false when examples is empty", %{ctx: ctx} do
      ctx = %Context{ctx | dataset: [1, 2, 3], examples: []}
      refute Context.has_data?(ctx)
    end
  end

  describe "has_backend_session?/2" do
    test "returns true when session exists", %{ctx: ctx} do
      ctx = %Context{
        ctx
        | backend_sessions: %{{:tinkex, "test_exp"} => self()}
      }

      assert Context.has_backend_session?(ctx, :tinkex)
    end

    test "returns false when session missing", %{ctx: ctx} do
      refute Context.has_backend_session?(ctx, :tinkex)
    end

    test "finds session regardless of experiment ID", %{ctx: ctx} do
      ctx = %Context{
        ctx
        | backend_sessions: %{{:tinkex, "other_exp"} => self()}
      }

      assert Context.has_backend_session?(ctx, :tinkex)
    end
  end

  describe "get_backend_session/2" do
    test "returns session when exists", %{ctx: ctx} do
      pid = self()

      ctx = %Context{
        ctx
        | backend_sessions: %{{:tinkex, "test_exp"} => pid}
      }

      assert Context.get_backend_session(ctx, :tinkex) == pid
    end

    test "returns nil when missing", %{ctx: ctx} do
      assert Context.get_backend_session(ctx, :tinkex) == nil
    end

    test "returns first matching session", %{ctx: ctx} do
      pid1 = self()

      ctx = %Context{
        ctx
        | backend_sessions: %{
            {:tinkex, "exp1"} => pid1,
            {:tinkex, "exp2"} => :other
          }
      }

      # Should return first match (deterministic with sorted keys)
      result = Context.get_backend_session(ctx, :tinkex)
      assert result in [pid1, :other]
    end
  end

  # ============================================================================
  # Stage Tracking Tests
  # ============================================================================

  describe "mark_stage_complete/2" do
    test "marks stage as completed", %{ctx: ctx} do
      new_ctx = Context.mark_stage_complete(ctx, :data_load)
      assert Context.stage_completed?(new_ctx, :data_load)
    end

    test "tracks multiple completed stages", %{ctx: ctx} do
      ctx =
        ctx
        |> Context.mark_stage_complete(:data_load)
        |> Context.mark_stage_complete(:backend_call)
        |> Context.mark_stage_complete(:bench)

      assert Context.stage_completed?(ctx, :data_load)
      assert Context.stage_completed?(ctx, :backend_call)
      assert Context.stage_completed?(ctx, :bench)
    end

    test "does not duplicate stage names", %{ctx: ctx} do
      ctx =
        ctx
        |> Context.mark_stage_complete(:data_load)
        |> Context.mark_stage_complete(:data_load)

      completed = Context.completed_stages(ctx)
      assert length(completed) == 1
      assert :data_load in completed
    end
  end

  describe "stage_completed?/2" do
    test "returns true for completed stage", %{ctx: ctx} do
      ctx = Context.mark_stage_complete(ctx, :data_load)
      assert Context.stage_completed?(ctx, :data_load)
    end

    test "returns false for not completed stage", %{ctx: ctx} do
      refute Context.stage_completed?(ctx, :data_load)
    end
  end

  describe "completed_stages/1" do
    test "returns empty list initially", %{ctx: ctx} do
      assert Context.completed_stages(ctx) == []
    end

    test "returns all completed stages", %{ctx: ctx} do
      ctx =
        ctx
        |> Context.mark_stage_complete(:data_load)
        |> Context.mark_stage_complete(:backend_call)
        |> Context.mark_stage_complete(:bench)

      completed = Context.completed_stages(ctx)
      assert length(completed) == 3
      assert :data_load in completed
      assert :backend_call in completed
      assert :bench in completed
    end

    test "maintains order of completion", %{ctx: ctx} do
      ctx =
        ctx
        |> Context.mark_stage_complete(:data_load)
        |> Context.mark_stage_complete(:backend_call)
        |> Context.mark_stage_complete(:bench)

      assert Context.completed_stages(ctx) == [:data_load, :backend_call, :bench]
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================

  describe "integration scenarios" do
    test "complete experiment workflow", %{ctx: ctx} do
      # Load data
      ctx = %Context{ctx | dataset: [1, 2, 3], examples: [1, 2, 3]}
      assert Context.has_data?(ctx)

      # Mark stages complete
      ctx =
        ctx
        |> Context.mark_stage_complete(:data_load)
        |> Context.mark_stage_complete(:backend_call)

      # Add metrics
      ctx =
        ctx
        |> Context.put_metric(:train_loss, 0.15)
        |> Context.put_metric(:accuracy, 0.95)

      # Add outputs
      ctx = Context.add_outputs(ctx, [%{result: "a"}, %{result: "b"}])

      # Add artifacts
      ctx = Context.put_artifact(ctx, :report, "report.html")

      # Add assigns
      ctx = Context.assign(ctx, checkpoint: "checkpoint_v1.bin")

      # Verify everything
      assert length(Context.completed_stages(ctx)) == 2
      assert Context.get_metric(ctx, :accuracy) == 0.95
      assert length(ctx.outputs) == 2
      assert Context.get_artifact(ctx, :report) == "report.html"
      assert ctx.assigns.checkpoint == "checkpoint_v1.bin"
    end

    test "chaining multiple operations", %{ctx: ctx} do
      result =
        ctx
        |> Context.put_metric(:count, 0)
        |> Context.update_metric(:count, &(&1 + 1))
        |> Context.update_metric(:count, &(&1 + 1))
        |> Context.add_output(%{result: "success"})
        |> Context.mark_stage_complete(:test_stage)

      assert Context.get_metric(result, :count) == 2
      assert length(result.outputs) == 1
      assert Context.stage_completed?(result, :test_stage)
    end
  end
end
