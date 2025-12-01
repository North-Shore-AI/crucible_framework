# CrucibleFramework v0.4.0 Enhancements Demo
#
# This example demonstrates the new features added in v0.4.0:
# 1. Enhanced Context helper functions
# 2. Pre-flight validation with Stage.Validate
# 3. Automatic stage completion tracking
#
# Run: mix run examples/v0.4.0_enhancements_demo.exs

alias Crucible.Context
alias Crucible.IR.{BackendRef, DatasetRef, Experiment, OutputSpec, ReliabilityConfig, StageDef}

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("CrucibleFramework v0.4.0 Enhancements Demo")
IO.puts(String.duplicate("=", 80) <> "\n")

# ============================================================================
# Part 1: Context Helper Functions Demo
# ============================================================================

IO.puts("Part 1: Enhanced Context Helpers")
IO.puts(String.duplicate("-", 40))

# Create a sample context
experiment = %Experiment{
  id: "demo_exp",
  backend: %BackendRef{id: :tinkex, options: %{}},
  pipeline: []
}

ctx = %Context{
  experiment_id: "demo_exp",
  run_id: "demo_run_001",
  experiment: experiment
}

IO.puts("Initial context created\n")

# Metrics Management
IO.puts("1. Metrics Management:")

ctx =
  ctx
  |> Context.put_metric(:accuracy, 0.95)
  |> Context.put_metric(:loss, 0.05)
  |> Context.merge_metrics(%{f1_score: 0.93, precision: 0.96})

IO.puts("  Added metrics: accuracy=0.95, loss=0.05, f1_score=0.93, precision=0.96")
IO.puts("  Get accuracy: #{Context.get_metric(ctx, :accuracy)}")
IO.puts("  Has f1_score?: #{Context.has_metric?(ctx, :f1_score)}")

# Update metric with function
ctx = Context.update_metric(ctx, :accuracy, fn acc -> acc + 0.02 end)
IO.puts("  Updated accuracy: #{Context.get_metric(ctx, :accuracy)}\n")

# Output Management
IO.puts("2. Output Management:")

ctx =
  ctx
  |> Context.add_output(%{prompt: "test1", response: "output1"})
  |> Context.add_outputs([
    %{prompt: "test2", response: "output2"},
    %{prompt: "test3", response: "output3"}
  ])

IO.puts("  Added 3 outputs")
IO.puts("  Total outputs: #{length(ctx.outputs)}\n")

# Artifact Management
IO.puts("3. Artifact Management:")

ctx =
  ctx
  |> Context.put_artifact(:report, "/tmp/report.md")
  |> Context.put_artifact(:checkpoint, "/tmp/model.bin")

IO.puts("  Stored artifacts: report, checkpoint")
IO.puts("  Get report: #{Context.get_artifact(ctx, :report)}")
IO.puts("  Has checkpoint?: #{Context.has_artifact?(ctx, :checkpoint)}\n")

# Assigns Management (Phoenix-style)
IO.puts("4. Assigns Management:")

ctx =
  ctx
  |> Context.assign(:user, "alice")
  |> Context.assign(priority: :high, team: "ml_research")

IO.puts("  Assigned: user=alice, priority=high, team=ml_research")
IO.puts("  Get user: #{ctx.assigns.user}")
IO.puts("  Get priority: #{ctx.assigns.priority}\n")

# Query Functions
IO.puts("5. Query Functions:")

# Add dataset to test has_data?
ctx = %Context{ctx | dataset: [1, 2, 3], examples: [1, 2, 3]}

IO.puts("  Has data?: #{Context.has_data?(ctx)}")
IO.puts("  Has backend session (tinkex)?: #{Context.has_backend_session?(ctx, :tinkex)}\n")

# Stage Tracking
IO.puts("6. Stage Tracking:")

ctx =
  ctx
  |> Context.mark_stage_complete(:data_load)
  |> Context.mark_stage_complete(:backend_call)
  |> Context.mark_stage_complete(:bench)

IO.puts("  Marked stages complete: data_load, backend_call, bench")
IO.puts("  Is data_load complete?: #{Context.stage_completed?(ctx, :data_load)}")
IO.puts("  Completed stages: #{inspect(Context.completed_stages(ctx))}\n")

# ============================================================================
# Part 2: Validation Stage Demo
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("Part 2: Pre-Flight Validation Demo")
IO.puts(String.duplicate("-", 40))

# Configure stage registry for validation
Application.put_env(:crucible_framework, :stage_registry, %{
  validate: Crucible.Stage.Validate,
  data_load: Crucible.Stage.DataLoad,
  backend_call: Crucible.Stage.BackendCall
})

Application.put_env(:crucible_framework, :backends, %{
  tinkex: Crucible.Backend.Tinkex
})

# Example 1: Valid Experiment
IO.puts("\nExample 1: Valid Experiment")

valid_experiment = %Experiment{
  id: "valid_exp",
  description: "A properly configured experiment",
  backend: %BackendRef{id: :tinkex, options: %{model: "test"}},
  dataset: %DatasetRef{
    provider: Crucible.Data.InMemory,
    name: "test_data",
    options: %{limit: 10}
  },
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :data_load},
    %StageDef{name: :backend_call}
  ],
  reliability: %ReliabilityConfig{},
  outputs: [
    %OutputSpec{name: :report, formats: [:markdown], sink: :file}
  ]
}

valid_ctx = %Context{
  experiment_id: valid_experiment.id,
  run_id: "run_001",
  experiment: valid_experiment
}

case Crucible.Stage.Validate.run(valid_ctx, %{}) do
  {:ok, result_ctx} ->
    validation = Context.get_metric(result_ctx, :validation)
    IO.puts("  Status: #{validation.status}")
    IO.puts("  Backend: #{validation.details.backend.status}")
    IO.puts("  Stages: #{validation.details.stages.status}")
    IO.puts("  Dataset: #{validation.details.dataset.status}")
    IO.puts("  Result: ✓ PASSED\n")

  {:error, {:validation_failed, errors}} ->
    IO.puts("  Result: ✗ FAILED")
    Enum.each(errors, &IO.puts("    - #{&1}"))
end

# Example 2: Experiment with Warnings
IO.puts("Example 2: Experiment with Warnings")

warning_experiment = %Experiment{
  id: "warning_exp",
  backend: %BackendRef{id: :tinkex, options: nil},
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :data_load}
  ]
}

warning_ctx = %Context{
  experiment_id: warning_experiment.id,
  run_id: "run_002",
  experiment: warning_experiment
}

case Crucible.Stage.Validate.run(warning_ctx, %{}) do
  {:ok, result_ctx} ->
    validation = Context.get_metric(result_ctx, :validation)
    IO.puts("  Status: #{validation.status}")
    IO.puts("  Warnings:")
    Enum.each(validation.warnings, &IO.puts("    - #{&1}"))
    IO.puts("  Result: ✓ PASSED (with warnings)\n")

  {:error, {:validation_failed, errors}} ->
    IO.puts("  Result: ✗ FAILED")
    Enum.each(errors, &IO.puts("    - #{&1}"))
end

# Example 3: Invalid Experiment
IO.puts("Example 3: Invalid Experiment")

invalid_experiment = %Experiment{
  id: "invalid_exp",
  backend: %BackendRef{id: :nonexistent_backend, options: %{}},
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :unknown_stage}
  ]
}

invalid_ctx = %Context{
  experiment_id: invalid_experiment.id,
  run_id: "run_003",
  experiment: invalid_experiment
}

case Crucible.Stage.Validate.run(invalid_ctx, %{}) do
  {:ok, result_ctx} ->
    validation = Context.get_metric(result_ctx, :validation)
    IO.puts("  Status: #{validation.status}")
    IO.puts("  Result: ✓ PASSED\n")

  {:error, {:validation_failed, errors}} ->
    IO.puts("  Result: ✗ FAILED")
    IO.puts("  Errors:")
    Enum.each(errors, &IO.puts("    - #{&1}"))
    IO.puts()
end

# ============================================================================
# Part 3: Integration Example
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("Part 3: Complete Workflow Integration")
IO.puts(String.duplicate("-", 40))

# Simulate a complete experiment workflow
IO.puts("\nSimulating experiment workflow...\n")

workflow_ctx = %Context{
  experiment_id: "workflow_exp",
  run_id: "workflow_run_001",
  experiment: valid_experiment
}

# Stage 1: Validation
IO.puts("Stage 1: Validation")

case Crucible.Stage.Validate.run(workflow_ctx, %{}) do
  {:ok, ctx_after_validation} ->
    ctx_after_validation = Context.mark_stage_complete(ctx_after_validation, :validate)
    IO.puts("  ✓ Validation passed")

    # Stage 2: Simulate data loading
    IO.puts("\nStage 2: Data Loading (simulated)")

    ctx_after_data =
      ctx_after_validation
      |> Context.assign(:dataset_loaded, true)
      |> Context.put_metric(:data_load, %{examples: 100, batches: 25})
      |> Context.mark_stage_complete(:data_load)

    IO.puts("  ✓ Loaded 100 examples in 25 batches")

    # Stage 3: Simulate backend call
    IO.puts("\nStage 3: Backend Call (simulated)")

    ctx_after_backend =
      ctx_after_data
      |> Context.add_outputs([
        %{prompt: "What is AI?", response: "AI is..."},
        %{prompt: "Explain ML", response: "ML is..."}
      ])
      |> Context.put_metric(:backend, %{samples: 2, avg_latency_ms: 234})
      |> Context.mark_stage_complete(:backend_call)

    IO.puts("  ✓ Generated 2 outputs (avg latency: 234ms)")

    # Stage 4: Simulate analysis
    IO.puts("\nStage 4: Analysis (simulated)")

    ctx_final =
      ctx_after_backend
      |> Context.merge_metrics(%{
        accuracy: 0.95,
        f1_score: 0.93,
        latency_p50: 200,
        latency_p99: 450
      })
      |> Context.put_artifact(:report, "/tmp/experiment_report.md")
      |> Context.mark_stage_complete(:analysis)

    IO.puts("  ✓ Analysis complete")

    # Final Summary
    IO.puts("\n" <> String.duplicate("-", 40))
    IO.puts("Experiment Summary")
    IO.puts(String.duplicate("-", 40))
    IO.puts("Completed stages: #{inspect(Context.completed_stages(ctx_final))}")
    IO.puts("Total outputs: #{length(ctx_final.outputs)}")
    IO.puts("Accuracy: #{Context.get_metric(ctx_final, :accuracy)}")
    IO.puts("F1 Score: #{Context.get_metric(ctx_final, :f1_score)}")
    IO.puts("Artifacts: #{inspect(Map.keys(ctx_final.artifacts))}")
    IO.puts("\n✓ Workflow completed successfully!")

  {:error, {:validation_failed, errors}} ->
    IO.puts("  ✗ Validation failed:")
    Enum.each(errors, &IO.puts("    - #{&1}"))
end

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("Demo Complete!")
IO.puts(String.duplicate("=", 80) <> "\n")

IO.puts("""
Key Takeaways:
1. Context helpers reduce boilerplate by 40-60%
2. Pre-flight validation catches errors early
3. Stage tracking provides built-in progress monitoring
4. All features are backward compatible
5. Phoenix-style patterns improve developer experience

See docs/20251125/enhancements_design.md for full details.
""")
