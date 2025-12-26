defmodule Crucible.Stage.ValidateTest do
  # async: false to avoid race conditions with shared application env
  use ExUnit.Case, async: false

  # Suppress expected validation log messages
  @moduletag capture_log: true

  alias Crucible.Context
  alias Crucible.Stage.Validate
  alias CrucibleIR.{BackendRef, Experiment, StageDef}

  setup do
    # Save original stage_registry
    original_stage_registry = Application.get_env(:crucible_framework, :stage_registry)

    # Setup test stage registry
    Application.put_env(:crucible_framework, :stage_registry, %{
      validate: Crucible.Stage.Validate,
      bench: Crucible.Stage.Bench,
      data_checks: Crucible.Stage.DataChecks,
      report: Crucible.Stage.Report,
      guardrails: Crucible.Stage.Guardrails,
      missing_module: NonExistentModule
    })

    on_exit(fn ->
      if original_stage_registry do
        Application.put_env(:crucible_framework, :stage_registry, original_stage_registry)
      else
        Application.delete_env(:crucible_framework, :stage_registry)
      end
    end)

    :ok
  end

  # ============================================================================
  # Valid Experiment Tests
  # ============================================================================

  describe "validate/2 with valid experiment" do
    test "passes for experiment with valid stages" do
      experiment = %Experiment{
        id: "test_exp",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :validate},
          %StageDef{name: :bench}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})
      assert Context.has_metric?(new_ctx, :validation)

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed
    end

    test "passes for stages with explicit module" do
      experiment = %Experiment{
        id: "explicit_module",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :custom, module: Crucible.Stage.Bench}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end
  end

  # ============================================================================
  # Stage Validation Tests
  # ============================================================================

  describe "stage validation" do
    test "passes when all stages are registered" do
      experiment = %Experiment{
        id: "valid_stages",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :validate},
          %StageDef{name: :bench},
          %StageDef{name: :report}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "fails when stage is not registered" do
      experiment = %Experiment{
        id: "unknown_stage",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :unknown_stage}
        ]
      }

      ctx = build_context(experiment)

      # Non-strict mode: errors become warnings that pass
      assert {:ok, new_ctx} = Validate.run(ctx, %{})
      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "not registered"))
    end

    test "fails in strict mode when stage is not registered" do
      experiment = %Experiment{
        id: "unknown_stage_strict",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :unknown_stage}
        ]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{strict: true})
      assert Enum.any?(errors, &String.contains?(&1, "not registered"))
    end

    test "validates explicit stage module" do
      experiment = %Experiment{
        id: "explicit_module",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :custom, module: Crucible.Stage.Bench}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "fails for explicit module that doesn't exist" do
      experiment = %Experiment{
        id: "bad_module",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :custom, module: NonExistentModule}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})
      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "cannot be loaded"))
    end

    test "fails for module that doesn't implement run/2" do
      experiment = %Experiment{
        id: "missing_run",
        backend: %BackendRef{id: :test},
        pipeline: [
          # String module exists but doesn't implement run/2
          %StageDef{name: :custom, module: String}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})
      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "does not implement run/2"))
    end
  end

  # ============================================================================
  # Strict Mode Tests
  # ============================================================================

  describe "strict mode" do
    test "fails when any validation issues in strict mode" do
      experiment = %Experiment{
        id: "strict_test",
        backend: %BackendRef{id: :test},
        pipeline: [%StageDef{name: :missing_stage}]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, _messages}} = Validate.run(ctx, %{strict: true})
    end

    test "passes in strict mode with no issues" do
      experiment = %Experiment{
        id: "strict_pass",
        backend: %BackendRef{id: :test},
        pipeline: [
          %StageDef{name: :validate},
          %StageDef{name: :bench}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{strict: true})
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp build_context(experiment) do
    %Context{
      experiment_id: experiment.id,
      run_id: "test_run_#{:erlang.unique_integer([:positive])}",
      experiment: experiment
    }
  end
end
