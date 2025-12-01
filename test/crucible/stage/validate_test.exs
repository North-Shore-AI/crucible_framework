defmodule Crucible.Stage.ValidateTest do
  # async: false to avoid race conditions with shared :backends application env
  use ExUnit.Case, async: false

  # Suppress expected validation log messages
  @moduletag capture_log: true

  alias Crucible.Context
  alias Crucible.Stage.Validate
  alias CrucibleIR.{BackendRef, DatasetRef, Experiment, StageDef, OutputSpec}

  setup do
    # Save original configs
    original_stage_registry = Application.get_env(:crucible_framework, :stage_registry)
    original_backends = Application.get_env(:crucible_framework, :backends)

    # Setup mock stage registry
    Application.put_env(:crucible_framework, :stage_registry, %{
      data_load: Crucible.Stage.DataLoad,
      backend_call: Crucible.Stage.BackendCall,
      validate: Crucible.Stage.Validate,
      missing_module: NonExistentModule
    })

    # Setup mock backend registry
    Application.put_env(:crucible_framework, :backends, %{
      tinkex: Crucible.Backend.Tinkex,
      missing_backend: NonExistentBackendModule
    })

    on_exit(fn ->
      if original_stage_registry do
        Application.put_env(:crucible_framework, :stage_registry, original_stage_registry)
      else
        Application.delete_env(:crucible_framework, :stage_registry)
      end

      if original_backends do
        Application.put_env(:crucible_framework, :backends, original_backends)
      else
        Application.delete_env(:crucible_framework, :backends)
      end
    end)

    :ok
  end

  # ============================================================================
  # Valid Experiment Tests
  # ============================================================================

  describe "validate/2 with valid experiment" do
    test "passes for minimal valid experiment" do
      experiment = %Experiment{
        id: "test_exp",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [
          %StageDef{name: :validate},
          %StageDef{name: :data_load}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})
      assert Context.has_metric?(new_ctx, :validation)

      validation = Context.get_metric(new_ctx, :validation)
      # Minimal experiment passes with warnings due to empty backend options
      assert validation.status in [:passed, :passed_with_warnings]
    end

    test "passes for complete experiment" do
      experiment = %Experiment{
        id: "complete_exp",
        backend: %BackendRef{id: :tinkex, options: %{model: "test"}},
        dataset: %DatasetRef{
          provider: Crucible.Data.InMemory,
          name: "test_dataset",
          options: %{}
        },
        pipeline: [
          %StageDef{name: :validate},
          %StageDef{name: :data_load},
          %StageDef{name: :backend_call}
        ],
        reliability: %CrucibleIR.Reliability.Config{},
        outputs: [
          %OutputSpec{name: :report, formats: [:markdown], sink: :file}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed
    end
  end

  # ============================================================================
  # Backend Validation Tests
  # ============================================================================

  describe "backend validation" do
    test "fails when backend is nil" do
      experiment = %Experiment{
        id: "no_backend",
        backend: nil,
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{})
      assert "No backend configured" in errors
    end

    test "fails when backend is not registered" do
      experiment = %Experiment{
        id: "unknown_backend",
        backend: %BackendRef{id: :unknown_backend, options: %{}},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{})
      assert Enum.any?(errors, &String.contains?(&1, "not registered"))
    end

    test "warns when backend has no options" do
      experiment = %Experiment{
        id: "no_options",
        backend: %BackendRef{id: :tinkex, options: nil},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed_with_warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "no options"))
    end

    test "skips backend validation when requested" do
      experiment = %Experiment{
        id: "skip_backend",
        backend: %BackendRef{id: :nonexistent, options: %{}},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{skip_backend: true})
    end
  end

  # ============================================================================
  # Stage Validation Tests
  # ============================================================================

  describe "stage validation" do
    test "passes when all stages are registered" do
      experiment = %Experiment{
        id: "valid_stages",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [
          %StageDef{name: :data_load},
          %StageDef{name: :backend_call}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "fails when stage is not registered" do
      experiment = %Experiment{
        id: "unknown_stage",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [
          %StageDef{name: :unknown_stage}
        ]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{})
      assert Enum.any?(errors, &String.contains?(&1, "not registered"))
    end

    test "warns about duplicate stages" do
      experiment = %Experiment{
        id: "duplicate_stages",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [
          %StageDef{name: :data_load},
          %StageDef{name: :data_load}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed_with_warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "Duplicate"))
    end

    test "validates explicit stage module" do
      experiment = %Experiment{
        id: "explicit_module",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [
          %StageDef{name: :custom, module: Crucible.Stage.DataLoad}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end
  end

  # ============================================================================
  # Dataset Validation Tests
  # ============================================================================

  describe "dataset validation" do
    test "passes with nil dataset" do
      experiment = %Experiment{
        id: "no_dataset",
        backend: %BackendRef{id: :tinkex, options: %{}},
        dataset: nil,
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "warns when dataset has no provider" do
      experiment = %Experiment{
        id: "no_provider",
        backend: %BackendRef{id: :tinkex, options: %{}},
        dataset: %DatasetRef{provider: nil, name: "test"},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed_with_warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "no provider"))
    end

    test "skips dataset validation when requested" do
      experiment = %Experiment{
        id: "skip_dataset",
        backend: %BackendRef{id: :tinkex, options: %{}},
        dataset: %DatasetRef{provider: NonExistentProvider, name: "test"},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{skip_dataset: true})
    end
  end

  # ============================================================================
  # Reliability Config Validation Tests
  # ============================================================================

  describe "reliability configuration validation" do
    test "passes with default reliability config" do
      experiment = %Experiment{
        id: "default_reliability",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        reliability: %CrucibleIR.Reliability.Config{}
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "validates ensemble configuration" do
      experiment = %Experiment{
        id: "ensemble_exp",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        reliability: %CrucibleIR.Reliability.Config{
          ensemble: %CrucibleIR.Reliability.Ensemble{
            strategy: :majority_vote,
            models: [
              %BackendRef{id: :tinkex, options: %{}}
            ]
          }
        }
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end

    test "fails with invalid ensemble strategy" do
      experiment = %Experiment{
        id: "bad_ensemble",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        reliability: %CrucibleIR.Reliability.Config{
          ensemble: %CrucibleIR.Reliability.Ensemble{
            strategy: :invalid_strategy,
            models: [%BackendRef{id: :tinkex, options: %{}}]
          }
        }
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{})
      assert Enum.any?(errors, &String.contains?(&1, "Unknown ensemble strategy"))
    end

    test "fails when ensemble members are not registered" do
      experiment = %Experiment{
        id: "bad_members",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        reliability: %CrucibleIR.Reliability.Config{
          ensemble: %CrucibleIR.Reliability.Ensemble{
            strategy: :majority_vote,
            models: [
              %BackendRef{id: :unknown_backend, options: %{}}
            ]
          }
        }
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, errors}} = Validate.run(ctx, %{})
      assert Enum.any?(errors, &String.contains?(&1, "not registered"))
    end
  end

  # ============================================================================
  # Output Validation Tests
  # ============================================================================

  describe "output validation" do
    test "warns when no outputs configured" do
      experiment = %Experiment{
        id: "no_outputs",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        outputs: []
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed_with_warnings
      assert Enum.any?(validation.warnings, &String.contains?(&1, "No outputs"))
    end

    test "passes with valid outputs" do
      experiment = %Experiment{
        id: "valid_outputs",
        backend: %BackendRef{id: :tinkex, options: %{}},
        pipeline: [%StageDef{name: :validate}],
        outputs: [
          %OutputSpec{name: :report, formats: [:markdown, :json], sink: :file}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{})
    end
  end

  # ============================================================================
  # Strict Mode Tests
  # ============================================================================

  describe "strict mode" do
    test "fails when warnings present in strict mode" do
      experiment = %Experiment{
        id: "strict_test",
        backend: %BackendRef{id: :tinkex, options: nil},
        pipeline: [%StageDef{name: :validate}]
      }

      ctx = build_context(experiment)

      assert {:error, {:validation_failed, messages}} = Validate.run(ctx, %{strict: true})
      assert Enum.any?(messages, &String.contains?(&1, "no options"))
    end

    test "passes in strict mode with no warnings" do
      experiment = %Experiment{
        id: "strict_pass",
        backend: %BackendRef{id: :tinkex, options: %{model: "test"}},
        pipeline: [%StageDef{name: :validate}],
        outputs: [%OutputSpec{name: :report, formats: [:markdown], sink: :file}]
      }

      ctx = build_context(experiment)

      assert {:ok, _new_ctx} = Validate.run(ctx, %{strict: true})
    end
  end

  # ============================================================================
  # Integration Tests
  # ============================================================================

  describe "integration scenarios" do
    test "complete validation workflow" do
      experiment = %Experiment{
        id: "integration_test",
        description: "Full integration test",
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
        reliability: %CrucibleIR.Reliability.Config{},
        outputs: [
          %OutputSpec{name: :metrics, formats: [:json], sink: :file}
        ]
      }

      ctx = build_context(experiment)

      assert {:ok, new_ctx} = Validate.run(ctx, %{})

      validation = Context.get_metric(new_ctx, :validation)
      assert validation.status == :passed
      assert validation.details.backend.status == :ok
      assert validation.details.stages.status == :ok
      assert validation.details.dataset.status == :ok
      assert validation.details.outputs.status == :ok
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
