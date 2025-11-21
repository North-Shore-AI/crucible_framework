defmodule Crucible.Tinkex.ExperimentTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.Experiment

  doctest Experiment

  describe "new/1" do
    test "creates experiment with required fields" do
      {:ok, exp} =
        Experiment.new(
          name: "Test Experiment",
          base_model: "meta-llama/Llama-3.1-8B-Instruct"
        )

      assert exp.name == "Test Experiment"
      assert exp.base_model == "meta-llama/Llama-3.1-8B-Instruct"
      assert exp.id != nil
    end

    test "creates experiment with training config" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          training: %{
            epochs: 5,
            batch_size: 8,
            learning_rate: 2.0e-4,
            lora_rank: 16,
            lora_alpha: 32
          }
        )

      assert exp.training.epochs == 5
      assert exp.training.batch_size == 8
      assert exp.training.learning_rate == 2.0e-4
    end

    test "creates experiment with evaluation config" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          evaluation: %{
            test_data: "scifact_dev",
            max_samples: 50,
            metrics: [:schema_compliance, :citation_accuracy]
          }
        )

      assert exp.evaluation.test_data == "scifact_dev"
      assert :citation_accuracy in exp.evaluation.metrics
    end

    test "creates experiment with hyperparameter sweep" do
      {:ok, exp} =
        Experiment.new(
          name: "Sweep Test",
          base_model: "model",
          parameters: %{
            citation_validity_weight: [2.0, 5.0, 7.0],
            learning_rate: [1.0e-4, 2.0e-4]
          }
        )

      assert length(exp.parameters.citation_validity_weight) == 3
      assert length(exp.parameters.learning_rate) == 2
    end

    test "returns error for missing name" do
      assert {:error, "name is required"} = Experiment.new(base_model: "model")
    end

    test "returns error for missing base_model" do
      assert {:error, "base_model is required"} = Experiment.new(name: "Test")
    end
  end

  describe "generate_runs/1" do
    test "generates single run for no parameters" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model"
        )

      runs = Experiment.generate_runs(exp)

      assert length(runs) == 1
      assert hd(runs).run_id != nil
    end

    test "generates runs for parameter sweep" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          parameters: %{
            learning_rate: [1.0e-4, 2.0e-4],
            lora_rank: [16, 32]
          }
        )

      runs = Experiment.generate_runs(exp)

      # 2 * 2 = 4 combinations
      assert length(runs) == 4

      # Verify each run has unique parameters
      run_params = Enum.map(runs, & &1.params)
      assert length(Enum.uniq(run_params)) == 4
    end

    test "generates runs with repeat count" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          repeat: 3
        )

      runs = Experiment.generate_runs(exp)

      assert length(runs) == 3
    end
  end

  describe "validate/1" do
    test "validates complete experiment" do
      {:ok, exp} =
        Experiment.new(
          name: "Valid",
          base_model: "model",
          training: %{epochs: 5, batch_size: 8}
        )

      assert :ok = Experiment.validate(exp)
    end

    test "returns error for invalid epochs" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          training: %{epochs: 0}
        )

      assert {:error, "epochs must be positive"} = Experiment.validate(exp)
    end

    test "returns error for invalid batch_size" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          training: %{batch_size: 0}
        )

      assert {:error, "batch_size must be positive"} = Experiment.validate(exp)
    end
  end

  describe "to_config/1" do
    test "converts experiment to training config" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "meta-llama/Llama-3.1-8B-Instruct",
          training: %{
            epochs: 5,
            batch_size: 8,
            lora_rank: 16
          }
        )

      config = Experiment.to_training_config(exp)

      assert config.base_model == "meta-llama/Llama-3.1-8B-Instruct"
      assert config.epochs == 5
      assert config.batch_size == 8
      assert config.lora_rank == 16
    end
  end

  describe "quality_targets/1" do
    test "returns default quality targets" do
      {:ok, exp} = Experiment.new(name: "Test", base_model: "model")
      targets = Experiment.quality_targets(exp)

      assert targets.schema_compliance == 0.95
      assert targets.citation_accuracy == 0.95
    end

    test "returns custom quality targets" do
      {:ok, exp} =
        Experiment.new(
          name: "Test",
          base_model: "model",
          quality_targets: %{
            schema_compliance: 0.99,
            citation_accuracy: 0.98
          }
        )

      targets = Experiment.quality_targets(exp)
      assert targets.schema_compliance == 0.99
      assert targets.citation_accuracy == 0.98
    end
  end

  describe "status transitions" do
    test "starts in pending status" do
      {:ok, exp} = Experiment.new(name: "Test", base_model: "model")
      assert exp.status == :pending
    end

    test "can transition to running" do
      {:ok, exp} = Experiment.new(name: "Test", base_model: "model")
      {:ok, exp} = Experiment.start(exp)
      assert exp.status == :running
    end

    test "can transition to completed" do
      {:ok, exp} = Experiment.new(name: "Test", base_model: "model")
      {:ok, exp} = Experiment.start(exp)
      {:ok, exp} = Experiment.complete(exp, %{total_time: 100})
      assert exp.status == :completed
    end

    test "can transition to failed" do
      {:ok, exp} = Experiment.new(name: "Test", base_model: "model")
      {:ok, exp} = Experiment.start(exp)
      {:ok, exp} = Experiment.fail(exp, "Test error")
      assert exp.status == :failed
      assert exp.error == "Test error"
    end
  end
end
