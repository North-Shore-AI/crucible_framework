defmodule Crucible.Thinker.HarnessTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Harness

  describe "define/1" do
    test "creates experiment with required fields" do
      experiment = Harness.define(name: "test-experiment")

      assert experiment.name == "test-experiment"
      assert experiment.id != nil
      assert experiment.status == :pending
      assert %DateTime{} = experiment.created_at
    end

    test "uses default dataset config for :scifact" do
      experiment = Harness.define(name: "test", dataset: :scifact)

      assert experiment.dataset_config.source == :scifact
      assert experiment.dataset_config.limit == 15
    end

    test "uses custom dataset config" do
      experiment =
        Harness.define(
          name: "test",
          dataset: %{source: :scifact, limit: 5, split: :validation}
        )

      assert experiment.dataset_config.limit == 5
      assert experiment.dataset_config.split == :validation
    end

    test "applies default training config" do
      experiment = Harness.define(name: "test")

      assert experiment.training_config.lora_rank == 16
      assert experiment.training_config.epochs == 3
    end

    test "merges custom training config" do
      experiment =
        Harness.define(
          name: "test",
          training: %{lora_rank: 32, epochs: 5}
        )

      assert experiment.training_config.lora_rank == 32
      assert experiment.training_config.epochs == 5
      assert experiment.training_config.base_model != nil
    end

    test "applies default validation thresholds" do
      experiment = Harness.define(name: "test")

      assert experiment.validation_config.schema_threshold == 0.95
      assert experiment.validation_config.citation_threshold == 0.95
      assert experiment.validation_config.entailment_threshold == 0.50
    end

    test "raises on missing name" do
      assert_raise KeyError, fn ->
        Harness.define([])
      end
    end
  end

  describe "run/1" do
    test "runs experiment and returns results" do
      experiment =
        Harness.define(
          name: "test-run",
          dataset: %{source: :scifact, limit: 3},
          mode: :simulate
        )

      {:ok, result} = Harness.run(experiment)

      assert result.experiment_id == experiment.id
      assert Map.has_key?(result, :training)
      assert Map.has_key?(result, :evaluation)
      assert Map.has_key?(result, :antagonist)
      assert Map.has_key?(result, :quality_check)
    end

    test "quality check evaluates thresholds" do
      experiment =
        Harness.define(
          name: "test-quality",
          dataset: %{source: :scifact, limit: 3},
          validation: %{schema_threshold: 0.0},
          mode: :simulate
        )

      {:ok, result} = Harness.run(experiment)

      assert Map.has_key?(result.quality_check, :passed)
      assert is_list(result.quality_check.details)
    end

    test "evaluation includes aggregate scores" do
      experiment =
        Harness.define(
          name: "test-eval",
          dataset: %{source: :scifact, limit: 2},
          mode: :simulate
        )

      {:ok, result} = Harness.run(experiment)

      assert Map.has_key?(result.evaluation.aggregate, :schema_compliance)
      assert Map.has_key?(result.evaluation.aggregate, :citation_accuracy)
      assert Map.has_key?(result.evaluation.aggregate, :mean_entailment)
      assert Map.has_key?(result.evaluation.aggregate, :mean_similarity)
    end
  end

  describe "report/1" do
    test "generates markdown report" do
      experiment =
        Harness.define(
          name: "test-report",
          dataset: %{source: :scifact, limit: 2},
          mode: :simulate
        )

      {:ok, result} = Harness.run(experiment)
      report = Harness.report(result)

      assert String.contains?(report, "Thinker Experiment Report")
      assert String.contains?(report, result.experiment_id)
      assert String.contains?(report, "Training Results")
      assert String.contains?(report, "Evaluation Results")
      assert String.contains?(report, "Quality Check")
    end
  end
end
