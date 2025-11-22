defmodule Crucible.TinkexTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex
  alias Crucible.Tinkex.Config

  doctest Tinkex

  describe "generate_id/0" do
    test "generates unique IDs" do
      id1 = Tinkex.generate_id()
      id2 = Tinkex.generate_id()

      assert is_binary(id1)
      assert is_binary(id2)
      assert id1 != id2
      assert String.length(id1) == 16
    end
  end

  describe "create_experiment/1" do
    test "creates experiment with generated ID" do
      {:ok, experiment} = Tinkex.create_experiment(name: "Test Experiment")

      refute is_nil(experiment.id)
      assert experiment.name == "Test Experiment"
      assert experiment.status == :pending
      refute is_nil(experiment.created_at)
    end

    test "creates experiment with custom config" do
      config = Config.new(api_key: "test", default_lora_rank: 32)
      {:ok, experiment} = Tinkex.create_experiment(name: "Test", config: config)

      assert experiment.config.default_lora_rank == 32
    end

    test "returns error for invalid experiment" do
      assert {:error, _reason} = Tinkex.create_experiment([])
    end
  end

  describe "Session struct" do
    test "creates a valid session" do
      session = %Tinkex.Session{
        experiment_id: "exp-123",
        config: Config.new(api_key: "test")
      }

      assert session.experiment_id == "exp-123"
      assert session.service_client == nil
      assert session.training_client == nil
    end
  end

  describe "TrainingMetrics struct" do
    test "creates training metrics" do
      metrics = %Tinkex.TrainingMetrics{
        step: 100,
        epoch: 2,
        loss: 0.5,
        citation_invalid_rate: 0.0
      }

      assert metrics.step == 100
      assert metrics.loss == 0.5
    end
  end

  describe "EvaluationResult struct" do
    test "creates evaluation result" do
      result = %Tinkex.EvaluationResult{
        experiment_id: "exp-123",
        adapter_name: "test-adapter",
        metrics: %{accuracy: 0.95},
        samples: 100,
        evaluated_at: DateTime.utc_now()
      }

      assert result.experiment_id == "exp-123"
      assert result.metrics.accuracy == 0.95
    end
  end

  describe "batch_dataset/2" do
    test "batches dataset into chunks" do
      dataset = Enum.to_list(1..100)
      batches = Tinkex.batch_dataset(dataset, 10)

      assert length(batches) == 10
      assert hd(batches) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    end

    test "handles uneven batches" do
      dataset = Enum.to_list(1..25)
      batches = Tinkex.batch_dataset(dataset, 10)

      assert length(batches) == 3
      assert length(List.last(batches)) == 5
    end

    test "handles empty dataset" do
      batches = Tinkex.batch_dataset([], 10)
      assert batches == []
    end
  end

  describe "format_training_data/2" do
    test "formats data with default loss function" do
      batch = [
        %{input: "test input", output: "test output"}
      ]

      formatted = Tinkex.format_training_data(batch)

      assert is_list(formatted)
      assert length(formatted) == 1
    end

    test "formats data with custom weights" do
      batch = [
        %{input: "test", output: "output", weight: 2.0}
      ]

      formatted = Tinkex.format_training_data(batch, citation_validity_weight: 5.0)

      assert is_list(formatted)
    end
  end

  describe "calculate_metrics/1" do
    test "calculates basic statistics from results" do
      results = [
        %{loss: 1.0, citation_invalid_rate: 0.1},
        %{loss: 0.8, citation_invalid_rate: 0.05},
        %{loss: 0.6, citation_invalid_rate: 0.0}
      ]

      metrics = Tinkex.calculate_metrics(results)

      assert_in_delta metrics.mean_loss, 0.8, 0.01
      assert_in_delta metrics.loss_reduction, 0.4, 0.01
      assert_in_delta metrics.mean_citation_invalid_rate, 0.05, 0.01
    end

    test "handles empty results" do
      metrics = Tinkex.calculate_metrics([])

      assert metrics.mean_loss == nil
      assert metrics.loss_reduction == nil
    end
  end
end
