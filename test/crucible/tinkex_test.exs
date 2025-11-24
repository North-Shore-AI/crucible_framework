defmodule Crucible.TinkexTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex

  describe "behaviour compliance" do
    test "implements Lora.Adapter behaviour" do
      behaviours = Tinkex.__info__(:attributes)[:behaviour] || []
      assert Crucible.Lora.Adapter in behaviours
    end
  end

  describe "generate_id/0" do
    test "returns unique string identifier" do
      id1 = Tinkex.generate_id()
      id2 = Tinkex.generate_id()

      assert is_binary(id1)
      assert is_binary(id2)
      assert id1 != id2
    end
  end

  describe "create_experiment/1" do
    test "creates experiment with valid config" do
      opts = [
        base_model: "test-model",
        lora_rank: 8,
        lora_alpha: 16,
        target_modules: ["q_proj"]
      ]

      assert {:ok, experiment} = Tinkex.create_experiment(opts)
      assert Map.has_key?(experiment, :id)
      assert experiment.backend == :tinkex
    end

    test "returns error for missing base_model" do
      opts = [lora_rank: 8]
      assert {:error, _} = Tinkex.create_experiment(opts)
    end
  end

  describe "batch_dataset/2" do
    test "splits dataset into batches" do
      dataset = [1, 2, 3, 4, 5]
      batches = Tinkex.batch_dataset(dataset, 2)

      assert batches == [[1, 2], [3, 4], [5]]
    end
  end

  describe "format_training_data/2" do
    test "formats batch for Tinkex training" do
      batch = [
        %{input: "prompt1", output: "response1"},
        %{input: "prompt2", output: "response2"}
      ]

      formatted = Tinkex.format_training_data(batch, [])

      assert length(formatted) == 2
      assert Enum.all?(formatted, &Map.has_key?(&1, :formatted))
    end
  end

  describe "calculate_metrics/1" do
    test "aggregates training metrics" do
      results = [
        %{loss: 1.0, accuracy: 0.8},
        %{loss: 0.8, accuracy: 0.9}
      ]

      metrics = Tinkex.calculate_metrics(results)

      assert Map.has_key?(metrics, :mean_loss)
      assert Map.has_key?(metrics, :mean_accuracy)
    end
  end

  describe "validate_quality/2" do
    test "validates results against quality targets" do
      results = %{accuracy: 0.95, loss: 0.1}
      config = %{min_accuracy: 0.9}

      validation = Tinkex.validate_quality(results, config)

      assert Map.has_key?(validation, :passed)
    end
  end

  describe "sampling_params/1" do
    test "builds sampling parameters map" do
      params = Tinkex.sampling_params(temperature: 0.8, max_tokens: 256)

      assert params.temperature == 0.8
      assert params.max_tokens == 256
    end

    test "provides defaults for missing options" do
      params = Tinkex.sampling_params([])

      assert Map.has_key?(params, :temperature)
      assert Map.has_key?(params, :max_tokens)
    end
  end

  describe "checkpoint_name/2" do
    test "generates checkpoint name" do
      name = Tinkex.checkpoint_name("exp-123", 100)

      assert is_binary(name)
      assert String.contains?(name, "exp-123")
      assert String.contains?(name, "100")
    end
  end
end
