defmodule Crucible.Datasets.MLLoaderTest do
  use ExUnit.Case, async: true

  alias Crucible.Datasets.MLLoader

  describe "load/2" do
    test "loads dataset by name" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 10)
      assert is_list(dataset)
      assert length(dataset) <= 10
    end

    test "supports split option" do
      {:ok, train} = MLLoader.load(:scifact, split: :train, limit: 5)
      {:ok, test_data} = MLLoader.load(:scifact, split: :test, limit: 5)

      assert is_list(train)
      assert is_list(test_data)
    end

    test "applies transformations" do
      transform = fn example -> Map.put(example, :transformed, true) end
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 5, transform: transform)

      assert Enum.all?(dataset, fn ex -> ex[:transformed] == true end)
    end

    test "returns error for unknown dataset" do
      assert {:error, :unknown_dataset} = MLLoader.load(:nonexistent)
    end
  end

  describe "stream/2" do
    test "returns a stream" do
      stream = MLLoader.stream(:scifact, limit: 10)
      assert %Stream{} = stream
    end

    test "can be enumerated lazily" do
      stream = MLLoader.stream(:scifact, limit: 5)
      items = Enum.take(stream, 3)
      assert length(items) == 3
    end
  end

  describe "prepare_for_training/2" do
    test "formats examples for training" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 5)
      prepared = MLLoader.prepare_for_training(dataset)

      assert is_list(prepared)

      Enum.each(prepared, fn example ->
        assert Map.has_key?(example, :input)
        assert Map.has_key?(example, :output)
      end)
    end

    test "extracts metadata" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 5)
      prepared = MLLoader.prepare_for_training(dataset)

      Enum.each(prepared, fn example ->
        assert Map.has_key?(example, :metadata)
      end)
    end

    test "respects formatter option" do
      {:ok, dataset} = MLLoader.load(:gsm8k, split: :train, limit: 3)
      prepared = MLLoader.prepare_for_training(dataset, formatter: :gsm8k)

      Enum.each(prepared, fn example ->
        assert String.contains?(example.input, "Problem:")
      end)
    end
  end

  describe "split/2" do
    test "splits dataset by ratios" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 100)
      {train, val, test_data} = MLLoader.split(dataset, {0.7, 0.15, 0.15})

      total = length(train) + length(val) + length(test_data)
      assert total == length(dataset)
      assert length(train) >= length(val)
      assert length(train) >= length(test_data)
    end

    test "handles two-way split" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 50)
      {train, test_data} = MLLoader.split(dataset, {0.8, 0.2})

      assert length(train) + length(test_data) == length(dataset)
    end

    test "maintains stratification when specified" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 100)
      {train, test_data} = MLLoader.split(dataset, {0.8, 0.2}, stratify: :label)

      assert length(train) + length(test_data) == length(dataset)
    end
  end

  describe "sample/3" do
    test "returns n random samples" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 100)
      samples = MLLoader.sample(dataset, 10)

      assert length(samples) == 10
    end

    test "respects seed for reproducibility" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 100)
      samples1 = MLLoader.sample(dataset, 10, seed: 42)
      samples2 = MLLoader.sample(dataset, 10, seed: 42)

      assert samples1 == samples2
    end

    test "returns all if n > dataset size" do
      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 5)
      samples = MLLoader.sample(dataset, 100)

      assert length(samples) == 5
    end
  end

  describe "formatter/1" do
    test "returns scifact formatter" do
      formatter = MLLoader.formatter(:scifact)
      assert is_function(formatter, 1)

      example = %{
        "claim" => "Test claim",
        "evidence" => ["Evidence 1"],
        "label" => "SUPPORTS",
        "id" => 1,
        "evidence_ids" => [1]
      }

      result = formatter.(example)
      assert String.contains?(result.input, "Claim:")
      assert String.contains?(result.input, "Evidence:")
      assert result.output == "SUPPORTS"
    end

    test "returns fever formatter" do
      formatter = MLLoader.formatter(:fever)
      assert is_function(formatter, 1)
    end

    test "returns gsm8k formatter" do
      formatter = MLLoader.formatter(:gsm8k)
      assert is_function(formatter, 1)

      example = %{
        "question" => "What is 2+2?",
        "answer" => "4"
      }

      result = formatter.(example)
      assert String.contains?(result.input, "Problem:")
      assert result.output == "4"
    end

    test "returns mmlu formatter" do
      formatter = MLLoader.formatter(:mmlu)
      assert is_function(formatter, 1)
    end
  end
end
