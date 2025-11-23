defmodule Crucible.Thinker.Datasets.ScifactTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Datasets.Scifact

  describe "load/1" do
    test "loads dataset with default options" do
      # Uses mock/sample data for testing
      {:ok, dataset} = Scifact.load()
      assert is_list(dataset)
    end

    test "respects limit option" do
      {:ok, dataset} = Scifact.load(limit: 5)
      assert length(dataset) <= 5
    end

    test "loads specific split" do
      {:ok, dataset} = Scifact.load(split: :train)
      assert is_list(dataset)
    end

    test "each sample has required fields" do
      {:ok, [sample | _]} = Scifact.load(limit: 1)

      assert Map.has_key?(sample, :id)
      assert Map.has_key?(sample, :claim)
      assert Map.has_key?(sample, :evidence)
      assert Map.has_key?(sample, :cited_doc_ids)
    end
  end

  describe "format_for_training/1" do
    test "formats sample with input and output" do
      sample = %{
        id: 1,
        claim: "Test claim about science",
        evidence: [
          %{doc_id: 100, text: "Evidence text here"}
        ],
        cited_doc_ids: [100]
      }

      formatted = Scifact.format_for_training(sample)

      assert Map.has_key?(formatted, :input)
      assert Map.has_key?(formatted, :output)
      assert String.contains?(formatted.input, "Test claim")
    end

    test "builds output in CLAIM format" do
      sample = %{
        id: 1,
        claim: "Test claim",
        evidence: [
          %{doc_id: 100, text: "First evidence"},
          %{doc_id: 200, text: "Second evidence"}
        ],
        cited_doc_ids: [100, 200]
      }

      formatted = Scifact.format_for_training(sample)

      assert String.contains?(formatted.output, "CLAIM[c")
      assert String.contains?(formatted.output, "citing")
    end
  end

  describe "sample_data/0" do
    test "returns valid sample dataset" do
      samples = Scifact.sample_data()
      assert is_list(samples)
      assert length(samples) > 0
    end
  end
end
