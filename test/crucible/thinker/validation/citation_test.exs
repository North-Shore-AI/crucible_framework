defmodule Crucible.Thinker.Validation.CitationTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Validation.Citation

  @sample_corpus %{
    100 => %{
      id: 100,
      text:
        "The experimental results showed significant improvement in patient outcomes with the new treatment protocol."
    },
    200 => %{
      id: 200,
      text:
        "Machine learning models can effectively classify medical images with high accuracy when trained on sufficient data."
    },
    300 => %{
      id: 300,
      text:
        "Climate change has led to increased frequency of extreme weather events across multiple continents."
    }
  }

  describe "verify/2" do
    test "returns true when claim relates to document content" do
      claim = %{
        index: 1,
        text: "Patient outcomes improved significantly with treatment",
        doc_id: 100
      }

      assert Citation.verify(claim, @sample_corpus) == true
    end

    test "returns false when doc_id not in corpus" do
      claim = %{
        index: 1,
        text: "Some claim text",
        doc_id: 999
      }

      assert Citation.verify(claim, @sample_corpus) == false
    end

    test "returns false when claim text unrelated to document" do
      claim = %{
        index: 1,
        text: "Quantum computing enables faster calculations",
        doc_id: 100
      }

      assert Citation.verify(claim, @sample_corpus) == false
    end

    test "handles keyword overlap correctly" do
      claim = %{
        index: 1,
        text: "Machine learning models classify medical images accurately",
        doc_id: 200
      }

      assert Citation.verify(claim, @sample_corpus) == true
    end

    test "is case insensitive" do
      claim = %{
        index: 1,
        text: "CLIMATE CHANGE causes EXTREME WEATHER",
        doc_id: 300
      }

      assert Citation.verify(claim, @sample_corpus) == true
    end
  end

  describe "overlap_score/2" do
    test "returns overlap ratio between claim and document" do
      claim = %{
        index: 1,
        text: "Patient outcomes improved with treatment protocol",
        doc_id: 100
      }

      score = Citation.overlap_score(claim, @sample_corpus)
      assert is_float(score)
      assert score > 0.0
      assert score <= 1.0
    end

    test "returns 0 for non-existent doc" do
      claim = %{index: 1, text: "Some text", doc_id: 999}
      assert Citation.overlap_score(claim, @sample_corpus) == 0.0
    end
  end
end
