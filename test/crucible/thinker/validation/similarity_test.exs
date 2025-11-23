defmodule Crucible.Thinker.Validation.SimilarityTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Validation.Similarity

  describe "score/2" do
    test "returns 1.0 for identical texts" do
      claim = %{index: 1, text: "The treatment was effective", doc_id: 100}
      expected = "The treatment was effective"

      score = Similarity.score(claim, expected)
      assert_in_delta score, 1.0, 0.01
    end

    test "returns high score for similar texts" do
      claim = %{index: 1, text: "The treatment was very effective", doc_id: 100}
      expected = "The treatment was effective"

      score = Similarity.score(claim, expected)
      assert score > 0.5
    end

    test "returns low score for dissimilar texts" do
      claim = %{index: 1, text: "The weather is sunny today", doc_id: 100}
      expected = "Machine learning requires large datasets"

      score = Similarity.score(claim, expected)
      assert score < 0.3
    end

    test "returns float between 0 and 1" do
      claim = %{index: 1, text: "Some random text", doc_id: 100}
      expected = "Different random text"

      score = Similarity.score(claim, expected)
      assert is_float(score)
      assert score >= 0.0
      assert score <= 1.0
    end

    test "handles empty expected text" do
      claim = %{index: 1, text: "Some text", doc_id: 100}
      score = Similarity.score(claim, "")
      assert score == 0.0
    end
  end

  describe "batch_score/2" do
    test "scores multiple claims against expected" do
      claims = [
        %{index: 1, text: "First claim text", doc_id: 100},
        %{index: 2, text: "Second claim text", doc_id: 200}
      ]

      expected = ["First claim text", "Different text entirely"]

      scores = Similarity.batch_score(claims, expected)
      assert length(scores) == 2
      assert Enum.all?(scores, &is_float/1)
    end
  end
end
