defmodule Crucible.Thinker.Validation.EntailmentTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Validation.Entailment

  @sample_evidence [
    %{
      doc_id: 100,
      text:
        "The clinical trial demonstrated that patients receiving the new treatment showed 40% improvement in recovery rates compared to the control group."
    },
    %{
      doc_id: 100,
      text: "Side effects were minimal and comparable between treatment and control groups."
    }
  ]

  describe "score/2" do
    test "returns a float between 0 and 1" do
      claim = %{
        index: 1,
        text: "The new treatment improved patient recovery rates",
        doc_id: 100
      }

      score = Entailment.score(claim, @sample_evidence)
      assert is_float(score)
      assert score >= 0.0
      assert score <= 1.0
    end

    test "returns higher score for entailed claims" do
      entailed_claim = %{
        index: 1,
        text: "Patients showed improved recovery with the treatment",
        doc_id: 100
      }

      unrelated_claim = %{
        index: 2,
        text: "The economy grew significantly last quarter",
        doc_id: 100
      }

      entailed_score = Entailment.score(entailed_claim, @sample_evidence)
      unrelated_score = Entailment.score(unrelated_claim, @sample_evidence)

      assert entailed_score > unrelated_score
    end

    test "handles empty evidence" do
      claim = %{index: 1, text: "Some claim", doc_id: 100}
      score = Entailment.score(claim, [])
      assert score == 0.0
    end
  end

  describe "classify/2" do
    test "returns entailment classification" do
      claim = %{
        index: 1,
        text: "Treatment improved recovery rates",
        doc_id: 100
      }

      result = Entailment.classify(claim, @sample_evidence)
      assert result.label in [:entailment, :neutral, :contradiction]
      assert is_float(result.score)
    end
  end
end
