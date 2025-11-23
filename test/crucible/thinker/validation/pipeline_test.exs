defmodule Crucible.Thinker.Validation.PipelineTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Validation.Pipeline

  @sample_context %{
    corpus: %{
      100 => %{
        id: 100,
        text: "Experimental results showed significant improvement in patient outcomes."
      },
      200 => %{id: 200, text: "Machine learning models achieved high accuracy on test data."}
    },
    evidence: [
      %{doc_id: 100, text: "Patients showed 40% improvement in recovery rates."}
    ],
    expected: "CLAIM[c1]: Patient outcomes improved (citing 100)"
  }

  describe "validate/2" do
    test "returns aggregate scores and claim details" do
      output = "CLAIM[c1]: Patient outcomes improved significantly (citing 100)"

      result = Pipeline.validate(output, @sample_context)

      assert Map.has_key?(result, :claims)
      assert Map.has_key?(result, :aggregate)
      assert length(result.claims) == 1
    end

    test "aggregate includes all metric types" do
      output = "CLAIM[c1]: Patient recovery improved (citing 100)"

      result = Pipeline.validate(output, @sample_context)

      assert Map.has_key?(result.aggregate, :schema_compliance)
      assert Map.has_key?(result.aggregate, :citation_accuracy)
      assert Map.has_key?(result.aggregate, :mean_entailment)
      assert Map.has_key?(result.aggregate, :mean_similarity)
    end

    test "validates multiple claims" do
      output = """
      CLAIM[c1]: Patient outcomes improved (citing 100)
      CLAIM[c2]: ML models achieved accuracy (citing 200)
      """

      result = Pipeline.validate(output, @sample_context)
      assert length(result.claims) == 2
    end

    test "handles invalid format gracefully" do
      output = "This is not a valid claim format"

      result = Pipeline.validate(output, @sample_context)
      assert result.claims == []
      assert result.aggregate.schema_compliance == 0.0
    end

    test "each claim result has all validation fields" do
      output = "CLAIM[c1]: Test claim (citing 100)"

      result = Pipeline.validate(output, @sample_context)
      [claim_result] = result.claims

      assert Map.has_key?(claim_result, :claim)
      assert Map.has_key?(claim_result, :schema_valid)
      assert Map.has_key?(claim_result, :citation_valid)
      assert Map.has_key?(claim_result, :entailment_score)
      assert Map.has_key?(claim_result, :similarity_score)
    end
  end

  describe "aggregate_scores/1" do
    test "calculates correct percentages" do
      results = [
        %{schema_valid: true, citation_valid: true, entailment_score: 0.8, similarity_score: 0.9},
        %{schema_valid: true, citation_valid: false, entailment_score: 0.6, similarity_score: 0.7}
      ]

      aggregate = Pipeline.aggregate_scores(results)

      assert aggregate.schema_compliance == 1.0
      assert aggregate.citation_accuracy == 0.5
      assert_in_delta aggregate.mean_entailment, 0.7, 0.01
      assert_in_delta aggregate.mean_similarity, 0.8, 0.01
    end
  end
end
