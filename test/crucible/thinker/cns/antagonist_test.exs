defmodule Crucible.Thinker.CNS.AntagonistTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.CNS.Antagonist

  describe "analyze/1" do
    test "returns analysis with claims and summary" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test claim", doc_id: 100},
            schema_valid: true,
            citation_valid: true,
            entailment_score: 0.8,
            similarity_score: 0.9
          }
        ]
      }

      result = Antagonist.analyze(validation_result)

      assert Map.has_key?(result, :claims)
      assert Map.has_key?(result, :summary)
    end

    test "detects schema violation" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 100},
            schema_valid: false,
            citation_valid: true,
            entailment_score: 0.8,
            similarity_score: 0.9
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      issue_types = Enum.map(claim_analysis.issues, & &1.type)
      assert :schema_violation in issue_types
    end

    test "detects citation invalid" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 999},
            schema_valid: true,
            citation_valid: false,
            entailment_score: 0.8,
            similarity_score: 0.9
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      issue_types = Enum.map(claim_analysis.issues, & &1.type)
      assert :citation_invalid in issue_types
    end

    test "detects weak entailment" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 100},
            schema_valid: true,
            citation_valid: true,
            entailment_score: 0.2,
            similarity_score: 0.9
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      issue_types = Enum.map(claim_analysis.issues, & &1.type)
      assert :weak_entailment in issue_types
    end

    test "detects low similarity" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 100},
            schema_valid: true,
            citation_valid: true,
            entailment_score: 0.8,
            similarity_score: 0.3
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      issue_types = Enum.map(claim_analysis.issues, & &1.type)
      assert :low_similarity in issue_types
    end

    test "calculates severity correctly" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 999},
            schema_valid: false,
            citation_valid: false,
            entailment_score: 0.1,
            similarity_score: 0.2
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      assert claim_analysis.severity == :critical
    end

    test "generates suggestions for issues" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 100},
            schema_valid: false,
            citation_valid: true,
            entailment_score: 0.8,
            similarity_score: 0.9
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      assert length(claim_analysis.suggestions) > 0
    end

    test "summary includes total counts" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test1", doc_id: 100},
            schema_valid: true,
            citation_valid: true,
            entailment_score: 0.8,
            similarity_score: 0.9
          },
          %{
            claim: %{index: 2, text: "Test2", doc_id: 200},
            schema_valid: false,
            citation_valid: false,
            entailment_score: 0.2,
            similarity_score: 0.3
          }
        ]
      }

      result = Antagonist.analyze(validation_result)

      assert result.summary.total_claims == 2
      assert result.summary.claims_with_issues == 1
      assert result.summary.total_issues > 0
    end

    test "no issues for perfect claims" do
      validation_result = %{
        claims: [
          %{
            claim: %{index: 1, text: "Test", doc_id: 100},
            schema_valid: true,
            citation_valid: true,
            entailment_score: 0.9,
            similarity_score: 0.95
          }
        ]
      }

      result = Antagonist.analyze(validation_result)
      [claim_analysis] = result.claims

      assert claim_analysis.issues == []
      assert claim_analysis.severity == :none
    end
  end
end
