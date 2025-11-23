defmodule Crucible.Thinker.CNS.Antagonist do
  @moduledoc """
  Antagonist analysis for quality issue detection.

  Identifies issues in model outputs across multiple dimensions:
  - Schema violations
  - Citation problems
  - Semantic weakness
  - Low similarity

  Part of the CNS (Chiral Narrative Synthesis) dialectical pattern:
  - Proposer generates claims (thesis)
  - Antagonist critiques claims (antithesis)
  - Synthesizer resolves issues (synthesis)
  """

  @type issue :: %{
          type: atom(),
          message: String.t(),
          severity: :low | :medium | :high | :critical
        }

  @type claim_analysis :: %{
          claim: map(),
          issues: [issue()],
          severity: :none | :low | :medium | :high | :critical,
          suggestions: [String.t()]
        }

  @doc """
  Analyzes validation results and generates issue report.

  ## Examples

      iex> validation_result = %{claims: []}
      iex> result = Crucible.Thinker.CNS.Antagonist.analyze(validation_result)
      iex> result.summary.total_claims
      0

  """
  @spec analyze(map()) :: %{claims: [claim_analysis()], summary: map()}
  def analyze(%{claims: claim_results}) do
    claim_analyses =
      claim_results
      |> Enum.map(&analyze_claim/1)

    aggregate_analysis(claim_analyses)
  end

  defp analyze_claim(claim_result) do
    issues =
      []
      |> maybe_add_schema_issue(claim_result)
      |> maybe_add_citation_issue(claim_result)
      |> maybe_add_entailment_issue(claim_result)
      |> maybe_add_similarity_issue(claim_result)

    %{
      claim: claim_result.claim,
      issues: issues,
      severity: max_severity(issues),
      suggestions: generate_suggestions(issues)
    }
  end

  # Issue detection

  defp maybe_add_schema_issue(issues, %{schema_valid: false}) do
    issue = %{
      type: :schema_violation,
      message: "Claim does not follow CLAIM[c*]: format",
      severity: :high
    }

    [issue | issues]
  end

  defp maybe_add_schema_issue(issues, _), do: issues

  defp maybe_add_citation_issue(issues, %{citation_valid: false, claim: claim}) do
    issue = %{
      type: :citation_invalid,
      message: "Citation doc_id #{claim.doc_id} not found or irrelevant",
      severity: :critical
    }

    [issue | issues]
  end

  defp maybe_add_citation_issue(issues, _), do: issues

  defp maybe_add_entailment_issue(issues, %{entailment_score: score}) when score < 0.3 do
    issue = %{
      type: :weak_entailment,
      message: "Claim not entailed by evidence (score: #{Float.round(score, 2)})",
      severity: :high
    }

    [issue | issues]
  end

  defp maybe_add_entailment_issue(issues, %{entailment_score: score}) when score < 0.5 do
    issue = %{
      type: :moderate_entailment,
      message: "Weak entailment relationship (score: #{Float.round(score, 2)})",
      severity: :medium
    }

    [issue | issues]
  end

  defp maybe_add_entailment_issue(issues, _), do: issues

  defp maybe_add_similarity_issue(issues, %{similarity_score: score}) when score < 0.5 do
    issue = %{
      type: :low_similarity,
      message: "Claim diverges from expected output (score: #{Float.round(score, 2)})",
      severity: :medium
    }

    [issue | issues]
  end

  defp maybe_add_similarity_issue(issues, _), do: issues

  # Severity calculation

  defp max_severity([]), do: :none

  defp max_severity(issues) do
    issues
    |> Enum.map(& &1.severity)
    |> Enum.max_by(&severity_rank/1)
  end

  defp severity_rank(:critical), do: 4
  defp severity_rank(:high), do: 3
  defp severity_rank(:medium), do: 2
  defp severity_rank(:low), do: 1

  # Suggestion generation

  defp generate_suggestions(issues) do
    Enum.map(issues, fn issue ->
      case issue.type do
        :schema_violation ->
          "Reformat claim to: CLAIM[cN]: <text> (citing <doc_id>)"

        :citation_invalid ->
          "Verify document exists in corpus; use valid doc_id"

        :weak_entailment ->
          "Strengthen claim-evidence relationship; cite more specific passages"

        :moderate_entailment ->
          "Consider rephrasing to better match evidence language"

        :low_similarity ->
          "Review expected output format; align claim structure"

        _ ->
          "Review claim for quality issues"
      end
    end)
  end

  # Aggregation

  defp aggregate_analysis(claim_analyses) do
    total_issues = Enum.sum(Enum.map(claim_analyses, &length(&1.issues)))

    issue_breakdown =
      claim_analyses
      |> Enum.flat_map(& &1.issues)
      |> Enum.group_by(& &1.type)
      |> Enum.map(fn {type, issues} -> {type, length(issues)} end)
      |> Map.new()

    emit_telemetry(claim_analyses, total_issues)

    %{
      claims: claim_analyses,
      summary: %{
        total_claims: length(claim_analyses),
        total_issues: total_issues,
        issue_breakdown: issue_breakdown,
        claims_with_issues: Enum.count(claim_analyses, &(&1.severity != :none)),
        overall_severity: overall_severity(claim_analyses)
      }
    }
  end

  defp overall_severity(analyses) do
    severities = analyses |> Enum.map(& &1.severity) |> Enum.reject(&(&1 == :none))

    if Enum.empty?(severities) do
      :none
    else
      Enum.max_by(severities, &severity_rank/1)
    end
  end

  defp emit_telemetry(analyses, total_issues) do
    if Code.ensure_loaded?(:telemetry) do
      :telemetry.execute(
        [:crucible, :thinker, :antagonist, :complete],
        %{
          total_claims: length(analyses),
          total_issues: total_issues,
          claims_with_issues: Enum.count(analyses, &(&1.severity != :none))
        },
        %{}
      )
    end
  end
end
