# CNS Antagonist Module

## Overview

Flags quality issues in model outputs following CNS dialectical pattern:
- **Proposer** generates claims (thesis)
- **Antagonist** critiques claims (antithesis)
- **Synthesizer** resolves issues (synthesis)

This module implements the Antagonist role.

## Module Design

```elixir
defmodule Crucible.Thinker.CNS.Antagonist do
  @moduledoc """
  Antagonist analysis for quality issue detection.

  Identifies issues across multiple dimensions:
  - Schema violations
  - Citation problems
  - Semantic weakness
  - Logical inconsistencies
  """

  alias Crucible.Thinker.Validation.Pipeline

  defstruct [
    :claim,
    :issues,
    :severity,
    :suggestions
  ]

  @type issue :: %{
    type: atom(),
    message: String.t(),
    severity: :low | :medium | :high | :critical
  }

  @doc """
  Analyze claims and generate issue report.
  """
  def analyze(validation_result) do
    validation_result.claims
    |> Enum.map(&analyze_claim/1)
    |> aggregate_analysis()
  end

  defp analyze_claim(claim_result) do
    issues = []
    |> maybe_add_schema_issue(claim_result)
    |> maybe_add_citation_issue(claim_result)
    |> maybe_add_entailment_issue(claim_result)
    |> maybe_add_similarity_issue(claim_result)

    %__MODULE__{
      claim: claim_result.claim,
      issues: issues,
      severity: max_severity(issues),
      suggestions: generate_suggestions(issues)
    }
  end

  # Issue detection

  defp maybe_add_schema_issue(issues, %{schema_valid: false} = result) do
    issue = %{
      type: :schema_violation,
      message: "Claim does not follow CLAIM[c*]: format",
      severity: :high
    }
    [issue | issues]
  end
  defp maybe_add_schema_issue(issues, _), do: issues

  defp maybe_add_citation_issue(issues, %{citation_valid: false} = result) do
    issue = %{
      type: :citation_invalid,
      message: "Citation doc_id #{result.claim.doc_id} not found or irrelevant",
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

    issue_breakdown = claim_analyses
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
    severities = Enum.map(analyses, & &1.severity) |> Enum.reject(&(&1 == :none))
    if Enum.empty?(severities), do: :none, else: Enum.max_by(severities, &severity_rank/1)
  end

  defp emit_telemetry(analyses, total_issues) do
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
```

## Report Generation

```elixir
defmodule Crucible.Thinker.CNS.Antagonist.Report do
  @moduledoc """
  Generate antagonist analysis reports.
  """

  alias Crucible.Thinker.CNS.Antagonist

  def to_markdown(analysis) do
    """
    # Antagonist Analysis Report

    ## Summary
    - **Total Claims:** #{analysis.summary.total_claims}
    - **Claims with Issues:** #{analysis.summary.claims_with_issues}
    - **Total Issues:** #{analysis.summary.total_issues}
    - **Overall Severity:** #{analysis.summary.overall_severity}

    ## Issue Breakdown
    #{format_breakdown(analysis.summary.issue_breakdown)}

    ## Claim Details
    #{format_claims(analysis.claims)}
    """
  end

  defp format_breakdown(breakdown) do
    breakdown
    |> Enum.map(fn {type, count} -> "- #{type}: #{count}" end)
    |> Enum.join("\n")
  end

  defp format_claims(claims) do
    claims
    |> Enum.map(&format_claim/1)
    |> Enum.join("\n\n")
  end

  defp format_claim(claim_analysis) do
    issues_text = if Enum.empty?(claim_analysis.issues) do
      "✓ No issues"
    else
      claim_analysis.issues
      |> Enum.map(fn issue ->
        "- [#{issue.severity}] #{issue.message}"
      end)
      |> Enum.join("\n")
    end

    suggestions_text = if Enum.empty?(claim_analysis.suggestions) do
      ""
    else
      "\n**Suggestions:**\n" <>
      (claim_analysis.suggestions |> Enum.map(&("- #{&1}")) |> Enum.join("\n"))
    end

    """
    ### Claim #{claim_analysis.claim.index}
    **Text:** #{claim_analysis.claim.text}
    **Severity:** #{claim_analysis.severity}

    #{issues_text}#{suggestions_text}
    """
  end
end
```

## Integration with crucible_bench

```elixir
defmodule Crucible.Thinker.CNS.Antagonist.Stats do
  @moduledoc """
  Statistical analysis of antagonist results using crucible_bench.
  """

  alias Crucible.Bench

  def analyze_across_experiments(experiment_results) do
    # Collect issue counts per experiment
    issue_counts = Enum.map(experiment_results, fn result ->
      result.antagonist_analysis.summary.total_issues
    end)

    # Run statistical tests
    %{
      descriptive: Bench.describe(issue_counts),
      normality: Bench.test_normality(issue_counts),
      trend: analyze_trend(issue_counts)
    }
  end

  defp analyze_trend(counts) do
    # Check if issues decrease over training
    if length(counts) < 3 do
      :insufficient_data
    else
      first_half = Enum.take(counts, div(length(counts), 2))
      second_half = Enum.drop(counts, div(length(counts), 2))

      Bench.compare(first_half, second_half, test: :mann_whitney)
    end
  end
end
```

## Usage

```elixir
alias Crucible.Thinker.Validation.Pipeline
alias Crucible.Thinker.CNS.Antagonist
alias Crucible.Thinker.CNS.Antagonist.Report

# After validation
validation_result = Pipeline.validate(output, context)

# Antagonist analysis
analysis = Antagonist.analyze(validation_result)

# Generate report
report = Report.to_markdown(analysis)
File.write!("antagonist_report.md", report)

# Check if quality meets threshold
if analysis.summary.overall_severity in [:critical, :high] do
  Logger.warning("High severity issues detected - review required")
end
```
