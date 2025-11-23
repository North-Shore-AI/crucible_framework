defmodule Crucible.Thinker.Validation.Pipeline do
  @moduledoc """
  Orchestrates 4-stage semantic validation pipeline.

  Stages:
  1. Schema - Validates CLAIM[c*] format
  2. Citation - Verifies doc_id exists and relates to claim
  3. Entailment - Scores NLI relationship with evidence
  4. Similarity - Scores similarity to expected output
  """

  alias Crucible.Thinker.Validation.{Schema, Citation, Entailment, Similarity}

  @type validation_result :: %{
          claims: [map()],
          aggregate: map()
        }

  @doc """
  Runs full validation pipeline on model output.

  ## Examples

      iex> context = %{corpus: %{}, evidence: [], expected: ""}
      iex> result = Crucible.Thinker.Validation.Pipeline.validate("", context)
      iex> result.aggregate.schema_compliance
      0.0

  """
  @spec validate(String.t(), map()) :: validation_result()
  def validate(output, context) do
    claims = Schema.parse(output)

    results =
      Enum.map(claims, fn claim ->
        %{
          claim: claim,
          schema_valid: Schema.validate(claim),
          citation_valid: Citation.verify(claim, context.corpus),
          entailment_score: Entailment.score(claim, context.evidence),
          similarity_score: Similarity.score(claim, context.expected)
        }
      end)

    emit_telemetry(results)

    %{
      claims: results,
      aggregate: aggregate_scores(results)
    }
  end

  @doc """
  Calculates aggregate scores from validation results.

  ## Examples

      iex> results = [%{schema_valid: true, citation_valid: true, entailment_score: 0.8, similarity_score: 0.9}]
      iex> agg = Crucible.Thinker.Validation.Pipeline.aggregate_scores(results)
      iex> agg.schema_compliance
      1.0

  """
  @spec aggregate_scores([map()]) :: map()
  def aggregate_scores([]) do
    %{
      schema_compliance: 0.0,
      citation_accuracy: 0.0,
      mean_entailment: 0.0,
      mean_similarity: 0.0
    }
  end

  def aggregate_scores(results) do
    count = length(results)

    schema_valid_count = Enum.count(results, & &1.schema_valid)
    citation_valid_count = Enum.count(results, & &1.citation_valid)
    total_entailment = Enum.sum(Enum.map(results, & &1.entailment_score))
    total_similarity = Enum.sum(Enum.map(results, & &1.similarity_score))

    %{
      schema_compliance: schema_valid_count / count,
      citation_accuracy: citation_valid_count / count,
      mean_entailment: total_entailment / count,
      mean_similarity: total_similarity / count
    }
  end

  defp emit_telemetry(results) do
    if Code.ensure_loaded?(:telemetry) do
      :telemetry.execute(
        [:crucible, :thinker, :validation, :complete],
        aggregate_scores(results),
        %{claim_count: length(results)}
      )
    end
  end
end
