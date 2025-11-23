defmodule Crucible.Thinker.Validation.Entailment do
  @moduledoc """
  NLI-based entailment scoring for claims.

  Supports multiple backends:
  - `:heuristic` (default) - Keyword overlap heuristics
  - `:bumblebee` - Local Bumblebee inference with DeBERTa

  Configure via:
      config :crucible, :thinker_entailment_backend, :bumblebee
  """

  @behaviour Crucible.Thinker.Validation.EntailmentBehaviour

  @type evidence :: %{doc_id: integer(), text: String.t()}
  @type classification :: %{label: :entailment | :neutral | :contradiction, score: float()}

  @backends %{
    heuristic: __MODULE__,
    bumblebee: Crucible.Thinker.Validation.Entailment.Bumblebee
  }

  @doc """
  Scores how well evidence entails a claim.

  Returns a float between 0.0 and 1.0.

  ## Examples

      iex> evidence = [%{doc_id: 100, text: "The treatment improved outcomes"}]
      iex> claim = %{index: 1, text: "Treatment improved outcomes", doc_id: 100}
      iex> score = Crucible.Thinker.Validation.Entailment.score(claim, evidence)
      iex> score > 0.0
      true

  """
  @impl true
  @spec score(map(), [evidence()]) :: float()
  def score(claim, evidence) do
    backend = get_backend()

    if backend == __MODULE__ do
      score_heuristic(claim, evidence)
    else
      backend.score(claim, evidence)
    end
  end

  defp score_heuristic(_claim, []), do: 0.0

  defp score_heuristic(claim, evidence) when is_list(evidence) do
    premise = evidence_to_premise(evidence)
    calculate_entailment_score(claim.text, premise)
  end

  @doc """
  Classifies the entailment relationship.

  Returns a map with :label and :score.

  ## Examples

      iex> evidence = [%{doc_id: 100, text: "Evidence text"}]
      iex> claim = %{index: 1, text: "Claim text", doc_id: 100}
      iex> result = Crucible.Thinker.Validation.Entailment.classify(claim, evidence)
      iex> result.label in [:entailment, :neutral, :contradiction]
      true

  """
  @impl true
  @spec classify(map(), [evidence()]) :: classification()
  def classify(claim, evidence) do
    backend = get_backend()

    if backend == __MODULE__ do
      classify_heuristic(claim, evidence)
    else
      backend.classify(claim, evidence)
    end
  end

  defp classify_heuristic(claim, evidence) do
    score = score_heuristic(claim, evidence)

    label =
      cond do
        score >= 0.6 -> :entailment
        score >= 0.3 -> :neutral
        true -> :contradiction
      end

    %{label: label, score: score}
  end

  defp get_backend do
    backend_key = Application.get_env(:crucible, :thinker_entailment_backend, :heuristic)
    Map.get(@backends, backend_key, __MODULE__)
  end

  defp evidence_to_premise(evidence) do
    evidence
    |> Enum.map(& &1.text)
    |> Enum.join(" ")
  end

  # Heuristic-based entailment scoring
  # In production, this would call Tinkex API or Bumblebee
  defp calculate_entailment_score(hypothesis, premise) do
    hypothesis_words = tokenize(hypothesis)
    premise_words = tokenize(premise)

    if Enum.empty?(hypothesis_words) or Enum.empty?(premise_words) do
      0.0
    else
      hypothesis_set = MapSet.new(hypothesis_words)
      premise_set = MapSet.new(premise_words)

      overlap = MapSet.intersection(hypothesis_set, premise_set)
      overlap_ratio = MapSet.size(overlap) / MapSet.size(hypothesis_set)

      # Weight by premise coverage too
      premise_coverage = MapSet.size(overlap) / MapSet.size(premise_set)

      # Combined score
      (overlap_ratio * 0.7 + premise_coverage * 0.3)
      |> min(1.0)
    end
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.reject(&(&1 == ""))
    |> Enum.reject(&(String.length(&1) < 3))
  end
end
