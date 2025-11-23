defmodule Crucible.Thinker.Validation.Citation do
  @moduledoc """
  Verifies citations exist in corpus and relate to claim content.

  Uses keyword overlap to determine if a claim is semantically
  related to the cited document.
  """

  @overlap_threshold 0.3

  @doc """
  Verifies that a claim's citation is valid.

  Returns true if:
  - The doc_id exists in the corpus
  - The claim text has sufficient keyword overlap with the document

  ## Examples

      iex> corpus = %{100 => %{id: 100, text: "Machine learning improves accuracy"}}
      iex> claim = %{index: 1, text: "ML improves model accuracy", doc_id: 100}
      iex> Crucible.Thinker.Validation.Citation.verify(claim, corpus)
      true

  """
  @spec verify(map(), map()) :: boolean()
  def verify(claim, corpus) do
    case Map.get(corpus, claim.doc_id) do
      nil -> false
      doc -> text_in_document?(claim.text, doc)
    end
  end

  @doc """
  Calculates the overlap score between claim and document.

  Returns a float between 0.0 and 1.0 representing the ratio
  of claim words that appear in the document.

  ## Examples

      iex> corpus = %{100 => %{id: 100, text: "Test document text"}}
      iex> claim = %{index: 1, text: "Test text", doc_id: 100}
      iex> score = Crucible.Thinker.Validation.Citation.overlap_score(claim, corpus)
      iex> score > 0.0
      true

  """
  @spec overlap_score(map(), map()) :: float()
  def overlap_score(claim, corpus) do
    case Map.get(corpus, claim.doc_id) do
      nil -> 0.0
      doc -> calculate_overlap(claim.text, doc.text)
    end
  end

  defp text_in_document?(claim_text, doc) do
    calculate_overlap(claim_text, doc.text) > @overlap_threshold
  end

  defp calculate_overlap(claim_text, doc_text) do
    claim_words = tokenize(claim_text)
    doc_words = tokenize(doc_text)

    if Enum.empty?(claim_words) do
      0.0
    else
      overlap =
        MapSet.intersection(
          MapSet.new(claim_words),
          MapSet.new(doc_words)
        )

      MapSet.size(overlap) / length(claim_words)
    end
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.reject(&(&1 == ""))
  end
end
