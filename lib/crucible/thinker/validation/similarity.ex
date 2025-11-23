defmodule Crucible.Thinker.Validation.Similarity do
  @moduledoc """
  Embedding-based similarity scoring for claims.

  Supports multiple backends:
  - `:heuristic` (default) - Jaccard similarity
  - `:bumblebee` - Local Bumblebee embeddings with MiniLM

  Configure via:
      config :crucible, :thinker_similarity_backend, :bumblebee
  """

  @behaviour Crucible.Thinker.Validation.SimilarityBehaviour

  @backends %{
    heuristic: __MODULE__,
    bumblebee: Crucible.Thinker.Validation.Similarity.Bumblebee
  }

  @doc """
  Calculates similarity between claim text and expected output.

  Returns a float between 0.0 and 1.0.

  ## Examples

      iex> claim = %{index: 1, text: "Test claim", doc_id: 100}
      iex> expected = "Test claim"
      iex> Crucible.Thinker.Validation.Similarity.score(claim, expected)
      1.0

  """
  @impl true
  @spec score(map(), String.t()) :: float()
  def score(claim, expected) do
    backend = get_backend()

    if backend == __MODULE__ do
      score_heuristic(claim, expected)
    else
      backend.score(claim, expected)
    end
  end

  defp score_heuristic(_claim, ""), do: 0.0

  defp score_heuristic(claim, expected) when is_binary(expected) do
    calculate_similarity(claim.text, expected)
  end

  @doc """
  Scores multiple claims against expected outputs.

  ## Examples

      iex> claims = [%{index: 1, text: "First", doc_id: 100}]
      iex> expected = ["First"]
      iex> [score] = Crucible.Thinker.Validation.Similarity.batch_score(claims, expected)
      iex> score == 1.0
      true

  """
  @impl true
  @spec batch_score([map()], [String.t()]) :: [float()]
  def batch_score(claims, expected) when is_list(claims) and is_list(expected) do
    backend = get_backend()

    if backend == __MODULE__ do
      claims
      |> Enum.zip(expected)
      |> Enum.map(fn {claim, exp} -> score_heuristic(claim, exp) end)
    else
      backend.batch_score(claims, expected)
    end
  end

  defp get_backend do
    backend_key = Application.get_env(:crucible, :thinker_similarity_backend, :heuristic)
    Map.get(@backends, backend_key, __MODULE__)
  end

  # Jaccard similarity with word weighting
  defp calculate_similarity(text_a, text_b) do
    words_a = tokenize(text_a)
    words_b = tokenize(text_b)

    if Enum.empty?(words_a) or Enum.empty?(words_b) do
      0.0
    else
      set_a = MapSet.new(words_a)
      set_b = MapSet.new(words_b)

      intersection = MapSet.intersection(set_a, set_b)
      union = MapSet.union(set_a, set_b)

      if MapSet.size(union) == 0 do
        0.0
      else
        MapSet.size(intersection) / MapSet.size(union)
      end
    end
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.reject(&(&1 == ""))
  end
end
