defmodule Crucible.Thinker.Validation.SimilarityBehaviour do
  @moduledoc """
  Behaviour for similarity scoring implementations.

  Allows swapping between different backends:
  - Heuristic (default, Jaccard similarity)
  - Tinkex (API-based embeddings)
  - Bumblebee (local embeddings)
  """

  @type claim :: %{index: integer(), text: String.t(), doc_id: integer()}

  @callback score(claim(), String.t()) :: float()
  @callback batch_score([claim()], [String.t()]) :: [float()]
end
