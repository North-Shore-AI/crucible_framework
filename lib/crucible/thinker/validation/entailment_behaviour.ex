defmodule Crucible.Thinker.Validation.EntailmentBehaviour do
  @moduledoc """
  Behaviour for entailment scoring implementations.

  Allows swapping between different backends:
  - Heuristic (default, no external deps)
  - Tinkex (API-based)
  - Bumblebee (local inference)
  """

  @type evidence :: %{doc_id: integer(), text: String.t()}
  @type claim :: %{index: integer(), text: String.t(), doc_id: integer()}

  @callback score(claim(), [evidence()]) :: float()
  @callback classify(claim(), [evidence()]) :: %{label: atom(), score: float()}
end
