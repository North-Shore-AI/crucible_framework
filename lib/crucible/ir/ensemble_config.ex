defmodule Crucible.IR.EnsembleConfig do
  @derive {Jason.Encoder, only: [:strategy, :members, :options]}
  @moduledoc """
  Ensemble configuration for multi-model voting and blending.
  """

  alias Crucible.IR.BackendRef

  @type strategy ::
          :none
          | :majority_vote
          | :weighted_vote
          | :best_confidence
          | :custom

  @type t :: %__MODULE__{
          strategy: strategy(),
          members: [BackendRef.t()],
          options: map()
        }

  defstruct strategy: :none,
            members: [],
            options: %{}
end
