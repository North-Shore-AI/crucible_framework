defmodule Crucible.IR.BackendRef do
  @derive {Jason.Encoder, only: [:id, :profile, :options]}
  @moduledoc """
  Logical reference to a training/inference backend.

  `id` is resolved to a module implementing `Crucible.Backend` via configuration.
  """

  @type t :: %__MODULE__{
          id: atom(),
          profile: atom() | String.t() | nil,
          options: map()
        }

  @enforce_keys [:id]
  defstruct [
    :id,
    :profile,
    options: %{}
  ]
end
