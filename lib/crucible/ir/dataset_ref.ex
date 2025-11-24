defmodule Crucible.IR.DatasetRef do
  @derive {Jason.Encoder, only: [:provider, :name, :split, :options]}
  @moduledoc """
  Logical reference to a dataset used in an experiment.

  Providers map to loader implementations; the engine does not hard-code them.
  """

  @type split :: :train | :test | :validation | :all | String.t()

  @type t :: %__MODULE__{
          provider: module() | atom() | nil,
          name: String.t(),
          split: split(),
          options: map()
        }

  @enforce_keys [:name]
  defstruct [
    :provider,
    :name,
    split: :train,
    options: %{}
  ]
end
