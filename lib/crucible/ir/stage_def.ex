defmodule Crucible.IR.StageDef do
  @derive {Jason.Encoder, only: [:name, :module, :options]}
  @moduledoc """
  A single stage in an experiment pipeline.
  """

  @type t :: %__MODULE__{
          name: atom(),
          module: module() | nil,
          options: map()
        }

  @enforce_keys [:name]
  defstruct [
    :name,
    :module,
    options: %{}
  ]
end
