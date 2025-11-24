defmodule Crucible.Data.Provider do
  @moduledoc """
  Behaviour for dataset providers used by `Crucible.Stage.DataLoad`.
  """

  alias Crucible.IR.DatasetRef

  @callback load(DatasetRef.t(), map()) :: {:ok, Enumerable.t()} | {:error, term()}
end
