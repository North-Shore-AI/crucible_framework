defmodule Crucible.CNS.Adapter do
  @moduledoc """
  Behaviour describing the minimum interface needed to plug CNS into Crucible.
  """

  @callback evaluate(examples :: [map()], outputs :: list(), opts :: map()) ::
              {:ok, map()} | {:error, term()}
end
