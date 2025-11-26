defmodule Crucible.Analysis.Adapter do
  @moduledoc """
  Behaviour for plugging analysis/logic metrics into Crucible pipelines.

  CNS implementations should implement this behaviour in integration apps;
  Crucible core remains domain-agnostic.
  """

  @callback evaluate(examples :: [map()], outputs :: list(), opts :: map()) ::
              {:ok, map()} | {:error, term()}
end
