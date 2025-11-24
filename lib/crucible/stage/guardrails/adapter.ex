defmodule Crucible.Stage.Guardrails.Adapter do
  @moduledoc """
  Behaviour for guardrail adapters (e.g., LlmGuard wrappers).
  """

  @callback scan([map()], map()) :: {:ok, [map()]} | {:error, term()}
end
