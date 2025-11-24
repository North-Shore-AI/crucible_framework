defmodule Crucible.Stage.Guardrails.Noop do
  @moduledoc """
  Default guardrail adapter that performs no checks.
  """

  @behaviour Crucible.Stage.Guardrails.Adapter

  @impl true
  def scan(_examples, _opts), do: {:ok, []}
end
