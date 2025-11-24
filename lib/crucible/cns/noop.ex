defmodule Crucible.CNS.Noop do
  @moduledoc """
  Default CNS adapter that reports empty metrics.
  """

  @behaviour Crucible.CNS.Adapter

  @impl true
  def evaluate(_examples, _outputs, _opts), do: {:ok, %{status: :skipped}}
end
