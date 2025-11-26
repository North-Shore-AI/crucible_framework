defmodule Crucible.Analysis.Noop do
  @moduledoc """
  Default analysis adapter that reports empty metrics.
  """

  @behaviour Crucible.Analysis.Adapter

  @impl true
  def evaluate(_examples, _outputs, _opts), do: {:ok, %{status: :skipped}}
end
