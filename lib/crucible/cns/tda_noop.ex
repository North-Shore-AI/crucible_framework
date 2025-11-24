defmodule Crucible.CNS.TDANoop do
  @moduledoc """
  Default no-op TDA adapter. Used when no CNS TDA integration is configured.
  """

  @behaviour Crucible.CNS.TDAAdapter

  @impl true
  def compute_tda(_examples, _outputs, _opts) do
    {:ok,
     %{
       results: [],
       summary: %{
         beta0_mean: 0.0,
         beta1_mean: 0.0,
         beta2_mean: 0.0,
         high_loop_fraction: 0.0,
         avg_persistence: 0.0,
         n_snos: 0
       },
       status: :skipped
     }}
  end
end
