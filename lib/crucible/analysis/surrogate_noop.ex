defmodule Crucible.Analysis.SurrogateNoop do
  @moduledoc "Default no-op surrogate adapter."

  @behaviour Crucible.Analysis.SurrogateAdapter

  @impl true
  def compute_surrogates(_examples, _outputs, _opts) do
    {:ok,
     %{
       results: [],
       summary: %{
         beta1_mean: 0.0,
         beta1_high_fraction: 0.0,
         fragility_mean: 0.0,
         fragility_high_fraction: 0.0,
         n_snos: 0
       },
       status: :skipped
     }}
  end
end
