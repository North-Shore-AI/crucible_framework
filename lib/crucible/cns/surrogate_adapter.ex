defmodule Crucible.CNS.SurrogateAdapter do
  @moduledoc """
  Behaviour for CNS surrogate topology computation (β₁ surrogates, fragility, etc.).
  """

  @type sno_id :: String.t()

  @type surrogate_result :: %{
          sno_id: sno_id(),
          beta1_surrogate: float(),
          fragility_score: float(),
          cycle_count: non_neg_integer(),
          notes: String.t() | nil
        }

  @type summary :: %{
          beta1_mean: float(),
          beta1_high_fraction: float(),
          fragility_mean: float(),
          fragility_high_fraction: float(),
          n_snos: non_neg_integer()
        }

  @callback compute_surrogates(examples :: [map()], outputs :: [map()], opts :: map()) ::
              {:ok, %{results: [surrogate_result()], summary: summary()}} | {:error, term()}
end
