defmodule Crucible.Analysis.SurrogateAdapter do
  @moduledoc """
  Behaviour for surrogate topology/logic computation (β₁ surrogates, fragility, etc.).

  Integration apps (e.g., CNS) implement this and wire it into the framework.
  """

  @type sno_id :: String.t()

  @type result_status :: :completed | :skipped

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

  @type surrogate_payload :: %{
          required(:results) => [surrogate_result()],
          required(:summary) => summary(),
          optional(:status) => result_status(),
          optional(:message) => String.t()
        }

  @callback compute_surrogates(examples :: [map()], outputs :: [map()], opts :: map()) ::
              {:ok, surrogate_payload()} | {:error, term()}
end
