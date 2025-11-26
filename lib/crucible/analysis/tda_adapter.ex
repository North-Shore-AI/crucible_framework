defmodule Crucible.Analysis.TDAAdapter do
  @moduledoc """
  Behaviour for plugging topological data analysis into Crucible.

  Implementations live in integration apps (e.g., `cns_crucible`) and are
  responsible for turning experiment examples/outputs into structures suitable
  for persistent homology and related metrics.
  """

  @type sno_id :: String.t()
  @type barcode :: %{birth: float(), death: float(), persistence: float()}
  @type beta_index :: non_neg_integer()

  @type result_status :: :completed | :skipped

  @type tda_result :: %{
          sno_id: sno_id(),
          betti: %{beta_index() => non_neg_integer()},
          diagrams: %{beta_index() => [barcode()]},
          summary: %{
            total_features: non_neg_integer(),
            max_persistence: float(),
            mean_persistence: float(),
            persistent_cycle_ratio: float(),
            notes: String.t() | nil
          }
        }

  @type summary :: %{
          beta0_mean: float(),
          beta1_mean: float(),
          beta2_mean: float(),
          high_loop_fraction: float(),
          avg_persistence: float(),
          n_snos: non_neg_integer()
        }

  @type tda_payload :: %{
          required(:results) => [tda_result()],
          required(:summary) => summary(),
          optional(:status) => result_status(),
          optional(:message) => String.t()
        }

  @callback compute_tda(examples :: [map()], outputs :: [map()], opts :: map()) ::
              {:ok, tda_payload()} | {:error, term()}
end
