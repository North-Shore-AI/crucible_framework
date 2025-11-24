defmodule Crucible.IR.ReliabilityConfig do
  @derive {Jason.Encoder, only: [:ensemble, :hedging, :guardrails, :stats, :fairness]}
  @moduledoc """
  Configuration for reliability features:
  ensembles, hedging, guardrails, statistical testing, and fairness.
  """

  alias Crucible.IR.{
    EnsembleConfig,
    FairnessConfig,
    GuardrailConfig,
    HedgingConfig,
    StatsConfig
  }

  @type t :: %__MODULE__{
          ensemble: EnsembleConfig.t(),
          hedging: HedgingConfig.t(),
          guardrails: GuardrailConfig.t(),
          stats: StatsConfig.t(),
          fairness: FairnessConfig.t()
        }

  defstruct ensemble: %EnsembleConfig{},
            hedging: %HedgingConfig{},
            guardrails: %GuardrailConfig{},
            stats: %StatsConfig{},
            fairness: %FairnessConfig{}
end
