defmodule Crucible.IR.FairnessConfig do
  @derive {Jason.Encoder, only: [:enabled, :group_by, :metrics, :options]}
  @moduledoc """
  Fairness evaluation configuration (ExFairness integration point).
  """

  @type t :: %__MODULE__{
          enabled: boolean(),
          group_by: atom() | String.t() | nil,
          metrics: [atom()],
          options: map()
        }

  defstruct enabled: false,
            group_by: nil,
            metrics: [],
            options: %{}
end
