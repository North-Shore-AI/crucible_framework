defmodule Crucible.IR.StatsConfig do
  @derive {Jason.Encoder, only: [:tests, :alpha, :options]}
  @moduledoc """
  Statistical testing configuration (crucible_bench integration point).
  """

  @type t :: %__MODULE__{
          tests: [atom()],
          alpha: float(),
          options: map()
        }

  defstruct tests: [],
            alpha: 0.05,
            options: %{}
end
