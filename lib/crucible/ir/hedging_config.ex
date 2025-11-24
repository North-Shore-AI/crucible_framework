defmodule Crucible.IR.HedgingConfig do
  @derive {Jason.Encoder,
           only: [:strategy, :delay_ms, :percentile, :max_extra_requests, :options]}
  @moduledoc """
  Request hedging and tail-latency configuration.
  """

  @type strategy :: :off | :fixed_delay | :percentile | :adaptive

  @type t :: %__MODULE__{
          strategy: strategy(),
          delay_ms: non_neg_integer() | nil,
          percentile: float() | nil,
          max_extra_requests: non_neg_integer(),
          options: map()
        }

  defstruct strategy: :off,
            delay_ms: nil,
            percentile: nil,
            max_extra_requests: 1,
            options: %{}
end
