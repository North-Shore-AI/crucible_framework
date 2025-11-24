defmodule Crucible.Context do
  @moduledoc """
  Runtime context threaded through experiment stages.

  Stages mutate the context to add data, telemetry, metrics, or artifacts.
  """

  alias Crucible.IR.Experiment

  @type t :: %__MODULE__{
          experiment_id: String.t(),
          run_id: String.t(),
          experiment: Experiment.t(),
          dataset: term() | nil,
          batches: Enumerable.t() | nil,
          examples: list() | nil,
          backend_sessions: %{atom() => term()},
          backend_state: map(),
          outputs: list(),
          metrics: map(),
          artifacts: map(),
          trace: term() | nil,
          telemetry_context: map(),
          assigns: map()
        }

  @enforce_keys [:experiment_id, :run_id, :experiment]
  defstruct [
    :experiment_id,
    :run_id,
    :experiment,
    dataset: nil,
    batches: nil,
    examples: nil,
    backend_sessions: %{},
    backend_state: %{},
    outputs: [],
    metrics: %{},
    artifacts: %{},
    trace: nil,
    telemetry_context: %{},
    assigns: %{}
  ]
end
