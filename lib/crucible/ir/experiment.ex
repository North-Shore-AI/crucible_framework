defmodule Crucible.IR.Experiment do
  @derive {Jason.Encoder,
           only: [
             :id,
             :description,
             :owner,
             :tags,
             :metadata,
             :dataset,
             :pipeline,
             :backend,
             :reliability,
             :outputs,
             :created_at,
             :updated_at
           ]}
  @moduledoc """
  Backend-agnostic experiment definition for Crucible.

  This IR is consumed by all surfaces (Elixir DSL, Python SDK, CLI) and
  intentionally avoids assumptions about infrastructure or backends.
  """

  alias Crucible.IR.{
    BackendRef,
    DatasetRef,
    OutputSpec,
    ReliabilityConfig,
    StageDef
  }

  @type t :: %__MODULE__{
          id: String.t(),
          description: String.t() | nil,
          owner: String.t() | nil,
          tags: [String.t()],
          metadata: map(),
          dataset: DatasetRef.t() | [DatasetRef.t()] | nil,
          pipeline: [StageDef.t()],
          backend: BackendRef.t(),
          reliability: ReliabilityConfig.t(),
          outputs: [OutputSpec.t()],
          created_at: DateTime.t() | nil,
          updated_at: DateTime.t() | nil
        }

  @enforce_keys [:id, :backend, :pipeline]
  defstruct [
    :id,
    :description,
    :owner,
    tags: [],
    metadata: %{},
    dataset: nil,
    pipeline: [],
    backend: nil,
    reliability: %ReliabilityConfig{},
    outputs: [],
    created_at: nil,
    updated_at: nil
  ]
end
