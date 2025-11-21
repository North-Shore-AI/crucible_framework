defmodule Crucible.Lora.Adapter do
  @moduledoc """
  Behaviour that defines the contract for LoRA fine-tuning adapters.

  An adapter connects Crucible's experiment orchestration to a concrete
  fine-tuning backend (for example, the Tinkex SDK). Implementations must
  translate the generic Crucible API into backend-specific calls while
  preserving telemetry, checkpointing, and quality validation semantics.
  """

  @typedoc "Implementation-specific experiment metadata"
  @type experiment :: map()

  @typedoc "Adapter-specific options"
  @type options :: keyword()

  @callback generate_id() :: String.t()

  @callback create_experiment(options()) :: {:ok, experiment()} | {:error, String.t()}

  @callback batch_dataset(list(), pos_integer()) :: [[any()]]

  @callback format_training_data(list(), options()) :: list()

  @callback calculate_metrics(list()) :: map()

  @callback validate_quality(map(), any()) :: map()

  @callback sampling_params(options()) :: map()

  @callback checkpoint_name(String.t(), pos_integer()) :: String.t()
end
