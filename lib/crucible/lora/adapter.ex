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

  # Extended session-based callbacks (optional)

  @doc """
  Starts a training session with the given configuration.
  """
  @callback start_session(config :: map()) :: {:ok, pid()} | {:error, term()}

  @doc """
  Executes a forward-backward pass on a batch of data.
  """
  @callback forward_backward(session :: pid(), batch :: list(), opts :: keyword()) ::
              {:ok, map()} | {:error, term()}

  @doc """
  Performs an optimizer step with the given parameters.
  """
  @callback optim_step(session :: pid(), params :: map(), opts :: keyword()) ::
              {:ok, map()} | {:error, term()}

  @doc """
  Creates a sampling client from a checkpoint for inference.
  """
  @callback create_sampler(session :: pid(), checkpoint_name :: String.t()) ::
              {:ok, pid()} | {:error, term()}

  @doc """
  Generates samples from a prompt using the sampling client.
  """
  @callback sample(session :: pid(), prompt :: String.t(), opts :: keyword()) ::
              {:ok, list(String.t())} | {:error, term()}

  @optional_callbacks [
    start_session: 1,
    forward_backward: 3,
    optim_step: 3,
    create_sampler: 2,
    sample: 3
  ]
end
