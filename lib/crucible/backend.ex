defmodule Crucible.Backend do
  @moduledoc """
  Behaviour for training and inference backends.

  Implementations may talk to Tinkex, Nx/Axon, or external APIs.
  """

  alias Crucible.IR.Experiment

  @type backend_id :: atom()
  @type backend_config :: map()
  @type backend_state :: term()
  @type session :: term()
  @type batch :: list(map())
  @type checkpoint_ref :: term()
  @type sampler :: term()

  @callback init(backend_id, backend_config) ::
              {:ok, backend_state} | {:error, term()}

  @callback start_session(backend_state, Experiment.t()) ::
              {:ok, session} | {:error, term()}

  @callback train_step(session, batch) ::
              {:ok,
               %{
                 loss: float(),
                 batch_size: pos_integer(),
                 metrics: map()
               }}
              | {:error, term()}

  @callback save_checkpoint(session, step :: non_neg_integer()) ::
              {:ok, checkpoint_ref} | {:error, term()}

  @callback create_sampler(session, checkpoint_ref) ::
              {:ok, sampler} | {:error, term()}

  @callback sample(sampler, prompt :: binary(), opts :: map()) ::
              {:ok, [binary()]} | {:error, term()}
end
