defmodule Crucible.Backend.Tinkex.Client do
  @moduledoc """
  Behaviour wrapper around the Tinkex SDK to enable mocking in tests.
  """

  @callback start_service(map()) :: {:ok, term()} | {:error, term()}
  @callback create_training_client(term(), map()) :: {:ok, term()} | {:error, term()}
  @callback forward_backward(term(), list(), atom(), map()) ::
              {:ok, term()} | {:ok, Task.t()} | {:error, term()}
  @callback optim_step(term(), map()) :: {:ok, term()} | {:error, term()}
  @callback save_weights_and_get_sampler(term()) :: {:ok, term()} | {:error, term()}
  @callback sample(term(), term(), map(), map()) :: {:ok, term()} | {:error, term()}
end
