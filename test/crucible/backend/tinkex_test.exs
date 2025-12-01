defmodule Crucible.Backend.TinkexTest do
  use ExUnit.Case, async: true

  import Mox

  alias Crucible.Backend.Tinkex
  alias CrucibleIR.{BackendRef, Experiment}

  setup :set_mox_from_context
  setup :verify_on_exit!

  setup do
    Application.put_env(:crucible_framework, :tinkex_client, Crucible.Backend.Tinkex.ClientMock)
    :ok
  end

  test "initializes and runs training through the client" do
    expect(Crucible.Backend.Tinkex.ClientMock, :start_service, fn _config -> {:ok, :service} end)

    expect(Crucible.Backend.Tinkex.ClientMock, :create_training_client, fn :service, _opts ->
      {:ok, :trainer}
    end)

    {:ok, state} = Tinkex.init(:tinkex, %{})

    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :tinkex, options: %{}},
      pipeline: []
    }

    {:ok, session} = Tinkex.start_session(state, experiment)

    expect(Crucible.Backend.Tinkex.ClientMock, :forward_backward, fn :trainer,
                                                                     [_datum],
                                                                     :cross_entropy,
                                                                     %{} ->
      {:ok, %{total_loss: 0.2, num_examples: 1}}
    end)

    assert {:ok, %{loss: 0.2, batch_size: 1}} =
             Tinkex.train_step(session, [%{model_input: :ready, loss_fn_inputs: %{}}])
  end

  test "saves checkpoints and samples via client" do
    expect(Crucible.Backend.Tinkex.ClientMock, :start_service, fn _config -> {:ok, :service} end)

    expect(Crucible.Backend.Tinkex.ClientMock, :create_training_client, fn :service, _opts ->
      {:ok, :trainer}
    end)

    {:ok, state} = Tinkex.init(:tinkex, %{})

    experiment = %Experiment{
      id: "exp",
      backend: %BackendRef{id: :tinkex, options: %{}},
      pipeline: []
    }

    {:ok, session} = Tinkex.start_session(state, experiment)

    expect(Crucible.Backend.Tinkex.ClientMock, :save_weights_and_get_sampler, fn :trainer ->
      {:ok, :sampler}
    end)

    assert {:ok, :sampler} = Tinkex.save_checkpoint(session, 1)

    expect(Crucible.Backend.Tinkex.ClientMock, :sample, fn :sampler, _prompt, _params, _opts ->
      {:ok, ["hello"]}
    end)

    assert {:ok, ["hello"]} = Tinkex.sample(:sampler, "hi", %{})
  end
end
