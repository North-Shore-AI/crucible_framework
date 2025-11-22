defmodule Crucible.Lora.TrainerTest do
  use ExUnit.Case, async: true

  alias Crucible.Lora.{Trainer, Config}

  # Mock session for testing
  defmodule MockSession do
    use GenServer

    def start_link(opts \\ []) do
      GenServer.start_link(__MODULE__, opts)
    end

    @impl true
    def init(opts) do
      {:ok,
       %{
         steps: [],
         checkpoints: [],
         fail_at_step: Keyword.get(opts, :fail_at_step)
       }}
    end

    @impl true
    def handle_call({:forward_backward, _data}, _from, state) do
      {:reply, {:ok, %{loss: 0.5, gradients: %{}}}, state}
    end

    @impl true
    def handle_call({:optim_step, _params}, _from, state) do
      {:reply, {:ok, %{grad_norm: 1.0}}, state}
    end

    @impl true
    def handle_call({:save_checkpoint, name}, _from, state) do
      checkpoints = [name | state.checkpoints]
      {:reply, {:ok, name}, %{state | checkpoints: checkpoints}}
    end

    @impl true
    def handle_call(:get_checkpoints, _from, state) do
      {:reply, state.checkpoints, state}
    end
  end

  setup do
    {:ok, session} = MockSession.start_link()
    config = Config.new(epochs: 2, batch_size: 2, checkpoint_interval: 2)

    %{session: session, config: config}
  end

  describe "start_link/1" do
    test "starts trainer GenServer with config", %{config: config} do
      {:ok, trainer} = Trainer.start_link(config)
      assert is_pid(trainer)
      Trainer.stop(trainer)
    end
  end

  describe "train/3" do
    test "runs training loop for specified epochs", %{session: session, config: config} do
      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"},
        %{input: "test3", output: "out3"},
        %{input: "test4", output: "out4"}
      ]

      {:ok, result} = Trainer.train(trainer, dataset, session: session)

      # 4 items / batch_size 2 = 2 batches per epoch * 2 epochs = 4 steps
      assert result.total_steps == 4
      assert result.epochs_completed == 2

      Trainer.stop(trainer)
    end

    test "checkpoints at specified intervals", %{session: session, config: config} do
      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"},
        %{input: "test3", output: "out3"},
        %{input: "test4", output: "out4"}
      ]

      {:ok, result} = Trainer.train(trainer, dataset, session: session)

      # checkpoint_interval is 2, so checkpoints at steps 2 and 4
      assert length(result.checkpoints) == 2

      Trainer.stop(trainer)
    end

    test "returns final metrics", %{session: session, config: config} do
      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"}
      ]

      {:ok, result} = Trainer.train(trainer, dataset, session: session)

      assert is_map(result.metrics)
      assert Map.has_key?(result.metrics, :avg_loss)
      assert Map.has_key?(result.metrics, :final_loss)

      Trainer.stop(trainer)
    end

    test "emits telemetry events", %{session: session, config: config} do
      test_pid = self()

      :telemetry.attach(
        "test-step",
        [:crucible, :training, :step],
        fn _event, measurements, metadata, _ ->
          send(test_pid, {:step_event, measurements, metadata})
        end,
        nil
      )

      :telemetry.attach(
        "test-epoch",
        [:crucible, :training, :epoch],
        fn _event, measurements, metadata, _ ->
          send(test_pid, {:epoch_event, measurements, metadata})
        end,
        nil
      )

      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"}
      ]

      {:ok, _result} = Trainer.train(trainer, dataset, session: session)

      # Should receive step events
      assert_receive {:step_event, _, _}, 1000
      # Should receive epoch events
      assert_receive {:epoch_event, _, _}, 1000

      :telemetry.detach("test-step")
      :telemetry.detach("test-epoch")
      Trainer.stop(trainer)
    end

    test "invokes callbacks", %{session: session, config: config} do
      test_pid = self()

      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"}
      ]

      callbacks = %{
        on_step_end: fn info -> send(test_pid, {:step_end, info}) end,
        on_epoch_end: fn info -> send(test_pid, {:epoch_end, info}) end,
        on_checkpoint: fn info -> send(test_pid, {:checkpoint, info}) end
      }

      {:ok, _result} =
        Trainer.train(trainer, dataset,
          session: session,
          callbacks: callbacks
        )

      assert_receive {:step_end, _}, 1000
      assert_receive {:epoch_end, _}, 1000

      Trainer.stop(trainer)
    end
  end

  describe "save_checkpoint/2" do
    test "saves checkpoint with given name", %{config: config} do
      {:ok, trainer} = Trainer.start_link(config)

      {:ok, name} = Trainer.save_checkpoint(trainer, "test-checkpoint")
      assert name == "test-checkpoint"

      Trainer.stop(trainer)
    end
  end

  describe "get_metrics/1" do
    test "returns current metrics", %{session: session, config: config} do
      {:ok, trainer} = Trainer.start_link(config)

      dataset = [
        %{input: "test1", output: "out1"},
        %{input: "test2", output: "out2"}
      ]

      {:ok, _} = Trainer.train(trainer, dataset, session: session)

      metrics = Trainer.get_metrics(trainer)
      assert is_map(metrics)

      Trainer.stop(trainer)
    end
  end
end
