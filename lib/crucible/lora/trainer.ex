defmodule Crucible.Lora.Trainer do
  @moduledoc """
  High-level LoRA training orchestration.

  Provides a GenServer-based trainer that manages the training loop, checkpointing,
  metrics collection, and telemetry emission.

  ## Example

      config = Crucible.Lora.Config.new(epochs: 3, batch_size: 8)
      {:ok, trainer} = Crucible.Lora.Trainer.start_link(config)

      dataset = [
        %{input: "...", output: "..."},
        ...
      ]

      {:ok, result} = Crucible.Lora.Trainer.train(trainer, dataset,
        session: session,
        callbacks: %{
          on_step_end: fn info -> IO.inspect(info.loss) end
        }
      )

  """

  use GenServer
  require Logger

  alias Crucible.Lora.{Config, GradientHooks}

  defstruct [
    :config,
    :session,
    :dataset,
    :current_step,
    :current_epoch,
    :metrics_buffer,
    :checkpoints,
    :status,
    :gradient_hooks,
    :callbacks
  ]

  @type t :: %__MODULE__{
          config: Config.t(),
          session: pid() | nil,
          dataset: list() | nil,
          current_step: non_neg_integer(),
          current_epoch: non_neg_integer(),
          metrics_buffer: list(),
          checkpoints: list(),
          status: :idle | :training | :stopped,
          gradient_hooks: pid() | nil,
          callbacks: map()
        }

  # Client API

  @doc """
  Starts a new trainer with the given configuration.

  ## Examples

      config = Crucible.Lora.Config.new(epochs: 5)
      {:ok, trainer} = Crucible.Lora.Trainer.start_link(config)

  """
  @spec start_link(Config.t()) :: GenServer.on_start()
  def start_link(%Config{} = config) do
    GenServer.start_link(__MODULE__, config)
  end

  @doc """
  Runs training on the given dataset.

  ## Options

    * `:session` - Session pid for forward/backward passes (required)
    * `:callbacks` - Map of callback functions
      * `:on_step_end` - Called after each training step
      * `:on_epoch_end` - Called after each epoch
      * `:on_checkpoint` - Called when checkpoint is saved
    * `:gradient_hooks` - List of gradient hook specs

  ## Returns

    * `{:ok, result}` with training metrics and checkpoints

  ## Examples

      {:ok, result} = Trainer.train(trainer, dataset,
        session: session,
        callbacks: %{on_step_end: &handle_step/1}
      )

  """
  @spec train(GenServer.server(), list(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def train(trainer, dataset, opts \\ []) do
    GenServer.call(trainer, {:train, dataset, opts}, :infinity)
  end

  @doc """
  Saves a checkpoint with the given name.

  ## Examples

      {:ok, name} = Trainer.save_checkpoint(trainer, "epoch-3-step-100")

  """
  @spec save_checkpoint(GenServer.server(), String.t()) ::
          {:ok, String.t()} | {:error, term()}
  def save_checkpoint(trainer, name) do
    GenServer.call(trainer, {:save_checkpoint, name})
  end

  @doc """
  Gets current training metrics.
  """
  @spec get_metrics(GenServer.server()) :: map()
  def get_metrics(trainer) do
    GenServer.call(trainer, :get_metrics)
  end

  @doc """
  Stops the trainer.
  """
  @spec stop(GenServer.server()) :: :ok
  def stop(trainer) do
    GenServer.stop(trainer)
  end

  # Server callbacks

  @impl true
  def init(config) do
    state = %__MODULE__{
      config: config,
      current_step: 0,
      current_epoch: 0,
      metrics_buffer: [],
      checkpoints: [],
      status: :idle,
      callbacks: %{}
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:train, dataset, opts}, _from, state) do
    session = Keyword.fetch!(opts, :session)
    callbacks = Keyword.get(opts, :callbacks, %{})
    gradient_hook_specs = Keyword.get(opts, :gradient_hooks, [])

    # Setup gradient hooks if requested
    gradient_hooks =
      if gradient_hook_specs != [] do
        {:ok, hooks_pid} = GradientHooks.start_link()

        Enum.each(gradient_hook_specs, fn
          :norm ->
            GradientHooks.register_hook(hooks_pid, :norm, GradientHooks.gradient_norm_hook())

          :distribution ->
            GradientHooks.register_hook(
              hooks_pid,
              :dist,
              GradientHooks.gradient_distribution_hook()
            )

          :flow ->
            GradientHooks.register_hook(hooks_pid, :flow, GradientHooks.gradient_flow_hook())

          :health ->
            GradientHooks.register_hook(hooks_pid, :health, GradientHooks.gradient_health_hook())

          {name, hook} ->
            GradientHooks.register_hook(hooks_pid, name, hook)
        end)

        hooks_pid
      else
        nil
      end

    training_state = %{
      state
      | session: session,
        dataset: dataset,
        callbacks: callbacks,
        gradient_hooks: gradient_hooks,
        status: :training
    }

    # Run training loop
    final_state = run_training_loop(training_state)

    # Clean up gradient hooks
    if gradient_hooks, do: GenServer.stop(gradient_hooks)

    # Calculate final metrics
    metrics = calculate_final_metrics(final_state.metrics_buffer)

    result = %{
      total_steps: final_state.current_step,
      epochs_completed: final_state.current_epoch,
      metrics: metrics,
      checkpoints: final_state.checkpoints
    }

    {:reply, {:ok, result}, %{final_state | status: :idle}}
  end

  @impl true
  def handle_call({:save_checkpoint, name}, _from, state) do
    checkpoints = [name | state.checkpoints]
    {:reply, {:ok, name}, %{state | checkpoints: checkpoints}}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    metrics = calculate_final_metrics(state.metrics_buffer)
    {:reply, metrics, state}
  end

  # Private functions

  defp run_training_loop(state) do
    batches = batch_dataset(state.dataset, state.config.batch_size)

    Enum.reduce(1..state.config.epochs, state, fn epoch, acc_state ->
      epoch_state = %{acc_state | current_epoch: epoch}

      epoch_final = run_epoch(epoch_state, batches)

      # Emit epoch telemetry
      :telemetry.execute(
        [:crucible, :training, :epoch],
        %{epoch: epoch, avg_loss: avg_loss(epoch_final.metrics_buffer)},
        %{experiment_id: nil}
      )

      # Invoke epoch callback
      invoke_callback(epoch_final.callbacks, :on_epoch_end, %{
        epoch: epoch,
        avg_loss: avg_loss(epoch_final.metrics_buffer)
      })

      epoch_final
    end)
  end

  defp run_epoch(state, batches) do
    Enum.reduce(batches, state, fn batch, step_state ->
      step = step_state.current_step + 1

      # Execute training step
      step_result = train_step(step_state, batch)

      # Run gradient hooks
      step_result =
        if step_state.gradient_hooks do
          GradientHooks.run_hooks(step_state.gradient_hooks, step_result)
        else
          step_result
        end

      # Emit step telemetry
      :telemetry.execute(
        [:crucible, :training, :step],
        %{step: step, loss: step_result.loss, grad_norm: step_result.grad_norm},
        %{epoch: step_state.current_epoch}
      )

      # Invoke step callback
      invoke_callback(step_state.callbacks, :on_step_end, step_result)

      # Checkpoint if needed
      new_state =
        if should_checkpoint?(step, step_state.config.checkpoint_interval) do
          name = "step-#{step}"
          checkpoints = [name | step_state.checkpoints]

          invoke_callback(step_state.callbacks, :on_checkpoint, %{
            name: name,
            step: step
          })

          %{step_state | checkpoints: checkpoints}
        else
          step_state
        end

      # Update state
      %{
        new_state
        | current_step: step,
          metrics_buffer: [step_result | new_state.metrics_buffer]
      }
    end)
  end

  defp train_step(state, batch) do
    # Call session for forward-backward
    {:ok, fb_result} =
      GenServer.call(state.session, {:forward_backward, batch})

    # Call session for optimizer step
    adam_params = Config.to_adam_params(state.config)
    {:ok, optim_result} = GenServer.call(state.session, {:optim_step, adam_params})

    %{
      step: state.current_step + 1,
      epoch: state.current_epoch,
      loss: fb_result.loss,
      grad_norm: optim_result.grad_norm,
      gradients: Map.get(fb_result, :gradients, %{}),
      timestamp: DateTime.utc_now()
    }
  end

  defp should_checkpoint?(step, interval) when interval > 0 do
    rem(step, interval) == 0
  end

  defp should_checkpoint?(_, _), do: false

  defp batch_dataset(dataset, batch_size) do
    dataset
    |> Enum.chunk_every(batch_size)
  end

  defp calculate_final_metrics(metrics_buffer) when metrics_buffer == [] do
    %{avg_loss: 0.0, final_loss: 0.0}
  end

  defp calculate_final_metrics(metrics_buffer) do
    losses = Enum.map(metrics_buffer, & &1.loss)

    %{
      avg_loss: Enum.sum(losses) / length(losses),
      final_loss: hd(losses)
    }
  end

  defp avg_loss(metrics_buffer) when metrics_buffer == [], do: 0.0

  defp avg_loss(metrics_buffer) do
    losses = Enum.map(metrics_buffer, & &1.loss)
    Enum.sum(losses) / length(losses)
  end

  defp invoke_callback(callbacks, event, data) do
    case Map.get(callbacks, event) do
      nil -> :ok
      callback -> callback.(data)
    end
  end
end
