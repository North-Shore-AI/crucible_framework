defmodule Crucible.Tinkex.Session do
  @moduledoc """
  Manages the lifecycle of a Tinkex training session.

  This GenServer wraps Tinkex SDK clients for training and inference,
  providing experiment orchestration, checkpoint management, and
  telemetry integration for the Crucible Framework.

  ## Features

  - Session lifecycle management (init, training, evaluation, completion)
  - Forward-backward pass execution with automatic metrics buffering
  - Optimizer step execution
  - Checkpoint saving and loading
  - Sampling client creation for inference
  - Telemetry event emission for all operations

  ## Usage

      # Start a session
      {:ok, session} = Crucible.Tinkex.Session.start_link(experiment: experiment)

      # Execute training step
      {:ok, metrics} = GenServer.call(session, {:forward_backward, batch, opts})

      # Run optimizer step
      {:ok, result} = GenServer.call(session, {:optim_step, adam_params, opts})

      # Save checkpoint
      {:ok, checkpoint} = GenServer.call(session, {:save_checkpoint, step})

      # Create sampler for inference
      {:ok, sampler} = GenServer.call(session, {:create_sampler, checkpoint_name})

  """

  use GenServer

  alias Crucible.Tinkex.Config

  @type status :: :initializing | :ready | :training | :evaluating | :completed | :failed

  @type t :: %__MODULE__{
          experiment_id: String.t(),
          service_client: pid() | nil,
          training_client: pid() | nil,
          sampling_client: pid() | nil,
          config: Config.t(),
          checkpoints: list(),
          metrics_buffer: list(),
          started_at: DateTime.t() | nil,
          status: status()
        }

  defstruct [
    :experiment_id,
    :service_client,
    :training_client,
    :sampling_client,
    :config,
    :checkpoints,
    :metrics_buffer,
    :started_at,
    :status
  ]

  @doc """
  Starts a new training session.

  ## Options

  - `:experiment` - Experiment map with :id, :name, and :config (required)
  - `:name` - Optional registered name for the GenServer

  ## Examples

      {:ok, session} = Crucible.Tinkex.Session.start_link(
        experiment: %{
          id: "exp-123",
          name: "My Experiment",
          config: config
        }
      )
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @doc """
  Stops the session gracefully.
  """
  @spec stop(pid()) :: :ok
  def stop(session) do
    GenServer.stop(session)
  end

  @impl true
  def init(opts) do
    experiment = Keyword.fetch!(opts, :experiment)
    config = experiment.config

    case start_clients(config) do
      {:ok, service_client, training_client} ->
        state = %__MODULE__{
          experiment_id: experiment.id,
          service_client: service_client,
          training_client: training_client,
          sampling_client: nil,
          config: config,
          checkpoints: [],
          metrics_buffer: [],
          started_at: DateTime.utc_now(),
          status: :ready
        }

        emit_telemetry(:session_start, state)

        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:forward_backward, batch, opts}, from, state) do
    new_state = %{state | status: :training}

    emit_telemetry(:forward_backward_start, new_state, %{batch_size: length(batch)})

    case do_forward_backward(new_state, batch, opts) do
      {:ok, task} ->
        # Handle async completion
        spawn_link(fn ->
          result = Task.await(task, :infinity)
          handle_forward_backward_result(from, result, new_state)
        end)

        {:noreply, new_state}

      {:error, _reason} = error ->
        emit_telemetry(:forward_backward_error, new_state, %{error: error})
        {:reply, error, new_state}
    end
  end

  @impl true
  def handle_call({:optim_step, adam_params, opts}, from, state) do
    emit_telemetry(:optim_step_start, state)

    case do_optim_step(state, adam_params, opts) do
      {:ok, task} ->
        spawn_link(fn ->
          result = Task.await(task, :infinity)
          handle_optim_step_result(from, result, state)
        end)

        {:noreply, state}

      {:error, _reason} = error ->
        emit_telemetry(:optim_step_error, state, %{error: error})
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:save_checkpoint, step}, _from, state) do
    checkpoint_name = Crucible.Tinkex.checkpoint_name(state.experiment_id, step)

    case do_save_checkpoint(state, checkpoint_name) do
      {:ok, task} ->
        result = Task.await(task, :infinity)

        case result do
          {:ok, path_data} ->
            checkpoint = %{
              name: checkpoint_name,
              step: step,
              path: extract_path(path_data),
              created_at: DateTime.utc_now()
            }

            new_state = %{state | checkpoints: [checkpoint | state.checkpoints]}

            emit_telemetry(:checkpoint_saved, new_state, checkpoint)
            {:reply, {:ok, checkpoint}, new_state}

          {:error, _reason} = error ->
            {:reply, error, state}
        end

      {:error, _reason} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:create_sampler, checkpoint_name}, _from, state) do
    checkpoint = find_checkpoint(state.checkpoints, checkpoint_name)

    case checkpoint do
      nil ->
        {:reply, {:error, :checkpoint_not_found}, state}

      %{path: path} ->
        case do_create_sampler(state, path) do
          {:ok, sampler_pid} ->
            new_state = %{state | sampling_client: sampler_pid}
            emit_telemetry(:sampler_created, new_state, %{checkpoint: checkpoint_name})
            {:reply, {:ok, sampler_pid}, new_state}

          {:error, _reason} = error ->
            {:reply, error, state}
        end
    end
  end

  @impl true
  def handle_call({:sample, prompt, opts}, _from, state) do
    case state.sampling_client do
      nil ->
        {:reply, {:error, :no_sampler}, state}

      sampler ->
        case do_sample(sampler, prompt, opts) do
          {:ok, task} ->
            result = Task.await(task, :infinity)
            {:reply, result, state}

          {:error, _reason} = error ->
            {:reply, error, state}
        end
    end
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_cast({:training_result, metrics}, state) do
    new_state = %{state | metrics_buffer: [metrics | state.metrics_buffer]}
    {:noreply, new_state}
  end

  @impl true
  def terminate(_reason, state) do
    emit_telemetry(:session_end, state)
    :ok
  end

  # Private functions

  defp start_clients(config) do
    # Start Tinkex ServiceClient and TrainingClient
    service_opts = [
      config: tinkex_config(config)
    ]

    with {:ok, service} <- start_service_client(service_opts),
         {:ok, training} <- start_training_client(service, config) do
      {:ok, service, training}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp start_service_client(opts) do
    if Code.ensure_loaded?(Tinkex.ServiceClient) do
      Tinkex.ServiceClient.start_link(opts)
    else
      {:error, :tinkex_not_available}
    end
  end

  defp start_training_client(service, config) do
    if Code.ensure_loaded?(Tinkex.ServiceClient) do
      training_opts = [
        base_model: config.base_model,
        lora_config: config.lora_config
      ]

      Tinkex.ServiceClient.create_lora_training_client(service, training_opts)
    else
      {:error, :tinkex_not_available}
    end
  end

  defp tinkex_config(config) do
    if Code.ensure_loaded?(Tinkex.Config) do
      Tinkex.Config.new(
        api_key: config.api_key,
        base_url: config.base_url
      )
    else
      %{
        api_key: config.api_key,
        base_url: config.base_url
      }
    end
  end

  defp do_forward_backward(state, batch, opts) do
    if Code.ensure_loaded?(Tinkex.TrainingClient) do
      # Format batch for Tinkex
      data = format_for_tinkex(batch)
      loss_fn = Keyword.get(opts, :loss_fn, :cross_entropy)

      Tinkex.TrainingClient.forward_backward(
        state.training_client,
        data,
        loss_fn,
        opts
      )
    else
      {:error, :tinkex_not_available}
    end
  end

  defp do_optim_step(state, adam_params, opts) do
    if Code.ensure_loaded?(Tinkex.TrainingClient) do
      Tinkex.TrainingClient.optim_step(
        state.training_client,
        adam_params,
        opts
      )
    else
      {:error, :tinkex_not_available}
    end
  end

  defp do_save_checkpoint(state, checkpoint_name) do
    if Code.ensure_loaded?(Tinkex.TrainingClient) do
      path = "tinker://#{state.experiment_id}/checkpoints/#{checkpoint_name}"

      Tinkex.TrainingClient.save_weights_for_sampler(
        state.training_client,
        path: path
      )
    else
      {:error, :tinkex_not_available}
    end
  end

  defp do_create_sampler(state, model_path) do
    if Code.ensure_loaded?(Tinkex.ServiceClient) do
      # Create sampling client from the service client with the model_path
      opts = [model_path: model_path]
      Tinkex.ServiceClient.create_sampling_client(state.service_client, opts)
    else
      {:error, :tinkex_not_available}
    end
  end

  defp do_sample(sampler, prompt, opts) do
    if Code.ensure_loaded?(Tinkex.SamplingClient) do
      sampling_params = build_sampling_params(opts)

      # Build model input from prompt
      model_input =
        if Code.ensure_loaded?(Tinkex.Types.ModelInput) do
          Tinkex.Types.ModelInput.from_text(prompt, opts[:model] || "meta-llama/Llama-3.2-1B")
        else
          %{text: prompt}
        end

      Tinkex.SamplingClient.sample(sampler, model_input, sampling_params, opts)
    else
      {:error, :tinkex_not_available}
    end
  end

  defp format_for_tinkex(batch) do
    Enum.map(batch, fn example ->
      %{
        model_input: example[:input] || example["input"],
        loss_fn_inputs: %{
          target: example[:output] || example["output"]
        }
      }
    end)
  end

  defp build_sampling_params(opts) do
    %{
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.95),
      max_tokens: Keyword.get(opts, :max_tokens, 512)
    }
  end

  defp find_checkpoint(checkpoints, name) do
    Enum.find(checkpoints, fn cp -> cp.name == name end)
  end

  defp extract_path(path_data) when is_binary(path_data), do: path_data
  defp extract_path(%{"path" => path}), do: path
  defp extract_path(%{path: path}), do: path
  defp extract_path(_), do: nil

  defp handle_forward_backward_result(from, result, state) do
    case result do
      {:ok, output} ->
        metrics = %{
          loss: output.loss || 0.0,
          timestamp: DateTime.utc_now()
        }

        emit_telemetry(:forward_backward_stop, state, metrics)
        GenServer.reply(from, {:ok, metrics})

      {:error, _reason} = error ->
        emit_telemetry(:forward_backward_error, state, %{error: error})
        GenServer.reply(from, error)
    end
  end

  defp handle_optim_step_result(from, result, state) do
    case result do
      {:ok, output} ->
        metrics = %{
          grad_norm: output.grad_norm || 0.0,
          timestamp: DateTime.utc_now()
        }

        emit_telemetry(:optim_step_stop, state, metrics)
        GenServer.reply(from, {:ok, metrics})

      {:error, _reason} = error ->
        emit_telemetry(:optim_step_error, state, %{error: error})
        GenServer.reply(from, error)
    end
  end

  defp emit_telemetry(event, state, metadata \\ %{}) do
    :telemetry.execute(
      [:crucible, :tinkex, event],
      %{timestamp: System.system_time(:millisecond)},
      Map.merge(metadata, %{
        experiment_id: state.experiment_id,
        status: state.status
      })
    )
  end
end
