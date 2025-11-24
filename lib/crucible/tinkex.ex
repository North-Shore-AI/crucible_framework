defmodule Crucible.Tinkex do
  @moduledoc """
  Tinkex adapter for Crucible's LoRA fine-tuning workflows.

  Implements the `Crucible.Lora.Adapter` behaviour by delegating
  to the Tinkex SDK for actual training operations. This keeps
  proprietary Tinker concepts behind a single adapter boundary.
  """

  require Logger

  @behaviour Crucible.Lora.Adapter

  alias Tinkex.{Config, SessionManager, TrainingClient}
  alias Tinkex.Types.{Datum, ModelInput, LoraConfig}

  @impl true
  def generate_id do
    # Generate a unique ID without external UUID dependency
    :crypto.strong_rand_bytes(16)
    |> Base.encode16(case: :lower)
    |> then(fn hex ->
      # Format as UUID-like string
      <<a::binary-size(8), b::binary-size(4), c::binary-size(4), d::binary-size(4),
        e::binary-size(12)>> = hex

      "#{a}-#{b}-#{c}-#{d}-#{e}"
    end)
  end

  @impl true
  def create_experiment(opts) do
    base_model = Keyword.get(opts, :base_model)

    if is_nil(base_model) do
      {:error, "base_model is required"}
    else
      experiment = %{
        id: generate_id(),
        backend: :tinkex,
        base_model: base_model,
        lora_rank: Keyword.get(opts, :lora_rank, 8),
        lora_alpha: Keyword.get(opts, :lora_alpha, 16),
        target_modules: Keyword.get(opts, :target_modules, ["q_proj", "v_proj"]),
        created_at: DateTime.utc_now()
      }

      {:ok, experiment}
    end
  end

  @impl true
  def batch_dataset(dataset, batch_size) do
    Enum.chunk_every(dataset, batch_size)
  end

  @impl true
  def format_training_data(batch, _opts) do
    Enum.map(batch, fn item ->
      Map.put(item, :formatted, true)
    end)
  end

  @impl true
  def calculate_metrics(results) do
    if Enum.empty?(results) do
      %{mean_loss: 0.0, mean_accuracy: 0.0}
    else
      losses = Enum.map(results, &Map.get(&1, :loss, 0.0))
      accuracies = Enum.map(results, &Map.get(&1, :accuracy, 0.0))

      %{
        mean_loss: Enum.sum(losses) / length(losses),
        mean_accuracy: Enum.sum(accuracies) / length(accuracies),
        total_steps: length(results)
      }
    end
  end

  @impl true
  def validate_quality(results, config) do
    min_accuracy = Map.get(config, :min_accuracy, 0.0)
    max_loss = Map.get(config, :max_loss, 10.0)

    accuracy = Map.get(results, :accuracy, 0.0)
    loss = Map.get(results, :loss, 0.0)

    passed = accuracy >= min_accuracy and loss <= max_loss

    %{
      passed: passed,
      accuracy_check: accuracy >= min_accuracy,
      loss_check: loss <= max_loss,
      results: results
    }
  end

  @impl true
  def sampling_params(opts) do
    %{
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.95),
      max_tokens: Keyword.get(opts, :max_tokens, 512),
      stop_sequences: Keyword.get(opts, :stop_sequences, [])
    }
  end

  @impl true
  def checkpoint_name(experiment_id, step) do
    "#{experiment_id}-step-#{step}"
  end

  # Session-based training using real Tinkex API

  @impl true
  def start_session(experiment) do
    Logger.info("Starting Tinkex training session for #{experiment.id}")

    # Create Tinkex config from environment
    config = Config.new()

    # Start session via SessionManager
    case SessionManager.start_session(config) do
      {:ok, session_id} ->
        Logger.info("Created Tinkex session: #{session_id}")

        # Start TrainingClient GenServer
        # Note: Tinkex LoraConfig only uses rank, not alpha/target_modules
        lora_config = %LoraConfig{
          rank: experiment.lora_rank
        }

        client_opts = [
          config: config,
          session_id: session_id,
          model_seq_id: 0,
          base_model: experiment.base_model,
          lora_config: lora_config
        ]

        case DynamicSupervisor.start_child(
               Tinkex.ClientSupervisor,
               {TrainingClient, client_opts}
             ) do
          {:ok, client_pid} ->
            Logger.info("Started TrainingClient: #{inspect(client_pid)}")
            # Return session info containing client pid and config
            {:ok,
             %{
               client: client_pid,
               session_id: session_id,
               config: config,
               experiment: experiment
             }}

          {:error, reason} ->
            Logger.error("Failed to start TrainingClient: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, reason} ->
        Logger.error("Failed to create Tinkex session: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def forward_backward(session, batch, opts) do
    client = session.client
    model_name = session.experiment.base_model

    # Format batch as Tinkex Datum structs
    data =
      Enum.map(batch, fn item ->
        # Create ModelInput from text
        input_text = Map.get(item, :input, "")
        output_text = Map.get(item, :output, "")
        combined_text = "#{input_text}\n#{output_text}"

        case ModelInput.from_text(combined_text, model_name: model_name, training_client: client) do
          {:ok, model_input} ->
            # For language modeling, target tokens are the output
            case ModelInput.from_text(output_text,
                   model_name: model_name,
                   training_client: client
                 ) do
              {:ok, target_input} ->
                target_tokens = ModelInput.to_ints(target_input)
                num_tokens = length(target_tokens)
                # Create uniform weights (1.0 for each token) like Python example
                weights = List.duplicate(1.0, num_tokens)

                %Datum{
                  model_input: model_input,
                  loss_fn_inputs: %{
                    "target_tokens" => %Tinkex.Types.TensorData{
                      data: target_tokens,
                      dtype: :int64,
                      shape: [num_tokens]
                    },
                    "weights" => %Tinkex.Types.TensorData{
                      data: weights,
                      dtype: :float32,
                      shape: [num_tokens]
                    }
                  }
                }

              {:error, reason} ->
                Logger.warning("Failed to tokenize target: #{inspect(reason)}")
                nil
            end

          {:error, reason} ->
            Logger.warning("Failed to tokenize input: #{inspect(reason)}")
            nil
        end
      end)
      |> Enum.reject(&is_nil/1)

    if Enum.empty?(data) do
      {:error, "No valid training data after tokenization"}
    else
      loss_fn = Keyword.get(opts, :loss_fn, :cross_entropy)

      # Call real TrainingClient.forward_backward
      case TrainingClient.forward_backward(client, data, loss_fn, opts) do
        {:ok, task} ->
          # Wait for the async result
          case Task.await(task, :infinity) do
            {:ok, result} ->
              Logger.debug("Forward/backward complete: loss=#{result.total_loss}")

              {:ok,
               %{
                 loss: result.total_loss,
                 batch_size: result.num_examples,
                 grad_norm: Map.get(result, :grad_norm, 0.0)
               }}

            {:error, reason} ->
              Logger.error("Forward/backward failed: #{inspect(reason)}")
              {:error, reason}
          end

        {:error, reason} ->
          Logger.error("Failed to start forward/backward: #{inspect(reason)}")
          {:error, reason}
      end
    end
  end

  @impl true
  def optim_step(session, params, opts) do
    client = session.client

    adam_params =
      Map.merge(
        %{
          learning_rate: 1.0e-4,
          beta1: 0.9,
          beta2: 0.999,
          epsilon: 1.0e-8,
          weight_decay: 0.01
        },
        params
      )

    case TrainingClient.optim_step(client, adam_params, opts) do
      {:ok, task} ->
        case Task.await(task, :infinity) do
          {:ok, result} ->
            Logger.debug("Optimizer step complete")
            {:ok, result}

          {:error, reason} ->
            Logger.error("Optimizer step failed: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def create_sampler(session, checkpoint_name) do
    client = session.client

    # Save weights for sampling first
    case TrainingClient.save_weights_for_sampler(client, path: checkpoint_name) do
      {:ok, task} ->
        case Task.await(task, :infinity) do
          {:ok, %{"model_path" => model_path}} ->
            # Create sampling client from the saved weights
            sampler_task = TrainingClient.create_sampling_client_async(client, model_path)

            case Task.await(sampler_task, :infinity) do
              {:ok, sampler_pid} ->
                {:ok, %{sampler: sampler_pid, model_path: model_path}}

              {:error, reason} ->
                {:error, reason}
            end

          {:ok, result} ->
            Logger.warning("Unexpected save_weights result: #{inspect(result)}")
            {:error, :invalid_response}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def sample(sampler_session, prompt, opts) do
    sampler = sampler_session.sampler

    sampling_params = %Tinkex.Types.SamplingParams{
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.95),
      max_tokens: Keyword.get(opts, :max_tokens, 512)
    }

    case Tinkex.SamplingClient.sample(sampler, prompt, sampling_params, opts) do
      {:ok, task} ->
        case Task.await(task, :infinity) do
          {:ok, sequences} ->
            # Extract text from sampled sequences
            texts =
              Enum.map(sequences, fn seq ->
                Map.get(seq, :text, Map.get(seq, "text", ""))
              end)

            {:ok, texts}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end
end
