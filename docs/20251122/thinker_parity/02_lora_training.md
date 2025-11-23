# LoRA Training Module

## Overview

Training interface wrapping Tinkex API with crucible_telemetry instrumentation.

## Module Design

### Lora.Config

```elixir
defmodule Crucible.Thinker.Lora.Config do
  @moduledoc """
  LoRA training configuration.
  """

  defstruct [
    :name,
    :base_model,
    lora_rank: 16,
    lora_alpha: 32,
    learning_rate: 2.0e-4,
    epochs: 3,
    batch_size: 8,
    warmup_steps: 100,
    loss_fn_config: %{
      citation_validity_weight: 5.0
    }
  ]

  @type t :: %__MODULE__{
    name: String.t(),
    base_model: String.t(),
    lora_rank: pos_integer(),
    lora_alpha: pos_integer(),
    learning_rate: float(),
    epochs: pos_integer(),
    batch_size: pos_integer(),
    warmup_steps: non_neg_integer(),
    loss_fn_config: map()
  }

  def new(opts) do
    struct!(__MODULE__, opts)
  end

  def validate(%__MODULE__{} = config) do
    cond do
      is_nil(config.name) -> {:error, :name_required}
      is_nil(config.base_model) -> {:error, :base_model_required}
      config.lora_rank < 1 -> {:error, :invalid_lora_rank}
      config.learning_rate <= 0 -> {:error, :invalid_learning_rate}
      true -> :ok
    end
  end
end
```

### Lora.TrainingLoop

```elixir
defmodule Crucible.Thinker.Lora.TrainingLoop do
  @moduledoc """
  Training orchestration with Tinkex integration.
  """

  alias Crucible.Thinker.Lora.Config
  alias Crucible.Tinkex.TrainingClient

  require Logger

  @doc """
  Create and run a training experiment.

  ## Example
      config = Config.new(
        name: "claim-extractor",
        base_model: "meta-llama/Llama-3.1-8B-Instruct"
      )
      {:ok, result} = TrainingLoop.run(config, dataset)
  """
  def run(%Config{} = config, dataset) do
    with :ok <- Config.validate(config),
         {:ok, job_id} <- submit_training_job(config, dataset),
         {:ok, result} <- await_completion(job_id, config) do
      emit_completion_telemetry(config, result)
      {:ok, result}
    end
  end

  defp submit_training_job(config, dataset) do
    emit_start_telemetry(config, dataset)

    training_data = Enum.map(dataset, fn sample ->
      %{
        "prompt" => sample.prompt,
        "completion" => sample.expected_output
      }
    end)

    payload = %{
      "experiment_name" => config.name,
      "base_model" => config.base_model,
      "lora_config" => %{
        "rank" => config.lora_rank,
        "alpha" => config.lora_alpha
      },
      "training_config" => %{
        "learning_rate" => config.learning_rate,
        "epochs" => config.epochs,
        "batch_size" => config.batch_size,
        "warmup_steps" => config.warmup_steps
      },
      "loss_config" => config.loss_fn_config,
      "training_data" => training_data
    }

    TrainingClient.submit(payload)
  end

  defp await_completion(job_id, config) do
    timeout = config.epochs * 60_000  # 1 min per epoch estimate

    TrainingClient.await(job_id, timeout: timeout, on_progress: fn progress ->
      emit_progress_telemetry(config, progress)
    end)
  end

  # Telemetry emissions

  defp emit_start_telemetry(config, dataset) do
    :telemetry.execute(
      [:crucible, :thinker, :training, :start],
      %{dataset_size: length(dataset)},
      %{
        name: config.name,
        base_model: config.base_model,
        lora_rank: config.lora_rank,
        epochs: config.epochs
      }
    )
  end

  defp emit_progress_telemetry(config, progress) do
    :telemetry.execute(
      [:crucible, :thinker, :training, :progress],
      %{
        epoch: progress.epoch,
        step: progress.step,
        loss: progress.loss
      },
      %{name: config.name}
    )
  end

  defp emit_completion_telemetry(config, result) do
    :telemetry.execute(
      [:crucible, :thinker, :training, :complete],
      %{
        final_loss: result.final_loss,
        duration_ms: result.duration_ms
      },
      %{name: config.name}
    )
  end
end
```

## Tinkex TrainingClient

```elixir
defmodule Crucible.Tinkex.TrainingClient do
  @moduledoc """
  HTTP client for Tinkex training API.
  """

  use GenServer

  @base_url Application.compile_env(:crucible, :tinkex_url, "http://localhost:8080")

  def submit(payload) do
    case Req.post("#{@base_url}/v1/train", json: payload) do
      {:ok, %{status: 202, body: %{"job_id" => job_id}}} ->
        {:ok, job_id}
      {:ok, %{status: status, body: body}} ->
        {:error, {:api_error, status, body}}
      {:error, reason} ->
        {:error, {:request_failed, reason}}
    end
  end

  def await(job_id, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 300_000)
    on_progress = Keyword.get(opts, :on_progress, fn _ -> :ok end)

    poll_until_complete(job_id, timeout, on_progress, System.monotonic_time(:millisecond))
  end

  defp poll_until_complete(job_id, timeout, on_progress, start_time) do
    elapsed = System.monotonic_time(:millisecond) - start_time

    if elapsed > timeout do
      {:error, :timeout}
    else
      case get_status(job_id) do
        {:ok, %{status: "completed"} = result} ->
          {:ok, result}
        {:ok, %{status: "failed", error: error}} ->
          {:error, {:job_failed, error}}
        {:ok, %{status: "running"} = progress} ->
          on_progress.(progress)
          Process.sleep(5_000)
          poll_until_complete(job_id, timeout, on_progress, start_time)
        {:error, _} = error ->
          error
      end
    end
  end

  defp get_status(job_id) do
    case Req.get("#{@base_url}/v1/train/#{job_id}") do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
      {:error, reason} -> {:error, reason}
    end
  end
end
```

## crucible_telemetry Integration

Attach handlers in application startup:

```elixir
defmodule Crucible.Thinker.Telemetry do
  def attach_handlers do
    :telemetry.attach_many(
      "crucible-thinker-training",
      [
        [:crucible, :thinker, :training, :start],
        [:crucible, :thinker, :training, :progress],
        [:crucible, :thinker, :training, :complete]
      ],
      &handle_event/4,
      nil
    )
  end

  defp handle_event(event, measurements, metadata, _config) do
    # Store in crucible_telemetry for later analysis
    Crucible.Telemetry.Research.capture(event, measurements, metadata)
  end
end
```

## Usage Example

```elixir
alias Crucible.Thinker.{Datasets.SciFact, Lora.Config, Lora.TrainingLoop}

# Load dataset
{:ok, dataset} = SciFact.load(split: :train, limit: 15)

# Format for training
formatted = Enum.map(dataset, &SciFact.format_for_training/1)

# Configure and run
config = Config.new(
  name: "claim-extractor-scifact",
  base_model: "meta-llama/Llama-3.1-8B-Instruct",
  lora_rank: 16,
  learning_rate: 2.0e-4,
  epochs: 3,
  loss_fn_config: %{citation_validity_weight: 5.0}
)

{:ok, result} = TrainingLoop.run(config, formatted)
```
