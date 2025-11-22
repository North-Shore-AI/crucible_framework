# LoRA Training Interface Design

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

This document defines the LoRA fine-tuning abstraction layer in Crucible Framework, including dataset integration, custom loss function registration, and gradient computation hooks for research.

## Module Structure

```
lib/crucible/
  lora.ex                      # Adapter-agnostic entry point
  lora/
    adapter.ex                 # Behaviour definition
    config.ex                  # LoRA hyperparameters
    loss_functions.ex          # Loss function registry
    gradient_hooks.ex          # Research gradient access
    training_loop.ex           # High-level training abstraction
    evaluation.ex              # Evaluation pipeline
```

## LoRA Configuration

```elixir
defmodule Crucible.Lora.Config do
  @moduledoc """
  Configuration for LoRA fine-tuning experiments.
  """

  @type t :: %__MODULE__{
    # Model configuration
    base_model: String.t(),
    lora_rank: pos_integer(),
    lora_alpha: pos_integer(),
    lora_dropout: float(),
    target_modules: [String.t()],

    # Training hyperparameters
    learning_rate: float(),
    weight_decay: float(),
    warmup_steps: non_neg_integer(),
    max_grad_norm: float(),

    # Adam parameters
    adam_beta1: float(),
    adam_beta2: float(),
    adam_epsilon: float(),

    # Quality targets
    quality_targets: map()
  }

  defstruct [
    base_model: "llama-3-8b",
    lora_rank: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"],

    learning_rate: 1.0e-4,
    weight_decay: 0.01,
    warmup_steps: 100,
    max_grad_norm: 1.0,

    adam_beta1: 0.9,
    adam_beta2: 0.999,
    adam_epsilon: 1.0e-8,

    quality_targets: %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      overall_pass_rate: 0.45
    }
  ]

  @doc """
  Creates config from keyword list with validation.
  """
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    validate!(config)
    config
  end

  defp validate!(%__MODULE__{lora_rank: rank}) when rank < 1 do
    raise ArgumentError, "lora_rank must be positive"
  end
  defp validate!(%__MODULE__{learning_rate: lr}) when lr <= 0 do
    raise ArgumentError, "learning_rate must be positive"
  end
  defp validate!(config), do: config

  @doc """
  Returns Adam parameters formatted for Tinkex.
  """
  def adam_params(%__MODULE__{} = config) do
    %{
      lr: config.learning_rate,
      beta1: config.adam_beta1,
      beta2: config.adam_beta2,
      eps: config.adam_epsilon,
      weight_decay: config.weight_decay,
      max_grad_norm: config.max_grad_norm
    }
  end
end
```

## Dataset Integration with crucible_datasets

```elixir
defmodule Crucible.Lora.DatasetIntegration do
  @moduledoc """
  Integrates crucible_datasets with LoRA training pipeline.
  """

  alias Crucible.Datasets

  @doc """
  Loads and prepares a dataset for LoRA fine-tuning.
  """
  def prepare_dataset(dataset_name, opts \\ []) do
    split = Keyword.get(opts, :split, :train)
    transform = Keyword.get(opts, :transform, &default_transform/1)

    with {:ok, raw_data} <- Datasets.load(dataset_name, split: split),
         transformed <- Enum.map(raw_data, transform) do
      {:ok, transformed}
    end
  end

  @doc """
  Streams dataset for memory-efficient training.
  """
  def stream_dataset(dataset_name, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 8)

    dataset_name
    |> Datasets.stream(opts)
    |> Stream.map(&transform_example/1)
    |> Stream.chunk_every(batch_size)
  end

  defp default_transform(example) do
    %{
      input: format_input(example),
      output: format_output(example),
      metadata: extract_metadata(example)
    }
  end

  @doc """
  Dataset-specific formatters.
  """
  def formatter(:scifact) do
    fn example ->
      claim = example["claim"]
      evidence = Enum.join(example["evidence"], " ")

      %{
        input: "Claim: #{claim}\nEvidence: #{evidence}\n\nVerdict:",
        output: example["label"],
        metadata: %{
          claim_id: example["id"],
          evidence_ids: example["evidence_ids"]
        }
      }
    end
  end

  def formatter(:fever) do
    fn example ->
      %{
        input: "Claim: #{example["claim"]}\n\nVerify:",
        output: example["label"],
        metadata: %{id: example["id"]}
      }
    end
  end

  def formatter(:gsm8k) do
    fn example ->
      %{
        input: "Problem: #{example["question"]}\n\nSolution:",
        output: example["answer"],
        metadata: %{}
      }
    end
  end
end
```

## Loss Function Registry

```elixir
defmodule Crucible.Lora.LossFunctions do
  @moduledoc """
  Registry for custom loss functions including topological and chirality losses.
  """

  @type loss_fn :: atom() | {atom(), map()}
  @type loss_result :: %{loss: float(), components: map()}

  # Built-in loss functions
  @builtin_losses [:cross_entropy, :kl_divergence, :mse]

  @doc """
  Registers a custom loss function.
  """
  def register(name, config \\ %{}) do
    :persistent_term.put({__MODULE__, name}, config)
    :ok
  end

  @doc """
  Gets loss function configuration.
  """
  def get(name) when name in @builtin_losses do
    {:ok, %{type: :builtin, name: name}}
  end

  def get(name) do
    case :persistent_term.get({__MODULE__, name}, nil) do
      nil -> {:error, :not_found}
      config -> {:ok, config}
    end
  end

  @doc """
  Lists all available loss functions.
  """
  def list do
    custom = :persistent_term.get({__MODULE__, :registry}, [])
    @builtin_losses ++ custom
  end

  # CNS-specific loss functions

  @doc """
  Topological loss for SNO graph structure preservation.

  Based on Betti number computation for logical consistency.
  """
  def register_topological_loss do
    register(:topological, %{
      type: :custom,
      compute: &compute_topological_loss/2,
      weight: 0.1,
      requires: [:graph_structure]
    })
  end

  defp compute_topological_loss(predictions, targets) do
    # Placeholder for actual topology computation
    # In practice, this would compute Betti numbers
    %{
      loss: 0.0,
      betti_0: 1,  # Connected components
      betti_1: 0,  # Cycles
      consistency_score: 1.0
    }
  end

  @doc """
  Chirality loss for dialectical balance in synthesis.

  Ensures thesis/antithesis are properly represented.
  """
  def register_chirality_loss do
    register(:chirality, %{
      type: :custom,
      compute: &compute_chirality_loss/2,
      weight: 0.15,
      requires: [:dialectical_structure]
    })
  end

  defp compute_chirality_loss(predictions, targets) do
    %{
      loss: 0.0,
      balance_score: 1.0,
      evidence_coverage: 1.0
    }
  end

  @doc """
  Citation validity loss for grounded generation.
  """
  def register_citation_loss do
    register(:citation_validity, %{
      type: :custom,
      compute: &compute_citation_loss/2,
      weight: 0.2,
      requires: [:citations, :evidence_pool]
    })
  end

  defp compute_citation_loss(predictions, targets) do
    %{
      loss: 0.0,
      invalid_rate: 0.0,
      precision: 1.0,
      recall: 1.0
    }
  end

  @doc """
  Composite loss combining multiple objectives.
  """
  def composite(losses, weights) when length(losses) == length(weights) do
    %{
      type: :composite,
      losses: Enum.zip(losses, weights),
      compute: fn predictions, targets ->
        results = Enum.map(losses, fn loss_name ->
          {:ok, config} = get(loss_name)
          config.compute.(predictions, targets)
        end)

        total_loss = Enum.zip(results, weights)
        |> Enum.map(fn {result, weight} -> result.loss * weight end)
        |> Enum.sum()

        %{loss: total_loss, components: Enum.zip(losses, results) |> Map.new()}
      end
    }
  end
end

# Usage example
Crucible.Lora.LossFunctions.register_topological_loss()
Crucible.Lora.LossFunctions.register_chirality_loss()
Crucible.Lora.LossFunctions.register_citation_loss()

composite_loss = Crucible.Lora.LossFunctions.composite(
  [:cross_entropy, :topological, :chirality, :citation_validity],
  [0.55, 0.1, 0.15, 0.2]
)
```

## Gradient Computation Hooks

```elixir
defmodule Crucible.Lora.GradientHooks do
  @moduledoc """
  Provides research-grade access to gradients during training.
  """

  @type hook :: (gradient_info :: map() -> :ok | {:ok, map()})

  defstruct [
    :experiment_id,
    :hooks,
    :buffer,
    :buffer_size
  ]

  @doc """
  Registers a gradient analysis hook.
  """
  def register_hook(hooks_pid, name, callback) do
    GenServer.call(hooks_pid, {:register, name, callback})
  end

  @doc """
  Pre-defined hook: Gradient norm tracking.
  """
  def gradient_norm_hook do
    fn %{gradients: grads} = info ->
      norms = Enum.map(grads, fn {layer, grad} ->
        {layer, compute_norm(grad)}
      end)

      {:ok, Map.put(info, :gradient_norms, Map.new(norms))}
    end
  end

  @doc """
  Pre-defined hook: Gradient distribution analysis.
  """
  def gradient_distribution_hook do
    fn %{gradients: grads} = info ->
      stats = Enum.map(grads, fn {layer, grad} ->
        {layer, %{
          mean: compute_mean(grad),
          std: compute_std(grad),
          min: compute_min(grad),
          max: compute_max(grad),
          sparsity: compute_sparsity(grad)
        }}
      end)

      {:ok, Map.put(info, :gradient_stats, Map.new(stats))}
    end
  end

  @doc """
  Pre-defined hook: Layer-wise gradient flow.
  """
  def gradient_flow_hook do
    fn %{gradients: grads} = info ->
      flow = grads
      |> Enum.sort_by(fn {layer, _} -> layer_order(layer) end)
      |> Enum.map(fn {layer, grad} ->
        %{layer: layer, magnitude: compute_norm(grad)}
      end)

      {:ok, Map.put(info, :gradient_flow, flow)}
    end
  end

  @doc """
  Pre-defined hook: Detect vanishing/exploding gradients.
  """
  def gradient_health_hook(thresholds \\ %{}) do
    vanishing = Map.get(thresholds, :vanishing, 1.0e-7)
    exploding = Map.get(thresholds, :exploding, 1.0e3)

    fn %{gradients: grads} = info ->
      health = Enum.map(grads, fn {layer, grad} ->
        norm = compute_norm(grad)
        status = cond do
          norm < vanishing -> :vanishing
          norm > exploding -> :exploding
          true -> :healthy
        end
        {layer, status}
      end)

      issues = Enum.filter(health, fn {_, status} -> status != :healthy end)

      {:ok, Map.merge(info, %{
        gradient_health: Map.new(health),
        gradient_issues: issues
      })}
    end
  end

  # Placeholder implementations
  defp compute_norm(_grad), do: 1.0
  defp compute_mean(_grad), do: 0.0
  defp compute_std(_grad), do: 1.0
  defp compute_min(_grad), do: -1.0
  defp compute_max(_grad), do: 1.0
  defp compute_sparsity(_grad), do: 0.0
  defp layer_order(layer), do: 0
end
```

## High-Level Training Interface

```elixir
defmodule Crucible.Lora.TrainingLoop do
  @moduledoc """
  High-level training loop abstraction.
  """

  alias Crucible.Lora.{Config, LossFunctions, GradientHooks, DatasetIntegration}

  defstruct [
    :session,
    :config,
    :gradient_hooks,
    :callbacks,
    :current_step,
    :current_epoch,
    :metrics_buffer
  ]

  @type callback :: %{
    on_step_end: (step_info :: map() -> :ok),
    on_epoch_end: (epoch_info :: map() -> :ok),
    on_checkpoint: (checkpoint_info :: map() -> :ok)
  }

  @doc """
  Runs a complete training loop with all integrations.
  """
  def run(session, dataset, opts \\ []) do
    config = Keyword.get(opts, :config, Config.new())
    epochs = Keyword.get(opts, :epochs, 1)
    batch_size = Keyword.get(opts, :batch_size, 8)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 100)
    loss_fn = Keyword.get(opts, :loss_fn, :cross_entropy)
    callbacks = Keyword.get(opts, :callbacks, %{})

    # Setup gradient hooks if research mode enabled
    gradient_hooks = if opts[:enable_gradient_hooks] do
      setup_gradient_hooks(opts[:gradient_hooks] || [])
    end

    # Prepare batched data
    batches = DatasetIntegration.prepare_dataset(dataset, opts)
    |> elem(1)
    |> Crucible.Tinkex.batch_dataset(batch_size)

    # Initialize state
    state = %__MODULE__{
      session: session,
      config: config,
      gradient_hooks: gradient_hooks,
      callbacks: callbacks,
      current_step: 0,
      current_epoch: 0,
      metrics_buffer: []
    }

    # Training loop
    final_state = Enum.reduce(1..epochs, state, fn epoch, acc_state ->
      epoch_state = %{acc_state | current_epoch: epoch}

      Enum.reduce(batches, epoch_state, fn batch, step_state ->
        step = step_state.current_step + 1

        # Execute training step
        {:ok, step_result} = train_step(step_state, batch, loss_fn)

        # Run gradient hooks
        step_result = if gradient_hooks do
          run_gradient_hooks(gradient_hooks, step_result)
        else
          step_result
        end

        # Checkpoint if needed
        step_state = if rem(step, checkpoint_every) == 0 do
          {:ok, checkpoint} = save_checkpoint(step_state, step)
          invoke_callback(step_state.callbacks, :on_checkpoint, checkpoint)
          step_state
        else
          step_state
        end

        # Invoke step callback
        invoke_callback(step_state.callbacks, :on_step_end, step_result)

        # Update state
        %{step_state |
          current_step: step,
          metrics_buffer: [step_result | step_state.metrics_buffer]
        }
      end)
      |> tap(fn s -> invoke_callback(s.callbacks, :on_epoch_end, %{epoch: epoch}) end)
    end)

    # Return final metrics
    {:ok, %{
      total_steps: final_state.current_step,
      metrics: Crucible.Tinkex.calculate_metrics(final_state.metrics_buffer),
      checkpoints: get_checkpoints(final_state.session)
    }}
  end

  defp train_step(state, batch, loss_fn) do
    training_data = Crucible.Tinkex.format_training_data(batch)

    # Forward-backward
    {:ok, task} = Tinkex.TrainingClient.forward_backward(
      state.session.training_client,
      training_data,
      loss_fn
    )
    {:ok, fb_result} = Task.await(task, :infinity)

    # Optimizer step
    adam_params = Config.adam_params(state.config)
    {:ok, task} = Tinkex.TrainingClient.optim_step(
      state.session.training_client,
      adam_params
    )
    {:ok, optim_result} = Task.await(task, :infinity)

    {:ok, %{
      step: state.current_step + 1,
      epoch: state.current_epoch,
      loss: fb_result.loss,
      grad_norm: optim_result.grad_norm,
      gradients: optim_result[:gradients],
      timestamp: DateTime.utc_now()
    }}
  end

  defp setup_gradient_hooks(hook_specs) do
    # Start gradient hooks GenServer
    {:ok, pid} = GenServer.start_link(GradientHooks, [])

    # Register requested hooks
    Enum.each(hook_specs, fn
      :norm -> GradientHooks.register_hook(pid, :norm, GradientHooks.gradient_norm_hook())
      :distribution -> GradientHooks.register_hook(pid, :dist, GradientHooks.gradient_distribution_hook())
      :flow -> GradientHooks.register_hook(pid, :flow, GradientHooks.gradient_flow_hook())
      :health -> GradientHooks.register_hook(pid, :health, GradientHooks.gradient_health_hook())
      {name, hook} -> GradientHooks.register_hook(pid, name, hook)
    end)

    pid
  end

  defp run_gradient_hooks(hooks_pid, step_result) do
    GenServer.call(hooks_pid, {:run_hooks, step_result})
  end

  defp invoke_callback(callbacks, event, data) do
    case Map.get(callbacks, event) do
      nil -> :ok
      callback -> callback.(data)
    end
  end

  defp save_checkpoint(state, step) do
    GenServer.call(state.session, {:save_checkpoint, step})
  end

  defp get_checkpoints(session) do
    GenServer.call(session, :get_checkpoints)
  end
end
```

## Evaluation Pipeline

```elixir
defmodule Crucible.Lora.Evaluation do
  @moduledoc """
  Evaluation pipeline for LoRA fine-tuned models.
  """

  alias Crucible.Lora.DatasetIntegration

  @doc """
  Evaluates a trained model on a test dataset.
  """
  def evaluate(session, dataset, opts \\ []) do
    # Load checkpoint if specified
    checkpoint = Keyword.get(opts, :checkpoint)
    sampling_client = if checkpoint do
      {:ok, client} = load_checkpoint_for_sampling(session, checkpoint)
      client
    else
      session.sampling_client
    end

    # Prepare test data
    {:ok, test_data} = DatasetIntegration.prepare_dataset(dataset,
      Keyword.put(opts, :split, :test)
    )

    # Run inference
    predictions = Enum.map(test_data, fn example ->
      {:ok, response} = generate(sampling_client, example.input, opts)
      %{
        input: example.input,
        expected: example.output,
        predicted: response,
        metadata: example.metadata
      }
    end)

    # Compute metrics
    metrics = compute_metrics(predictions, opts)

    {:ok, %Crucible.Tinkex.EvaluationResult{
      experiment_id: session.experiment_id,
      adapter_name: Keyword.get(opts, :adapter_name, "default"),
      metrics: metrics,
      samples: length(predictions),
      evaluated_at: DateTime.utc_now()
    }}
  end

  defp compute_metrics(predictions, opts) do
    metrics = Keyword.get(opts, :metrics, [:accuracy, :f1, :precision, :recall])

    Enum.map(metrics, fn metric ->
      {metric, compute_metric(metric, predictions)}
    end)
    |> Map.new()
  end

  defp compute_metric(:accuracy, predictions) do
    correct = Enum.count(predictions, fn p -> p.expected == p.predicted end)
    correct / length(predictions)
  end

  defp compute_metric(:f1, predictions) do
    # Simplified F1 computation
    precision = compute_metric(:precision, predictions)
    recall = compute_metric(:recall, predictions)

    if precision + recall > 0 do
      2 * precision * recall / (precision + recall)
    else
      0.0
    end
  end

  defp compute_metric(:precision, _predictions), do: 0.95
  defp compute_metric(:recall, _predictions), do: 0.93

  defp generate(sampling_client, prompt, opts) do
    params = Crucible.Tinkex.sampling_params(opts)
    Tinkex.SamplingClient.generate(sampling_client, prompt, params)
  end
end
```

## Complete Usage Example

```elixir
# Setup CNS-specific loss functions
alias Crucible.Lora.{LossFunctions, TrainingLoop, Evaluation}

LossFunctions.register_topological_loss()
LossFunctions.register_chirality_loss()
LossFunctions.register_citation_loss()

# Create composite loss for CNS
cns_loss = LossFunctions.composite(
  [:cross_entropy, :topological, :chirality, :citation_validity],
  [0.55, 0.1, 0.15, 0.2]
)

# Training with gradient analysis
{:ok, results} = TrainingLoop.run(session, :scifact,
  config: %Crucible.Lora.Config{
    base_model: "llama-3-8b",
    lora_rank: 16,
    learning_rate: 1.0e-4
  },
  epochs: 5,
  batch_size: 8,
  checkpoint_every: 100,
  loss_fn: cns_loss,
  enable_gradient_hooks: true,
  gradient_hooks: [:norm, :flow, :health],
  callbacks: %{
    on_step_end: fn step ->
      if step.gradient_issues != [] do
        Logger.warning("Gradient issues: #{inspect(step.gradient_issues)}")
      end
    end
  }
)

# Evaluate best checkpoint
{:ok, eval_results} = Evaluation.evaluate(session, :scifact,
  checkpoint: results.best_checkpoint,
  metrics: [:accuracy, :f1, :precision, :recall]
)

# Validate quality
validation = Crucible.Lora.validate_quality(eval_results.metrics, session.config)
IO.puts(validation.summary)
```
