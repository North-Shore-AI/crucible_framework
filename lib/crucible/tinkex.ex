defmodule Crucible.Tinkex do
  @moduledoc """
  High-level interface for Tinkex integration with Crucible Framework.

  This module provides a unified API for training LLMs using the Tinkex SDK
  while leveraging Crucible's experiment infrastructure for telemetry,
  statistical analysis, and result reporting.

  ## Features

  - High-level training abstractions with automatic checkpointing
  - Integration with Crucible telemetry for experiment tracking
  - Support for LoRA fine-tuning with configurable parameters
  - Multi-adapter ensemble creation
  - Quality target validation

  ## Quick Start

      # Create an experiment
      {:ok, experiment} = Crucible.Tinkex.create_experiment(
        name: "SciFact Fine-tuning",
        config: %{
          api_key: "your-key",
          base_url: "https://tinker.example.com"
        }
      )

      # Run training
      {:ok, metrics} = Crucible.Tinkex.train(session, dataset,
        epochs: 5,
        batch_size: 8,
        checkpoint_every: 100
      )

  ## Integration with Crucible Components

  - **Telemetry**: All training events are captured via `crucible_telemetry`
  - **Bench**: Use `crucible_bench` for statistical analysis of results
  - **Ensemble**: Create multi-adapter ensembles for improved reliability

  """

  @behaviour Crucible.Lora.Adapter

  alias Crucible.Tinkex.Config

  defmodule SessionStruct do
    @moduledoc """
    Represents an active training session with Tinkex.

    Contains references to the service client, training client,
    and configuration for the current experiment.
    """

    @type t :: %__MODULE__{
            experiment_id: String.t(),
            service: pid() | nil,
            training_client: pid() | nil,
            config: Config.t(),
            started_at: DateTime.t() | nil
          }

    defstruct [
      :experiment_id,
      :service,
      :training_client,
      :config,
      :started_at
    ]
  end

  defmodule TrainingMetrics do
    @moduledoc """
    Captures metrics from a training step.
    """

    @type t :: %__MODULE__{
            step: pos_integer(),
            epoch: pos_integer(),
            loss: float(),
            citation_invalid_rate: float(),
            timestamp: DateTime.t() | nil
          }

    defstruct [
      :step,
      :epoch,
      :loss,
      :citation_invalid_rate,
      :timestamp
    ]
  end

  defmodule EvaluationResult do
    @moduledoc """
    Contains evaluation results for a fine-tuned adapter.
    """

    @type t :: %__MODULE__{
            experiment_id: String.t(),
            adapter_name: String.t(),
            metrics: map(),
            samples: non_neg_integer(),
            evaluated_at: DateTime.t()
          }

    defstruct [
      :experiment_id,
      :adapter_name,
      :metrics,
      :samples,
      :evaluated_at
    ]
  end

  @doc """
  Generates a unique ID for experiments and sessions.

  Returns a 16-character alphanumeric string.

  ## Examples

      iex> id = Crucible.Tinkex.generate_id()
      iex> String.length(id)
      16
  """
  @impl Crucible.Lora.Adapter
  @spec generate_id() :: String.t()
  def generate_id do
    :crypto.strong_rand_bytes(8)
    |> Base.encode16(case: :lower)
  end

  @doc """
  Creates a new experiment with Crucible telemetry integration.

  ## Options

  - `:name` - Experiment name (required)
  - `:description` - Experiment description
  - `:config` - Crucible.Tinkex.Config struct or keyword options
  - `:tags` - List of tags for categorization

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.create_experiment(name: "Test")
      iex> exp.name
      "Test"

      iex> {:error, _} = Crucible.Tinkex.create_experiment([])
  """
  @impl Crucible.Lora.Adapter
  @spec create_experiment(keyword()) :: {:ok, map()} | {:error, String.t()}
  def create_experiment(opts) when is_list(opts) do
    name = Keyword.get(opts, :name)

    if is_nil(name) or name == "" do
      {:error, "experiment name is required"}
    else
      config =
        case Keyword.get(opts, :config) do
          %Config{} = c -> c
          nil -> Config.new()
          keyword_opts when is_list(keyword_opts) -> Config.new(keyword_opts)
          map when is_map(map) -> Config.new(Enum.to_list(map))
        end

      experiment_id = generate_id()

      experiment = %{
        id: experiment_id,
        name: name,
        description: Keyword.get(opts, :description, ""),
        config: Config.with_experiment_id(config, experiment_id),
        tags: Keyword.get(opts, :tags, []),
        status: :pending,
        created_at: DateTime.utc_now(),
        metrics: [],
        checkpoints: []
      }

      {:ok, experiment}
    end
  end

  @doc """
  Batches a dataset into chunks of the specified size.

  ## Examples

      iex> batches = Crucible.Tinkex.batch_dataset([1, 2, 3, 4, 5], 2)
      iex> length(batches)
      3

      iex> Crucible.Tinkex.batch_dataset([], 10)
      []
  """
  @impl Crucible.Lora.Adapter
  @spec batch_dataset(list(), pos_integer()) :: [[any()]]
  def batch_dataset(dataset, batch_size) when is_list(dataset) and batch_size > 0 do
    Enum.chunk_every(dataset, batch_size)
  end

  @doc """
  Formats training data for Tinkex forward_backward calls.

  ## Options

  - `:citation_validity_weight` - Weight for citation validity loss component

  ## Examples

      iex> batch = [%{input: "test", output: "result"}]
      iex> formatted = Crucible.Tinkex.format_training_data(batch)
      iex> is_list(formatted)
      true
  """
  @impl Crucible.Lora.Adapter
  @spec format_training_data(list(), keyword()) :: list()
  def format_training_data(batch, opts \\ []) when is_list(batch) do
    _citation_weight = Keyword.get(opts, :citation_validity_weight, 1.0)

    Enum.map(batch, fn example ->
      %{
        input: Map.get(example, :input, ""),
        output: Map.get(example, :output, ""),
        weight: Map.get(example, :weight, 1.0)
      }
    end)
  end

  @doc """
  Calculates aggregate metrics from training results.

  ## Examples

      iex> results = [%{loss: 1.0, citation_invalid_rate: 0.0}]
      iex> metrics = Crucible.Tinkex.calculate_metrics(results)
      iex> metrics.mean_loss
      1.0
  """
  @impl Crucible.Lora.Adapter
  @spec calculate_metrics(list()) :: map()
  def calculate_metrics([]) do
    %{
      mean_loss: nil,
      loss_reduction: nil,
      mean_citation_invalid_rate: nil,
      total_steps: 0
    }
  end

  def calculate_metrics(results) when is_list(results) do
    losses = Enum.map(results, & &1.loss)
    citation_rates = Enum.map(results, & &1.citation_invalid_rate)

    mean_loss = Enum.sum(losses) / length(losses)
    first_loss = hd(losses)
    last_loss = List.last(losses)
    loss_reduction = first_loss - last_loss

    mean_citation_rate = Enum.sum(citation_rates) / length(citation_rates)

    %{
      mean_loss: mean_loss,
      loss_reduction: loss_reduction,
      mean_citation_invalid_rate: mean_citation_rate,
      total_steps: length(results),
      min_loss: Enum.min(losses),
      max_loss: Enum.max(losses),
      final_loss: last_loss
    }
  end

  @doc """
  Validates evaluation results against quality targets.

  ## Examples

      iex> results = %{schema_compliance: 0.96, citation_accuracy: 0.97, mean_entailment: 0.55, overall_pass_rate: 0.48}
      iex> config = Crucible.Tinkex.Config.new()
      iex> validation = Crucible.Tinkex.validate_quality(results, config)
      iex> validation.passed
      true
  """
  @impl Crucible.Lora.Adapter
  @spec validate_quality(map(), Config.t()) :: map()
  def validate_quality(results, %Config{} = config) do
    targets = Config.quality_targets(config)

    assessments =
      Enum.map(targets, fn {metric, target} ->
        actual = Map.get(results, metric, 0)
        passed = actual >= target

        %{
          metric: metric,
          target: target,
          actual: actual,
          passed: passed,
          delta: Float.round(actual - target, 3)
        }
      end)

    all_passed = Enum.all?(assessments, & &1.passed)
    passed_count = Enum.count(assessments, & &1.passed)

    %{
      passed: all_passed,
      assessments: assessments,
      summary: "#{passed_count}/#{length(assessments)} quality targets met"
    }
  end

  @doc """
  Creates a sampling parameters map for text generation.

  ## Options

  - `:temperature` - Sampling temperature (default: 0.7)
  - `:top_p` - Nucleus sampling threshold (default: 0.95)
  - `:max_tokens` - Maximum tokens to generate (default: 512)

  ## Examples

      iex> params = Crucible.Tinkex.sampling_params(temperature: 0.5)
      iex> params.temperature
      0.5
  """
  @impl Crucible.Lora.Adapter
  @spec sampling_params(keyword()) :: map()
  def sampling_params(opts \\ []) do
    %{
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.95),
      max_tokens: Keyword.get(opts, :max_tokens, 512),
      stop_sequences: Keyword.get(opts, :stop_sequences, [])
    }
  end

  @doc """
  Generates a checkpoint name for the current training state.

  ## Examples

      iex> name = Crucible.Tinkex.checkpoint_name("exp-123", 100)
      iex> String.contains?(name, "exp-123")
      true
      iex> String.contains?(name, "step_100")
      true
  """
  @impl Crucible.Lora.Adapter
  @spec checkpoint_name(String.t(), pos_integer()) :: String.t()
  def checkpoint_name(experiment_id, step) do
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    "#{experiment_id}_step_#{step}_#{timestamp}"
  end

  # Session-based API implementations

  @doc """
  Starts a new training session for an experiment.

  ## Options

  The config map should contain:
  - `:id` - Experiment ID
  - `:name` - Experiment name
  - `:config` - Crucible.Tinkex.Config struct

  ## Examples

      {:ok, session} = Crucible.Tinkex.start_session(%{
        id: "exp-123",
        name: "My Experiment",
        config: config
      })
  """
  @impl true
  @spec start_session(map()) :: {:ok, pid()} | {:error, term()}
  def start_session(experiment) when is_map(experiment) do
    Crucible.Tinkex.Session.start_link(experiment: experiment)
  end

  @doc """
  Executes a forward-backward pass on a batch of training data.

  ## Options

  - `:loss_fn` - Loss function to use (default: :cross_entropy)
  - `:loss_fn_config` - Additional loss function configuration

  ## Examples

      batch = [%{input: "text", output: "target", weight: 1.0}]
      {:ok, metrics} = Crucible.Tinkex.forward_backward(session, batch)
  """
  @impl true
  @spec forward_backward(pid(), list(), keyword()) :: {:ok, map()} | {:error, term()}
  def forward_backward(session, batch, opts \\ []) do
    GenServer.call(session, {:forward_backward, batch, opts}, :infinity)
  end

  @doc """
  Performs an optimizer step with Adam parameters.

  ## Parameters

  The params map should contain:
  - `:lr` - Learning rate
  - `:beta1` - First moment decay (default: 0.9)
  - `:beta2` - Second moment decay (default: 0.999)
  - `:eps` - Epsilon for numerical stability (default: 1.0e-8)
  - `:weight_decay` - Weight decay coefficient (default: 0.01)

  ## Examples

      adam_params = %{lr: 0.0001, beta1: 0.9, beta2: 0.999, eps: 1.0e-8, weight_decay: 0.01}
      {:ok, result} = Crucible.Tinkex.optim_step(session, adam_params)
  """
  @impl true
  @spec optim_step(pid(), map(), keyword()) :: {:ok, map()} | {:error, term()}
  def optim_step(session, params, opts \\ []) do
    GenServer.call(session, {:optim_step, params, opts}, :infinity)
  end

  @doc """
  Saves a checkpoint at the current training step.

  ## Examples

      {:ok, checkpoint} = Crucible.Tinkex.save_checkpoint(session, 100)
      checkpoint.name  # "exp-123_step_100_1700000000"
  """
  @spec save_checkpoint(pid(), pos_integer()) :: {:ok, map()} | {:error, term()}
  def save_checkpoint(session, step) do
    GenServer.call(session, {:save_checkpoint, step}, :infinity)
  end

  @doc """
  Creates a sampling client from a saved checkpoint.

  ## Examples

      {:ok, checkpoint} = Crucible.Tinkex.save_checkpoint(session, 100)
      {:ok, sampler} = Crucible.Tinkex.create_sampler(session, checkpoint.name)
  """
  @impl true
  @spec create_sampler(pid(), String.t()) :: {:ok, pid()} | {:error, term()}
  def create_sampler(session, checkpoint_name) do
    GenServer.call(session, {:create_sampler, checkpoint_name}, :infinity)
  end

  @doc """
  Generates text samples from a prompt using the session's sampler.

  The session must have a sampler created via `create_sampler/2` first.

  ## Options

  - `:temperature` - Sampling temperature (default: 0.7)
  - `:top_p` - Nucleus sampling threshold (default: 0.95)
  - `:max_tokens` - Maximum tokens to generate (default: 512)
  - `:num_samples` - Number of samples to generate (default: 1)

  ## Examples

      {:ok, samples} = Crucible.Tinkex.sample(session, "Complete this:", temperature: 0.5)
  """
  @impl true
  @spec sample(pid(), String.t(), keyword()) :: {:ok, list(String.t())} | {:error, term()}
  def sample(session, prompt, opts \\ []) do
    GenServer.call(session, {:sample, prompt, opts}, :infinity)
  end
end
