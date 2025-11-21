defmodule Crucible.Tinkex.Experiment do
  @moduledoc """
  Experiment definition and management for Tinkex training.

  This module provides a structured way to define, validate, and manage
  training experiments with Tinkex. It supports hyperparameter sweeps,
  quality targets, and integration with Crucible's experiment infrastructure.

  ## Example

      {:ok, experiment} = Crucible.Tinkex.Experiment.new(
        name: "SciFact Claim Extractor",
        base_model: "meta-llama/Llama-3.1-8B-Instruct",
        training: %{
          epochs: 5,
          batch_size: 8,
          learning_rate: 2.0e-4,
          lora_rank: 16
        },
        parameters: %{
          citation_validity_weight: [2.0, 5.0, 7.0]
        },
        quality_targets: %{
          schema_compliance: 0.95,
          citation_accuracy: 0.95
        }
      )

  ## Hyperparameter Sweeps

  Define parameter grids for automatic sweep:

      parameters: %{
        learning_rate: [1.0e-4, 2.0e-4],
        lora_rank: [16, 32]
      }

  This generates 4 runs (2 x 2 combinations).

  """

  alias Crucible.Tinkex

  @type training_config :: %{
          epochs: pos_integer(),
          batch_size: pos_integer(),
          learning_rate: float(),
          lora_rank: pos_integer(),
          lora_alpha: pos_integer(),
          lora_dropout: float(),
          checkpoint_every: pos_integer()
        }

  @type evaluation_config :: %{
          test_data: String.t(),
          max_samples: pos_integer(),
          metrics: [atom()]
        }

  @type t :: %__MODULE__{
          id: String.t(),
          name: String.t(),
          description: String.t(),
          base_model: String.t(),
          dataset: String.t() | nil,
          training: map(),
          evaluation: map(),
          parameters: map(),
          quality_targets: map(),
          repeat: pos_integer(),
          seed: integer() | nil,
          status: atom(),
          error: String.t() | nil,
          created_at: DateTime.t(),
          started_at: DateTime.t() | nil,
          completed_at: DateTime.t() | nil,
          results: map()
        }

  defstruct [
    :id,
    :name,
    :description,
    :base_model,
    :dataset,
    :error,
    :started_at,
    :completed_at,
    training: %{
      epochs: 3,
      batch_size: 8,
      learning_rate: 2.0e-4,
      lora_rank: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      checkpoint_every: 100
    },
    evaluation: %{
      test_data: nil,
      max_samples: 50,
      metrics: [:schema_compliance, :citation_accuracy, :mean_entailment, :overall_pass_rate]
    },
    parameters: %{},
    quality_targets: %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      overall_pass_rate: 0.45
    },
    repeat: 1,
    seed: nil,
    status: :pending,
    created_at: nil,
    results: %{}
  ]

  @doc """
  Creates a new experiment definition.

  ## Options

  - `:name` - Experiment name (required)
  - `:base_model` - Base model for fine-tuning (required)
  - `:description` - Experiment description
  - `:dataset` - Training dataset name
  - `:training` - Training configuration map
  - `:evaluation` - Evaluation configuration map
  - `:parameters` - Hyperparameter sweep definitions
  - `:quality_targets` - Quality target thresholds
  - `:repeat` - Number of times to repeat each run
  - `:seed` - Random seed for reproducibility

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> exp.status
      :pending

      iex> {:error, _} = Crucible.Tinkex.Experiment.new(name: "Test")
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(opts) when is_list(opts) do
    name = Keyword.get(opts, :name)
    base_model = Keyword.get(opts, :base_model)

    cond do
      is_nil(name) or name == "" ->
        {:error, "name is required"}

      is_nil(base_model) or base_model == "" ->
        {:error, "base_model is required"}

      true ->
        experiment = %__MODULE__{
          id: Tinkex.generate_id(),
          name: name,
          description: Keyword.get(opts, :description, ""),
          base_model: base_model,
          dataset: Keyword.get(opts, :dataset),
          training: merge_training_config(Keyword.get(opts, :training, %{})),
          evaluation: merge_evaluation_config(Keyword.get(opts, :evaluation, %{})),
          parameters: Keyword.get(opts, :parameters, %{}),
          quality_targets: merge_quality_targets(Keyword.get(opts, :quality_targets, %{})),
          repeat: Keyword.get(opts, :repeat, 1),
          seed: Keyword.get(opts, :seed),
          status: :pending,
          created_at: DateTime.utc_now(),
          results: %{}
        }

        {:ok, experiment}
    end
  end

  defp merge_training_config(custom) do
    default = %{
      epochs: 3,
      batch_size: 8,
      learning_rate: 2.0e-4,
      lora_rank: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      checkpoint_every: 100
    }

    Map.merge(default, custom)
  end

  defp merge_evaluation_config(custom) do
    default = %{
      test_data: nil,
      max_samples: 50,
      metrics: [:schema_compliance, :citation_accuracy, :mean_entailment, :overall_pass_rate]
    }

    Map.merge(default, custom)
  end

  defp merge_quality_targets(custom) do
    default = %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      overall_pass_rate: 0.45
    }

    Map.merge(default, custom)
  end

  @doc """
  Generates all runs for a hyperparameter sweep.

  Each run represents a unique combination of parameters.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model", repeat: 2)
      iex> runs = Crucible.Tinkex.Experiment.generate_runs(exp)
      iex> length(runs)
      2
  """
  @spec generate_runs(t()) :: [map()]
  def generate_runs(%__MODULE__{} = experiment) do
    param_combinations = generate_param_combinations(experiment.parameters)

    combinations =
      if param_combinations == [] do
        [%{}]
      else
        param_combinations
      end

    for params <- combinations,
        _rep <- 1..experiment.repeat do
      %{
        run_id: Tinkex.generate_id(),
        experiment_id: experiment.id,
        params: params,
        status: :pending
      }
    end
  end

  defp generate_param_combinations(parameters) when map_size(parameters) == 0 do
    []
  end

  defp generate_param_combinations(parameters) do
    params_list = Map.to_list(parameters)

    Enum.reduce(params_list, [[]], fn {key, values}, acc ->
      for combo <- acc, value <- values do
        [{key, value} | combo]
      end
    end)
    |> Enum.map(&Map.new/1)
  end

  @doc """
  Validates an experiment definition.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> Crucible.Tinkex.Experiment.validate(exp)
      :ok
  """
  @spec validate(t()) :: :ok | {:error, String.t()}
  def validate(%__MODULE__{} = experiment) do
    cond do
      experiment.training.epochs <= 0 ->
        {:error, "epochs must be positive"}

      experiment.training.batch_size <= 0 ->
        {:error, "batch_size must be positive"}

      experiment.repeat <= 0 ->
        {:error, "repeat must be positive"}

      true ->
        :ok
    end
  end

  @doc """
  Converts experiment to training configuration map.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "llama")
      iex> config = Crucible.Tinkex.Experiment.to_training_config(exp)
      iex> config.base_model
      "llama"
  """
  @spec to_training_config(t()) :: map()
  def to_training_config(%__MODULE__{} = experiment) do
    %{
      base_model: experiment.base_model,
      epochs: experiment.training.epochs,
      batch_size: experiment.training.batch_size,
      learning_rate: experiment.training.learning_rate,
      lora_rank: experiment.training.lora_rank,
      lora_alpha: experiment.training.lora_alpha,
      lora_dropout: experiment.training.lora_dropout,
      checkpoint_every: experiment.training.checkpoint_every
    }
  end

  @doc """
  Returns the quality targets for the experiment.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> targets = Crucible.Tinkex.Experiment.quality_targets(exp)
      iex> targets.schema_compliance
      0.95
  """
  @spec quality_targets(t()) :: map()
  def quality_targets(%__MODULE__{quality_targets: targets}), do: targets

  @doc """
  Starts the experiment.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> {:ok, exp} = Crucible.Tinkex.Experiment.start(exp)
      iex> exp.status
      :running
  """
  @spec start(t()) :: {:ok, t()} | {:error, String.t()}
  def start(%__MODULE__{status: :pending} = experiment) do
    {:ok, %{experiment | status: :running, started_at: DateTime.utc_now()}}
  end

  def start(%__MODULE__{status: status}) do
    {:error, "Cannot start experiment in #{status} status"}
  end

  @doc """
  Marks the experiment as completed.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> {:ok, exp} = Crucible.Tinkex.Experiment.start(exp)
      iex> {:ok, exp} = Crucible.Tinkex.Experiment.complete(exp, %{})
      iex> exp.status
      :completed
  """
  @spec complete(t(), map()) :: {:ok, t()} | {:error, String.t()}
  def complete(%__MODULE__{status: :running} = experiment, results) do
    {:ok,
     %{
       experiment
       | status: :completed,
         completed_at: DateTime.utc_now(),
         results: results
     }}
  end

  def complete(%__MODULE__{status: status}, _results) do
    {:error, "Cannot complete experiment in #{status} status"}
  end

  @doc """
  Marks the experiment as failed.

  ## Examples

      iex> {:ok, exp} = Crucible.Tinkex.Experiment.new(name: "Test", base_model: "model")
      iex> {:ok, exp} = Crucible.Tinkex.Experiment.start(exp)
      iex> {:ok, exp} = Crucible.Tinkex.Experiment.fail(exp, "Error message")
      iex> exp.status
      :failed
  """
  @spec fail(t(), String.t()) :: {:ok, t()} | {:error, String.t()}
  def fail(%__MODULE__{status: :running} = experiment, error) do
    {:ok,
     %{
       experiment
       | status: :failed,
         completed_at: DateTime.utc_now(),
         error: error
     }}
  end

  def fail(%__MODULE__{status: status}, _error) do
    {:error, "Cannot fail experiment in #{status} status"}
  end
end
