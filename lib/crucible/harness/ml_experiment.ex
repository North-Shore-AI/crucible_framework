defmodule Crucible.Harness.MLExperiment do
  @moduledoc """
  Configuration and DSL for ML experiments.

  Provides a structured way to define ML training experiments with stages,
  parameters for hyperparameter sweeps, and quality targets.

  ## Example

      {:ok, experiment} = MLExperiment.new(
        name: "Fine-tune Claim Extractor",
        description: "Fine-tune on SciFact",
        parameters: %{
          learning_rate: [1.0e-4, 2.0e-4],
          lora_rank: [16, 32]
        },
        quality_targets: %{
          schema_compliance: 0.95,
          citation_accuracy: 0.95
        }
      )

      experiment =
        experiment
        |> MLExperiment.add_stage(%{
          name: :train,
          type: :train,
          config: %{epochs: 5, batch_size: 8}
        })
        |> MLExperiment.add_stage(%{
          name: :eval,
          type: :eval,
          config: %{test_data: "scifact_dev"}
        })

      runs = MLExperiment.generate_runs(experiment)
      # => 4 runs (2 x 2 parameter combinations)

  """

  @type stage :: %{
          name: atom(),
          type: :train | :eval | :analysis,
          config: map()
        }

  @type t :: %__MODULE__{
          id: String.t(),
          name: String.t(),
          description: String.t(),
          stages: [stage()],
          parameters: map(),
          quality_targets: map(),
          output_dir: String.t() | nil,
          seed: integer() | nil
        }

  defstruct [
    :id,
    :name,
    :output_dir,
    :seed,
    description: "",
    stages: [],
    parameters: %{},
    quality_targets: %{}
  ]

  @doc """
  Creates a new experiment definition.

  ## Options

    * `:name` - Experiment name (required)
    * `:description` - Experiment description
    * `:parameters` - Hyperparameter definitions for sweep
    * `:quality_targets` - Quality target thresholds
    * `:output_dir` - Output directory for results
    * `:seed` - Random seed for reproducibility

  ## Examples

      {:ok, experiment} = MLExperiment.new(name: "my_experiment")

      {:ok, experiment} = MLExperiment.new(
        name: "sweep",
        parameters: %{lr: [1.0e-4, 2.0e-4]}
      )

  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(opts) when is_list(opts) do
    name = Keyword.get(opts, :name)

    if is_nil(name) or name == "" do
      {:error, "name is required"}
    else
      experiment = %__MODULE__{
        id: generate_id(),
        name: name,
        description: Keyword.get(opts, :description, ""),
        stages: [],
        parameters: Keyword.get(opts, :parameters, %{}),
        quality_targets: Keyword.get(opts, :quality_targets, %{}),
        output_dir: Keyword.get(opts, :output_dir),
        seed: Keyword.get(opts, :seed)
      }

      {:ok, experiment}
    end
  end

  @doc """
  Adds a stage to the experiment.

  Stages are executed in the order they are added.

  ## Stage Structure

    * `:name` - Stage name (atom)
    * `:type` - Stage type (:train, :eval, :analysis)
    * `:config` - Stage-specific configuration

  ## Examples

      experiment = MLExperiment.add_stage(experiment, %{
        name: :train,
        type: :train,
        config: %{epochs: 5}
      })

  """
  @spec add_stage(t() | {:ok, t()}, stage()) :: t() | {:ok, t()}
  def add_stage({:ok, experiment}, stage) do
    {:ok, do_add_stage(experiment, stage)}
  end

  def add_stage(%__MODULE__{} = experiment, stage) when is_map(stage) do
    do_add_stage(experiment, stage)
  end

  defp do_add_stage(experiment, stage) do
    %{experiment | stages: experiment.stages ++ [stage]}
  end

  @doc """
  Validates the experiment definition.

  Checks for required fields and valid configurations.

  ## Examples

      :ok = MLExperiment.validate(experiment)

      {:error, "epochs must be positive"} = MLExperiment.validate(invalid_experiment)

  """
  @spec validate(t()) :: :ok | {:error, String.t()}
  def validate(%__MODULE__{} = experiment) do
    cond do
      experiment.name == "" or is_nil(experiment.name) ->
        {:error, "name is required"}

      has_invalid_stages?(experiment.stages) ->
        {:error, "Stages must have name and type fields"}

      has_invalid_stage_config?(experiment.stages) ->
        {:error, "Invalid epochs or batch_size in stage config"}

      has_invalid_quality_targets?(experiment.quality_targets) ->
        {:error, "Quality targets must be between 0 and 1"}

      true ->
        :ok
    end
  end

  @doc """
  Generates all runs for a hyperparameter sweep.

  Each run represents a unique combination of parameters.

  ## Examples

      runs = MLExperiment.generate_runs(experiment)
      # => [
      #   %{run_id: "...", experiment_id: "...", params: %{lr: 1.0e-4}, status: :pending},
      #   %{run_id: "...", experiment_id: "...", params: %{lr: 2.0e-4}, status: :pending}
      # ]

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

    for params <- combinations do
      %{
        run_id: generate_id(),
        experiment_id: experiment.id,
        params: params,
        status: :pending
      }
    end
  end

  # Private functions

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp generate_param_combinations(parameters) when map_size(parameters) == 0 do
    []
  end

  defp generate_param_combinations(parameters) do
    params_list = Map.to_list(parameters)

    Enum.reduce(params_list, [[]], fn {key, values}, acc ->
      values_list = if is_list(values), do: values, else: [values]

      for combo <- acc, value <- values_list do
        [{key, value} | combo]
      end
    end)
    |> Enum.map(&Map.new/1)
  end

  defp has_invalid_stages?(stages) do
    Enum.any?(stages, fn stage ->
      not Map.has_key?(stage, :name) or not Map.has_key?(stage, :type)
    end)
  end

  defp has_invalid_stage_config?(stages) do
    Enum.any?(stages, fn stage ->
      config = Map.get(stage, :config, %{})

      epochs = Map.get(config, :epochs)
      batch_size = Map.get(config, :batch_size)

      (not is_nil(epochs) and epochs <= 0) or
        (not is_nil(batch_size) and batch_size <= 0)
    end)
  end

  defp has_invalid_quality_targets?(targets) when map_size(targets) == 0, do: false

  defp has_invalid_quality_targets?(targets) do
    Enum.any?(targets, fn {_key, value} ->
      value < 0 or value > 1
    end)
  end
end
