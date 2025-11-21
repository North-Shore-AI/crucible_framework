defmodule Crucible.Lora do
  @moduledoc """
  Adapter-agnostic entry point for LoRA fine-tuning workflows.

  This module delegates to the configured adapter (default: `Crucible.Tinkex`)
  which implements the `Crucible.Lora.Adapter` behaviour. Consumers should
  call these functions instead of binding directly to a specific backend,
  enabling future adapters without API changes.
  """

  alias Crucible.Lora.Adapter

  @default_adapter Crucible.Tinkex

  @doc """
  Returns the currently configured adapter module.
  """
  @spec adapter_module() :: module()
  def adapter_module do
    Application.get_env(:crucible_framework, :lora_adapter, @default_adapter)
  end

  @doc """
  Generates a unique identifier via the active adapter.
  """
  @spec generate_id() :: String.t()
  def generate_id do
    adapter_module().generate_id()
  end

  @doc """
  Creates a new experiment using the active adapter.
  """
  @spec create_experiment(Adapter.options()) ::
          {:ok, Adapter.experiment()} | {:error, String.t()}
  def create_experiment(opts) do
    adapter_module().create_experiment(opts)
  end

  @doc """
  Splits a dataset into batches through the adapter.
  """
  @spec batch_dataset(list(), pos_integer()) :: [[any()]]
  def batch_dataset(dataset, batch_size) do
    adapter_module().batch_dataset(dataset, batch_size)
  end

  @doc """
  Normalizes a batch of examples for the adapter's preferred format.
  """
  @spec format_training_data(list(), Adapter.options()) :: list()
  def format_training_data(batch, opts \\ []) do
    adapter_module().format_training_data(batch, opts)
  end

  @doc """
  Aggregates training metrics according to the adapter implementation.
  """
  @spec calculate_metrics(list()) :: map()
  def calculate_metrics(results) do
    adapter_module().calculate_metrics(results)
  end

  @doc """
  Validates evaluation results with the adapter's quality targets.
  """
  @spec validate_quality(map(), any()) :: map()
  def validate_quality(results, config) do
    adapter_module().validate_quality(results, config)
  end

  @doc """
  Produces sampling parameters for text generation.
  """
  @spec sampling_params(Adapter.options()) :: map()
  def sampling_params(opts \\ []) do
    adapter_module().sampling_params(opts)
  end

  @doc """
  Builds a checkpoint name for the current training state.
  """
  @spec checkpoint_name(String.t(), pos_integer()) :: String.t()
  def checkpoint_name(experiment_id, step) do
    adapter_module().checkpoint_name(experiment_id, step)
  end
end
