defmodule Crucible.Tinkex.Config do
  @moduledoc """
  Configuration management for Crucible's Tinkex integration.

  This module provides configuration handling for connecting Crucible Framework
  with the Tinkex ML training and inference SDK. It manages API credentials,
  default parameters, and quality targets for experiments.

  ## Examples

      # Create config with defaults
      config = Crucible.Tinkex.Config.new()

      # Create config with explicit values
      config = Crucible.Tinkex.Config.new(
        api_key: "your-api-key",
        base_url: "https://tinker.example.com",
        timeout: 60_000
      )

      # Validate configuration
      :ok = Crucible.Tinkex.Config.validate(config)

  ## Configuration Options

  - `:api_key` - Tinkex API key (required for API calls)
  - `:base_url` - Tinkex service URL
  - `:timeout` - HTTP timeout in milliseconds (default: 120_000)
  - `:max_retries` - Maximum retry attempts (default: 3)
  - `:default_base_model` - Default base model for training
  - `:default_lora_rank` - Default LoRA rank for fine-tuning
  - `:quality_targets` - Map of quality target thresholds

  ## Application Environment

  Configuration can also be set via application environment:

      config :crucible_framework,
        api_key: "your-api-key",
        base_url: "https://tinker.example.com"

  """

  @app_env :crucible_framework

  @type t :: %__MODULE__{
          api_key: String.t() | nil,
          base_url: String.t() | nil,
          timeout: pos_integer(),
          max_retries: non_neg_integer(),
          default_base_model: String.t(),
          default_lora_rank: pos_integer(),
          quality_targets: map(),
          experiment_id: String.t() | nil,
          user_metadata: map() | nil
        }

  defstruct [
    :api_key,
    :base_url,
    :experiment_id,
    :user_metadata,
    timeout: 120_000,
    max_retries: 3,
    default_base_model: "meta-llama/Llama-3.1-8B-Instruct",
    default_lora_rank: 16,
    quality_targets: %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      overall_pass_rate: 0.45
    }
  ]

  @doc """
  Creates a new configuration struct.

  ## Options

  - `:api_key` - Tinkex API key
  - `:base_url` - Tinkex service base URL
  - `:timeout` - Request timeout in milliseconds (default: 120_000)
  - `:max_retries` - Maximum retry attempts (default: 3)
  - `:default_base_model` - Default model for training (default: "meta-llama/Llama-3.1-8B-Instruct")
  - `:default_lora_rank` - Default LoRA rank (default: 16)
  - `:quality_targets` - Map of quality thresholds
  - `:user_metadata` - Custom metadata to attach to requests

  ## Examples

      iex> config = Crucible.Tinkex.Config.new(api_key: "test", timeout: 60_000)
      iex> config.timeout
      60000

      iex> config = Crucible.Tinkex.Config.new()
      iex> config.default_lora_rank
      16
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      api_key: opts[:api_key] || Application.get_env(@app_env, :api_key),
      base_url: opts[:base_url] || Application.get_env(@app_env, :base_url),
      timeout: opts[:timeout] || 120_000,
      max_retries: opts[:max_retries] || 3,
      default_base_model:
        opts[:default_base_model] ||
          Application.get_env(
            @app_env,
            :default_base_model,
            "meta-llama/Llama-3.1-8B-Instruct"
          ),
      default_lora_rank:
        opts[:default_lora_rank] ||
          Application.get_env(@app_env, :default_lora_rank, 16),
      quality_targets:
        opts[:quality_targets] ||
          Application.get_env(@app_env, :quality_targets, %{
            schema_compliance: 0.95,
            citation_accuracy: 0.95,
            mean_entailment: 0.50,
            overall_pass_rate: 0.45
          }),
      experiment_id: opts[:experiment_id],
      user_metadata: opts[:user_metadata]
    }
  end

  @doc """
  Validates a configuration struct.

  Returns `:ok` if valid, or `{:error, reason}` if invalid.

  ## Examples

      iex> config = Crucible.Tinkex.Config.new(api_key: "test", base_url: "https://api.example.com")
      iex> Crucible.Tinkex.Config.validate(config)
      :ok

      iex> config = Crucible.Tinkex.Config.new()
      iex> Crucible.Tinkex.Config.validate(config)
      {:error, "api_key is required"}
  """
  @spec validate(t()) :: :ok | {:error, String.t()}
  def validate(%__MODULE__{} = config) do
    cond do
      is_nil(config.api_key) ->
        {:error, "api_key is required"}

      is_nil(config.base_url) ->
        {:error, "base_url is required"}

      not is_integer(config.timeout) or config.timeout <= 0 ->
        {:error, "timeout must be a positive integer"}

      not is_integer(config.max_retries) or config.max_retries < 0 ->
        {:error, "max_retries must be a non-negative integer"}

      true ->
        :ok
    end
  end

  @doc """
  Returns the quality targets from the configuration.

  Quality targets define the minimum thresholds for various metrics
  used in evaluating fine-tuned models.

  ## Examples

      iex> config = Crucible.Tinkex.Config.new()
      iex> targets = Crucible.Tinkex.Config.quality_targets(config)
      iex> targets.schema_compliance
      0.95
  """
  @spec quality_targets(t()) :: map()
  def quality_targets(%__MODULE__{quality_targets: targets}), do: targets

  @doc """
  Converts configuration to Tinkex-compatible options.

  Returns a keyword list suitable for passing to `Tinkex.Config.new/1`.

  ## Examples

      iex> config = Crucible.Tinkex.Config.new(api_key: "key", base_url: "https://api.test.com")
      iex> opts = Crucible.Tinkex.Config.to_tinkex_opts(config)
      iex> Keyword.get(opts, :api_key)
      "key"
  """
  @spec to_tinkex_opts(t()) :: keyword()
  def to_tinkex_opts(%__MODULE__{} = config) do
    [
      api_key: config.api_key,
      base_url: config.base_url,
      timeout: config.timeout,
      max_retries: config.max_retries,
      user_metadata: config.user_metadata
    ]
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
  end

  @doc """
  Associates an experiment ID with the configuration.

  This is used for telemetry isolation and result tracking.

  ## Examples

      iex> config = Crucible.Tinkex.Config.new(api_key: "test")
      iex> config = Crucible.Tinkex.Config.with_experiment_id(config, "exp-123")
      iex> config.experiment_id
      "exp-123"
  """
  @spec with_experiment_id(t(), String.t()) :: t()
  def with_experiment_id(%__MODULE__{} = config, experiment_id) when is_binary(experiment_id) do
    %{config | experiment_id: experiment_id}
  end

  @doc """
  Merges two configurations, with the second taking precedence for non-nil values.

  ## Examples

      iex> base = Crucible.Tinkex.Config.new(api_key: "key", timeout: 60_000)
      iex> override = Crucible.Tinkex.Config.new(timeout: 30_000)
      iex> merged = Crucible.Tinkex.Config.merge(base, override)
      iex> {merged.api_key, merged.timeout}
      {"key", 30000}
  """
  @spec merge(t(), t()) :: t()
  def merge(%__MODULE__{} = base, %__MODULE__{} = override) do
    %__MODULE__{
      api_key: override.api_key || base.api_key,
      base_url: override.base_url || base.base_url,
      timeout: if(override.timeout != 120_000, do: override.timeout, else: base.timeout),
      max_retries:
        if(override.max_retries != 3, do: override.max_retries, else: base.max_retries),
      default_base_model:
        if(
          override.default_base_model != "meta-llama/Llama-3.1-8B-Instruct",
          do: override.default_base_model,
          else: base.default_base_model
        ),
      default_lora_rank:
        if(override.default_lora_rank != 16,
          do: override.default_lora_rank,
          else: base.default_lora_rank
        ),
      quality_targets: Map.merge(base.quality_targets || %{}, override.quality_targets || %{}),
      experiment_id: override.experiment_id || base.experiment_id,
      user_metadata: Map.merge(base.user_metadata || %{}, override.user_metadata || %{})
    }
  end
end
