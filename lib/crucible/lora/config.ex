defmodule Crucible.Lora.Config do
  @moduledoc """
  LoRA training hyperparameters configuration.

  Provides a structured configuration for LoRA fine-tuning experiments with
  validation and conversion utilities for Tinkex types.

  ## Example

      config = Crucible.Lora.Config.new(
        rank: 32,
        learning_rate: 5.0e-5,
        epochs: 5
      )

      adam_params = Crucible.Lora.Config.to_adam_params(config)

  """

  @type t :: %__MODULE__{
          # Model config
          base_model: String.t(),
          rank: pos_integer(),
          alpha: float(),
          dropout: float(),
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

          # Training schedule
          epochs: pos_integer(),
          batch_size: pos_integer(),
          checkpoint_interval: pos_integer()
        }

  defstruct base_model: "meta-llama/Llama-3.1-8B-Instruct",
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"],
            learning_rate: 1.0e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            max_grad_norm: 1.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1.0e-8,
            epochs: 3,
            batch_size: 8,
            checkpoint_interval: 100

  @doc """
  Creates a new LoRA config with validation.

  ## Options

    * `:base_model` - Base model identifier (default: "meta-llama/Llama-3.1-8B-Instruct")
    * `:rank` - LoRA rank, must be positive (default: 16)
    * `:alpha` - LoRA alpha scaling factor, must be positive (default: 32.0)
    * `:dropout` - Dropout rate, must be in [0, 1] (default: 0.05)
    * `:learning_rate` - Learning rate, must be positive (default: 1.0e-4)
    * `:epochs` - Number of training epochs (default: 3)
    * `:batch_size` - Training batch size (default: 8)

  ## Examples

      iex> config = Crucible.Lora.Config.new(rank: 32)
      iex> config.rank
      32

      iex> Crucible.Lora.Config.new(rank: 0)
      ** (ArgumentError) rank must be positive

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    validate!(config)
  end

  @doc """
  Validates a config struct, raising on invalid values.

  ## Examples

      iex> config = %Crucible.Lora.Config{rank: 16, learning_rate: 1.0e-4}
      iex> Crucible.Lora.Config.validate!(config)
      %Crucible.Lora.Config{rank: 16, learning_rate: 1.0e-4, ...}

  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{rank: rank}) when rank < 1 do
    raise ArgumentError, "rank must be positive"
  end

  def validate!(%__MODULE__{learning_rate: lr}) when lr <= 0 do
    raise ArgumentError, "learning_rate must be positive"
  end

  def validate!(%__MODULE__{alpha: alpha}) when alpha <= 0 do
    raise ArgumentError, "alpha must be positive"
  end

  def validate!(%__MODULE__{dropout: dropout}) when dropout < 0 or dropout > 1 do
    raise ArgumentError, "dropout must be between 0 and 1"
  end

  def validate!(config), do: config

  @doc """
  Converts config to Tinkex AdamParams struct.

  ## Examples

      iex> config = Crucible.Lora.Config.new()
      iex> params = Crucible.Lora.Config.to_adam_params(config)
      iex> params.learning_rate
      1.0e-4

  """
  @spec to_adam_params(t()) :: Tinkex.Types.AdamParams.t()
  def to_adam_params(%__MODULE__{} = config) do
    %Tinkex.Types.AdamParams{
      learning_rate: config.learning_rate,
      beta1: config.adam_beta1,
      beta2: config.adam_beta2,
      eps: config.adam_epsilon
    }
  end

  @doc """
  Converts config to Tinkex LoraConfig struct.

  ## Examples

      iex> config = Crucible.Lora.Config.new(rank: 32)
      iex> lora_config = Crucible.Lora.Config.to_tinkex_lora_config(config)
      iex> lora_config.rank
      32

  """
  @spec to_tinkex_lora_config(t()) :: Tinkex.Types.LoraConfig.t()
  def to_tinkex_lora_config(%__MODULE__{} = config) do
    %Tinkex.Types.LoraConfig{
      rank: config.rank,
      train_mlp: "o_proj" in config.target_modules,
      train_attn: "q_proj" in config.target_modules or "v_proj" in config.target_modules
    }
  end
end
