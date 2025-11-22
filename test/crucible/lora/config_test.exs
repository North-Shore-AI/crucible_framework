defmodule Crucible.Lora.ConfigTest do
  use ExUnit.Case, async: true

  alias Crucible.Lora.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()

      assert config.base_model == "meta-llama/Llama-3.1-8B-Instruct"
      assert config.rank == 16
      assert config.alpha == 32.0
      assert config.dropout == 0.05
      assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
      assert config.learning_rate == 1.0e-4
      assert config.weight_decay == 0.01
      assert config.warmup_steps == 100
      assert config.max_grad_norm == 1.0
      assert config.adam_beta1 == 0.9
      assert config.adam_beta2 == 0.999
      assert config.adam_epsilon == 1.0e-8
      assert config.epochs == 3
      assert config.batch_size == 8
      assert config.checkpoint_interval == 100
    end

    test "overrides defaults with provided options" do
      config =
        Config.new(
          rank: 32,
          learning_rate: 5.0e-5,
          epochs: 10,
          batch_size: 16
        )

      assert config.rank == 32
      assert config.learning_rate == 5.0e-5
      assert config.epochs == 10
      assert config.batch_size == 16
      # Defaults preserved
      assert config.alpha == 32.0
    end

    test "validates rank is positive" do
      assert_raise ArgumentError, ~r/rank must be positive/, fn ->
        Config.new(rank: 0)
      end

      assert_raise ArgumentError, ~r/rank must be positive/, fn ->
        Config.new(rank: -1)
      end
    end

    test "validates learning_rate is positive" do
      assert_raise ArgumentError, ~r/learning_rate must be positive/, fn ->
        Config.new(learning_rate: 0)
      end

      assert_raise ArgumentError, ~r/learning_rate must be positive/, fn ->
        Config.new(learning_rate: -0.001)
      end
    end

    test "validates alpha is positive" do
      assert_raise ArgumentError, ~r/alpha must be positive/, fn ->
        Config.new(alpha: 0)
      end
    end

    test "validates dropout is in valid range" do
      assert_raise ArgumentError, ~r/dropout must be between 0 and 1/, fn ->
        Config.new(dropout: -0.1)
      end

      assert_raise ArgumentError, ~r/dropout must be between 0 and 1/, fn ->
        Config.new(dropout: 1.5)
      end
    end
  end

  describe "validate!/1" do
    test "returns config when valid" do
      config = %Config{rank: 16, learning_rate: 1.0e-4, alpha: 32.0, dropout: 0.05}
      assert Config.validate!(config) == config
    end

    test "raises for invalid config" do
      config = %Config{rank: 0, learning_rate: 1.0e-4, alpha: 32.0, dropout: 0.05}

      assert_raise ArgumentError, fn ->
        Config.validate!(config)
      end
    end
  end

  describe "to_adam_params/1" do
    test "converts to Tinkex AdamParams format" do
      config =
        Config.new(
          learning_rate: 2.0e-4,
          adam_beta1: 0.85,
          adam_beta2: 0.95,
          adam_epsilon: 1.0e-12
        )

      adam_params = Config.to_adam_params(config)

      assert adam_params.learning_rate == 2.0e-4
      assert adam_params.beta1 == 0.85
      assert adam_params.beta2 == 0.95
      assert adam_params.eps == 1.0e-12
    end

    test "returns struct compatible with Tinkex.Types.AdamParams" do
      config = Config.new()
      adam_params = Config.to_adam_params(config)

      assert is_struct(adam_params, Tinkex.Types.AdamParams)
    end
  end

  describe "to_tinkex_lora_config/1" do
    test "converts to Tinkex LoraConfig format" do
      config =
        Config.new(
          rank: 32,
          target_modules: ["q_proj", "v_proj"]
        )

      lora_config = Config.to_tinkex_lora_config(config)

      assert lora_config.rank == 32
      assert is_struct(lora_config, Tinkex.Types.LoraConfig)
    end
  end
end
