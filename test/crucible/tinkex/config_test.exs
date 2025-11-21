defmodule Crucible.Tinkex.ConfigTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.Config

  doctest Config

  describe "new/1" do
    test "creates config with default values" do
      config = Config.new()

      assert config.base_url == nil
      assert config.api_key == nil
      assert config.timeout == 120_000
      assert config.max_retries == 3
      assert config.default_base_model == "meta-llama/Llama-3.1-8B-Instruct"
      assert config.default_lora_rank == 16
    end

    test "creates config with custom values" do
      config =
        Config.new(
          api_key: "test-key",
          base_url: "https://api.example.com",
          timeout: 60_000,
          max_retries: 5
        )

      assert config.api_key == "test-key"
      assert config.base_url == "https://api.example.com"
      assert config.timeout == 60_000
      assert config.max_retries == 5
    end

    test "reads api_key from application env if not provided" do
      Application.put_env(:crucible_tinkex, :api_key, "env-key")
      on_exit(fn -> Application.delete_env(:crucible_tinkex, :api_key) end)

      config = Config.new()
      assert config.api_key == "env-key"
    end

    test "reads base_url from application env if not provided" do
      Application.put_env(:crucible_tinkex, :base_url, "https://env.example.com")
      on_exit(fn -> Application.delete_env(:crucible_tinkex, :base_url) end)

      config = Config.new()
      assert config.base_url == "https://env.example.com"
    end

    test "explicit values override application env" do
      Application.put_env(:crucible_tinkex, :api_key, "env-key")
      on_exit(fn -> Application.delete_env(:crucible_tinkex, :api_key) end)

      config = Config.new(api_key: "explicit-key")
      assert config.api_key == "explicit-key"
    end
  end

  describe "validate/1" do
    test "returns :ok for valid config with api_key" do
      config = Config.new(api_key: "test-key", base_url: "https://api.example.com")
      assert Config.validate(config) == :ok
    end

    test "returns error for missing api_key" do
      config = Config.new(base_url: "https://api.example.com")
      assert {:error, "api_key is required"} = Config.validate(config)
    end

    test "returns error for missing base_url" do
      config = Config.new(api_key: "test-key")
      assert {:error, "base_url is required"} = Config.validate(config)
    end

    test "returns error for invalid timeout" do
      config = Config.new(api_key: "test", base_url: "https://example.com", timeout: -1)
      assert {:error, "timeout must be a positive integer"} = Config.validate(config)
    end

    test "returns error for invalid max_retries" do
      config = Config.new(api_key: "test", base_url: "https://example.com", max_retries: -1)
      assert {:error, "max_retries must be a non-negative integer"} = Config.validate(config)
    end
  end

  describe "quality_targets/1" do
    test "returns default quality targets" do
      config = Config.new()
      targets = Config.quality_targets(config)

      assert targets.schema_compliance == 0.95
      assert targets.citation_accuracy == 0.95
      assert targets.mean_entailment == 0.50
      assert targets.overall_pass_rate == 0.45
    end

    test "returns custom quality targets from config" do
      config =
        Config.new(
          quality_targets: %{
            schema_compliance: 0.98,
            citation_accuracy: 0.99,
            mean_entailment: 0.60,
            overall_pass_rate: 0.50
          }
        )

      targets = Config.quality_targets(config)
      assert targets.schema_compliance == 0.98
      assert targets.citation_accuracy == 0.99
      assert targets.mean_entailment == 0.60
      assert targets.overall_pass_rate == 0.50
    end
  end

  describe "to_tinkex_config/1" do
    test "converts to Tinkex.Config format" do
      config =
        Config.new(
          api_key: "test-key",
          base_url: "https://api.example.com",
          timeout: 60_000,
          max_retries: 5
        )

      tinkex_opts = Config.to_tinkex_opts(config)

      assert Keyword.get(tinkex_opts, :api_key) == "test-key"
      assert Keyword.get(tinkex_opts, :base_url) == "https://api.example.com"
      assert Keyword.get(tinkex_opts, :timeout) == 60_000
      assert Keyword.get(tinkex_opts, :max_retries) == 5
    end
  end

  describe "with_experiment_id/2" do
    test "adds experiment_id to config" do
      config = Config.new(api_key: "test")
      new_config = Config.with_experiment_id(config, "exp-123")

      assert new_config.experiment_id == "exp-123"
    end
  end

  describe "merge/2" do
    test "merges two configs, preferring second" do
      base = Config.new(api_key: "base-key", timeout: 60_000)
      override = Config.new(timeout: 30_000)

      merged = Config.merge(base, override)

      assert merged.api_key == "base-key"
      assert merged.timeout == 30_000
    end
  end
end
