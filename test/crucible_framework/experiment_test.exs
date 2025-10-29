defmodule CrucibleFramework.ExperimentTest do
  use ExUnit.Case
  doctest CrucibleFramework.Experiment

  describe "validate/1" do
    test "validates a complete experiment config" do
      config = %{
        name: "Test Experiment",
        conditions: ["baseline", "treatment"],
        metrics: [:accuracy, :latency]
      }

      assert {:ok, ^config} = CrucibleFramework.Experiment.validate(config)
    end

    test "returns error for missing name" do
      config = %{
        conditions: ["baseline", "treatment"],
        metrics: [:accuracy, :latency]
      }

      assert {:error, "Missing required field: name"} =
               CrucibleFramework.Experiment.validate(config)
    end

    test "returns error for missing conditions" do
      config = %{
        name: "Test",
        metrics: [:accuracy]
      }

      assert {:error, "Missing required field: conditions"} =
               CrucibleFramework.Experiment.validate(config)
    end

    test "returns error for missing metrics" do
      config = %{
        name: "Test",
        conditions: ["a", "b"]
      }

      assert {:error, "Missing required field: metrics"} =
               CrucibleFramework.Experiment.validate(config)
    end
  end

  describe "new/1" do
    test "creates a new experiment config with all options" do
      {:ok, config} =
        CrucibleFramework.Experiment.new(
          name: "My Experiment",
          description: "Testing ensemble methods",
          conditions: ["baseline", "treatment"],
          metrics: [:accuracy, :latency],
          repeat: 3,
          seed: 42
        )

      assert config.name == "My Experiment"
      assert config.description == "Testing ensemble methods"
      assert config.conditions == ["baseline", "treatment"]
      assert config.metrics == [:accuracy, :latency]
      assert config.repeat == 3
      assert config.seed == 42
    end

    test "creates experiment with minimal options" do
      {:ok, config} =
        CrucibleFramework.Experiment.new(
          name: "Minimal",
          conditions: ["a"],
          metrics: [:m1]
        )

      assert config.name == "Minimal"
      assert config.conditions == ["a"]
      assert config.metrics == [:m1]
      assert config.repeat == 1
      assert is_integer(config.seed)
    end

    test "uses defaults for optional fields" do
      {:ok, config} =
        CrucibleFramework.Experiment.new(
          name: "Test",
          conditions: [],
          metrics: []
        )

      assert config.description == ""
      assert config.repeat == 1
      assert is_integer(config.seed)
    end
  end

  describe "__using__/1 macro" do
    defmodule TestExperiment do
      use CrucibleFramework.Experiment

      def name, do: "Test Experiment"
      def description, do: "A test experiment"
      def conditions, do: ["baseline", "treatment"]
      def metrics, do: [:accuracy, :latency]
    end

    test "creates config/0 function" do
      config = TestExperiment.config()
      assert is_map(config)
      assert config.name == "Test Experiment"
      assert config.description == "A test experiment"
      assert config.conditions == ["baseline", "treatment"]
      assert config.metrics == [:accuracy, :latency]
      assert config.status == :configured
    end

    test "provides default implementations" do
      defmodule MinimalExperiment do
        use CrucibleFramework.Experiment
      end

      config = MinimalExperiment.config()
      assert config.name == "Unnamed Experiment"
      assert config.description == "No description provided"
      assert config.conditions == []
      assert config.metrics == []
    end
  end
end
