defmodule Crucible.Harness.MLExperimentTest do
  use ExUnit.Case, async: true

  alias Crucible.Harness.MLExperiment

  describe "new/1" do
    test "creates experiment with name" do
      {:ok, experiment} = MLExperiment.new(name: "test_experiment")

      assert experiment.name == "test_experiment"
      assert experiment.stages == []
      assert experiment.parameters == %{}
    end

    test "creates experiment with all options" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "A test experiment",
          seed: 42,
          output_dir: "/tmp/test"
        )

      assert experiment.name == "test_experiment"
      assert experiment.description == "A test experiment"
      assert experiment.seed == 42
      assert experiment.output_dir == "/tmp/test"
    end

    test "returns error for missing name" do
      {:error, reason} = MLExperiment.new([])

      assert reason =~ "name"
    end
  end

  describe "DSL" do
    test "defines experiment with stages" do
      {:ok, experiment} =
        MLExperiment.new(name: "test")
        |> MLExperiment.add_stage(%{
          name: :train,
          type: :train,
          config: %{epochs: 5, batch_size: 8}
        })
        |> MLExperiment.add_stage(%{
          name: :eval,
          type: :eval,
          config: %{test_data: "dev_set"}
        })

      assert length(experiment.stages) == 2
      assert Enum.at(experiment.stages, 0).name == :train
      assert Enum.at(experiment.stages, 1).name == :eval
    end

    test "validates required fields" do
      {:ok, experiment} = MLExperiment.new(name: "test")

      result = MLExperiment.validate(experiment)

      assert result == :ok
    end

    test "validates stage configurations" do
      {:ok, experiment} =
        MLExperiment.new(name: "test")
        |> MLExperiment.add_stage(%{
          name: :train,
          type: :train,
          config: %{epochs: -1}
        })

      {:error, reason} = MLExperiment.validate(experiment)

      assert reason =~ "epochs" or reason =~ "invalid"
    end
  end

  describe "add_stage/2" do
    test "adds stage to experiment" do
      {:ok, experiment} = MLExperiment.new(name: "test")

      experiment =
        MLExperiment.add_stage(experiment, %{
          name: :train,
          type: :train,
          config: %{}
        })

      assert length(experiment.stages) == 1
    end

    test "preserves stage order" do
      {:ok, experiment} = MLExperiment.new(name: "test")

      experiment =
        experiment
        |> MLExperiment.add_stage(%{name: :first, type: :train, config: %{}})
        |> MLExperiment.add_stage(%{name: :second, type: :eval, config: %{}})
        |> MLExperiment.add_stage(%{name: :third, type: :analysis, config: %{}})

      names = Enum.map(experiment.stages, & &1.name)
      assert names == [:first, :second, :third]
    end
  end

  describe "generate_runs/1" do
    test "generates single run for no parameters" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          parameters: %{}
        )

      runs = MLExperiment.generate_runs(experiment)

      assert length(runs) == 1
      assert Enum.at(runs, 0).params == %{}
    end

    test "generates parameter combinations" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          parameters: %{
            learning_rate: [1.0e-4, 2.0e-4],
            lora_rank: [16, 32]
          }
        )

      runs = MLExperiment.generate_runs(experiment)

      # 2 x 2 = 4 combinations
      assert length(runs) == 4

      param_sets = Enum.map(runs, & &1.params) |> MapSet.new()

      assert MapSet.member?(param_sets, %{learning_rate: 1.0e-4, lora_rank: 16})
      assert MapSet.member?(param_sets, %{learning_rate: 1.0e-4, lora_rank: 32})
      assert MapSet.member?(param_sets, %{learning_rate: 2.0e-4, lora_rank: 16})
      assert MapSet.member?(param_sets, %{learning_rate: 2.0e-4, lora_rank: 32})
    end

    test "supports grid search" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          parameters: %{
            a: [1, 2, 3],
            b: [10, 20]
          }
        )

      runs = MLExperiment.generate_runs(experiment)

      # 3 x 2 = 6 combinations
      assert length(runs) == 6
    end

    test "assigns unique run IDs" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          parameters: %{
            learning_rate: [1.0e-4, 2.0e-4]
          }
        )

      runs = MLExperiment.generate_runs(experiment)

      run_ids = Enum.map(runs, & &1.run_id)
      assert length(Enum.uniq(run_ids)) == length(runs)
    end

    test "includes experiment_id in each run" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          parameters: %{x: [1, 2]}
        )

      runs = MLExperiment.generate_runs(experiment)

      Enum.each(runs, fn run ->
        assert run.experiment_id == experiment.id
      end)
    end
  end

  describe "validate/1" do
    test "validates valid experiment" do
      {:ok, experiment} = MLExperiment.new(name: "test")

      assert MLExperiment.validate(experiment) == :ok
    end

    test "validates stages have required fields" do
      {:ok, experiment} =
        MLExperiment.new(name: "test")
        |> MLExperiment.add_stage(%{config: %{}})

      {:error, reason} = MLExperiment.validate(experiment)

      assert reason =~ "name" or reason =~ "type"
    end

    test "validates quality targets are valid" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          quality_targets: %{
            accuracy: -1.0
          }
        )

      {:error, reason} = MLExperiment.validate(experiment)

      assert reason =~ "quality" or reason =~ "target"
    end
  end

  describe "quality_targets/1" do
    test "sets quality targets" do
      targets = %{
        schema_compliance: 0.95,
        citation_accuracy: 0.95,
        mean_entailment: 0.50,
        overall_pass_rate: 0.45
      }

      {:ok, experiment} =
        MLExperiment.new(
          name: "test",
          quality_targets: targets
        )

      assert experiment.quality_targets == targets
    end

    test "uses default targets when not specified" do
      {:ok, experiment} = MLExperiment.new(name: "test")

      assert experiment.quality_targets == %{}
    end
  end
end
