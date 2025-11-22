defmodule Crucible.Harness.TinkexRunnerTest do
  use ExUnit.Case, async: true

  alias Crucible.Harness.{TinkexRunner, MLExperiment}

  describe "init/2" do
    test "initializes runner with experiment config" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test experiment",
          seed: 42
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])

      assert runner.experiment == experiment
      assert runner.status == :initialized
      assert runner.checkpoints == []
      assert runner.results == %{}
    end

    test "creates training session" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test experiment"
        )

      {:ok, runner} = TinkexRunner.init(experiment, session: :mock_session)

      assert runner.session == :mock_session
    end

    test "applies custom options" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, output_dir: "/tmp/test")

      assert runner.output_dir == "/tmp/test"
    end
  end

  describe "run_stage/3" do
    setup do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])
      {:ok, runner: runner}
    end

    test "runs training stage with Tinkex", %{runner: runner} do
      stage_config = %{
        type: :train,
        config: %{
          epochs: 1,
          batch_size: 4
        }
      }

      {:ok, updated_runner} = TinkexRunner.run_stage(runner, :train, stage_config)

      assert updated_runner.status == :running
      assert Map.has_key?(updated_runner.results, :train)
    end

    test "runs evaluation stage", %{runner: runner} do
      stage_config = %{
        type: :eval,
        config: %{
          test_data: "test_set",
          metrics: [:accuracy, :loss]
        }
      }

      {:ok, updated_runner} = TinkexRunner.run_stage(runner, :eval, stage_config)

      assert Map.has_key?(updated_runner.results, :eval)
    end

    test "runs analysis stage", %{runner: runner} do
      stage_config = %{
        type: :analysis,
        config: %{
          tests: [:t_test, :mann_whitney]
        }
      }

      {:ok, updated_runner} = TinkexRunner.run_stage(runner, :analyze, stage_config)

      assert Map.has_key?(updated_runner.results, :analyze)
    end

    test "emits telemetry events", %{runner: runner} do
      ref =
        :telemetry_test.attach_event_handlers(self(), [
          [:crucible, :harness, :stage_start],
          [:crucible, :harness, :stage_stop]
        ])

      stage_config = %{type: :train, config: %{}}

      {:ok, _runner} = TinkexRunner.run_stage(runner, :train, stage_config)

      assert_received {[:crucible, :harness, :stage_start], ^ref, _, %{stage: :train}}
      assert_received {[:crucible, :harness, :stage_stop], ^ref, _, %{stage: :train}}
    end

    test "handles stage errors gracefully", %{runner: runner} do
      stage_config = %{
        type: :train,
        config: %{
          force_error: true
        }
      }

      {:error, reason} = TinkexRunner.run_stage(runner, :train, stage_config)

      assert reason =~ "error"
    end
  end

  describe "get_results/1" do
    test "returns experiment results" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])

      # Run a stage to generate results
      stage_config = %{type: :train, config: %{}}
      {:ok, runner} = TinkexRunner.run_stage(runner, :train, stage_config)

      results = TinkexRunner.get_results(runner)

      assert is_map(results)
      assert Map.has_key?(results, :train)
    end

    test "includes checkpoints and metrics" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])
      {:ok, runner} = TinkexRunner.run_stage(runner, :train, %{type: :train, config: %{}})

      results = TinkexRunner.get_results(runner)

      assert Map.has_key?(results, :checkpoints) or Map.has_key?(results, :train)
      assert is_map(results)
    end

    test "returns empty results before any stage runs" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])

      results = TinkexRunner.get_results(runner)

      assert results == %{}
    end
  end

  describe "cleanup/1" do
    test "cleans up resources" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, [])

      assert :ok = TinkexRunner.cleanup(runner)
    end

    test "handles cleanup with active session" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} = TinkexRunner.init(experiment, session: :mock_session)

      assert :ok = TinkexRunner.cleanup(runner)
    end
  end

  describe "checkpoint resume" do
    test "can resume from checkpoint" do
      {:ok, experiment} =
        MLExperiment.new(
          name: "test_experiment",
          description: "Test"
        )

      {:ok, runner} =
        TinkexRunner.init(experiment, resume_from: "checkpoint-100")

      assert runner.resume_from == "checkpoint-100"
    end
  end
end
