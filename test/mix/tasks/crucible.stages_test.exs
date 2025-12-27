defmodule Mix.Tasks.Crucible.StagesTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO

  alias Mix.Tasks.Crucible.Stages, as: StagesTask

  setup do
    # Save original stage_registry
    original = Application.get_env(:crucible_framework, :stage_registry)

    # Setup test stage registry
    Application.put_env(:crucible_framework, :stage_registry, %{
      bench: Crucible.Stage.Bench,
      validate: Crucible.Stage.Validate,
      data_checks: Crucible.Stage.DataChecks,
      guardrails: Crucible.Stage.Guardrails,
      report: Crucible.Stage.Report
    })

    on_exit(fn ->
      if original do
        Application.put_env(:crucible_framework, :stage_registry, original)
      else
        Application.delete_env(:crucible_framework, :stage_registry)
      end
    end)

    :ok
  end

  describe "run/1" do
    test "lists all stages when called without arguments" do
      output =
        capture_io(fn ->
          StagesTask.run([])
        end)

      assert output =~ "Available Stages:"
      assert output =~ ":bench"
      assert output =~ ":validate"
      assert output =~ "Crucible.Stage.Bench"
      assert output =~ "Crucible.Stage.Validate"
    end

    test "shows stage descriptions" do
      output =
        capture_io(fn ->
          StagesTask.run([])
        end)

      assert output =~ "Description:"
      assert output =~ "Statistical benchmarking"
    end

    test "shows required and optional fields" do
      output =
        capture_io(fn ->
          StagesTask.run([])
        end)

      assert output =~ "Required:"
      assert output =~ "Optional:"
    end

    test "shows details for specific stage with --name" do
      output =
        capture_io(fn ->
          StagesTask.run(["--name", "bench"])
        end)

      assert output =~ "Stage: :bench"
      assert output =~ "Module: Crucible.Stage.Bench"
      assert output =~ "Description:"
      assert output =~ "Required Options:"
      assert output =~ "Optional Options:"
      assert output =~ "tests"
      assert output =~ "alpha"
    end

    test "shows details for specific stage with -n alias" do
      output =
        capture_io(fn ->
          StagesTask.run(["-n", "validate"])
        end)

      assert output =~ "Stage: :validate"
      assert output =~ "Module: Crucible.Stage.Validate"
      assert output =~ "strict"
    end

    test "shows error for unknown stage" do
      output =
        capture_io(fn ->
          StagesTask.run(["--name", "unknown_stage"])
        end)

      assert output =~ "Unknown stage: :unknown_stage"
    end
  end

  describe "with no registry configured" do
    test "shows helpful message when no stages registered" do
      Application.delete_env(:crucible_framework, :stage_registry)

      output =
        capture_io(fn ->
          StagesTask.run([])
        end)

      assert output =~ "No stages registered"
    end
  end
end
