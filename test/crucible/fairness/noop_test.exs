defmodule Crucible.Fairness.NoopTest do
  use ExUnit.Case, async: true

  alias Crucible.Fairness.Noop

  describe "evaluate/4" do
    test "returns skipped status" do
      {:ok, result} = Noop.evaluate([1, 0], [1, 0], [0, 1], [])

      assert result.status == :skipped
      assert result.overall_passes == true
      assert result.passed_count == 0
      assert result.failed_count == 0
      assert result.metrics == %{}
    end

    test "ignores all input parameters" do
      {:ok, result1} = Noop.evaluate([], [], [], [])
      {:ok, result2} = Noop.evaluate([1, 2, 3], [1, 2, 3], [0, 0, 1], metrics: [:all])

      assert result1.status == :skipped
      assert result2.status == :skipped
    end
  end

  describe "generate_report/2" do
    test "returns skipped message for any format" do
      {:ok, report} = Noop.generate_report(%{status: :skipped}, :markdown)

      assert report =~ "skipped"
    end
  end
end
