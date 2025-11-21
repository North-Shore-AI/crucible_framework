defmodule Crucible.Tinkex.QualityValidatorTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.QualityValidator

  doctest QualityValidator

  describe "validate/2" do
    test "passes when all targets are met" do
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      validation = QualityValidator.validate(results)

      assert validation.passed == true
      assert validation.passed_count == 4
      assert validation.failed_count == 0
    end

    test "fails when some targets are not met" do
      results = %{
        schema_compliance: 0.90,
        citation_accuracy: 0.85,
        mean_entailment: 0.40,
        overall_pass_rate: 0.30
      }

      validation = QualityValidator.validate(results)

      assert validation.passed == false
      assert validation.passed_count == 0
      assert validation.failed_count == 4
    end

    test "handles partial failures" do
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.90,
        mean_entailment: 0.55,
        overall_pass_rate: 0.40
      }

      validation = QualityValidator.validate(results)

      assert validation.passed == false
      assert validation.passed_count == 2
      assert validation.failed_count == 2
    end

    test "includes secondary targets when requested" do
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48,
        entailment_pass_rate: 0.50,
        similarity_pass_rate: 0.40
      }

      validation = QualityValidator.validate(results, include_secondary: true)

      assert validation.total_count == 6
      assert validation.passed == true
    end

    test "uses custom targets" do
      results = %{
        schema_compliance: 0.99,
        custom_metric: 0.80
      }

      validation =
        QualityValidator.validate(results,
          targets: %{schema_compliance: 0.98, custom_metric: 0.75}
        )

      assert validation.passed == true
    end

    test "calculates delta correctly" do
      results = %{
        schema_compliance: 0.90
      }

      validation = QualityValidator.validate(results, targets: %{schema_compliance: 0.95})
      assessment = hd(validation.assessments)

      assert_in_delta assessment.delta, -0.05, 0.001
    end
  end

  describe "passed?/2" do
    test "returns true when all targets pass" do
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      assert QualityValidator.passed?(results) == true
    end

    test "returns false when any target fails" do
      results = %{
        schema_compliance: 0.80
      }

      assert QualityValidator.passed?(results) == false
    end
  end

  describe "validate_training/2" do
    test "passes when training targets are met" do
      metrics = %{
        loss_reduction: 0.98,
        citation_invalid_rate: 0.0,
        convergence_steps: 320
      }

      validation = QualityValidator.validate_training(metrics)

      assert validation.passed == true
    end

    test "fails when convergence takes too long" do
      metrics = %{
        loss_reduction: 0.98,
        citation_invalid_rate: 0.0,
        convergence_steps: 600
      }

      validation = QualityValidator.validate_training(metrics)

      assert validation.passed == false
    end
  end

  describe "default_targets/0" do
    test "returns default quality targets" do
      targets = QualityValidator.default_targets()

      assert targets.schema_compliance == 0.95
      assert targets.citation_accuracy == 0.95
      assert targets.mean_entailment == 0.50
      assert targets.overall_pass_rate == 0.45
    end
  end

  describe "secondary_targets/0" do
    test "returns secondary quality targets" do
      targets = QualityValidator.secondary_targets()

      assert targets.entailment_pass_rate == 0.45
      assert targets.similarity_pass_rate == 0.35
    end
  end

  describe "training_targets/0" do
    test "returns training quality targets" do
      targets = QualityValidator.training_targets()

      assert targets.loss_reduction == 0.95
      assert targets.citation_invalid_rate == 0.0
    end
  end

  describe "monitor_callback/1" do
    test "returns ok for passing results" do
      callback = QualityValidator.monitor_callback()

      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      assert callback.(results) == :ok
    end

    test "returns fail for failing results" do
      callback = QualityValidator.monitor_callback()

      results = %{
        schema_compliance: 0.80
      }

      assert callback.(results) == :fail
    end

    test "calls on_failure callback" do
      test_pid = self()

      callback =
        QualityValidator.monitor_callback(
          on_failure: fn failed -> send(test_pid, {:failed, failed}) end
        )

      results = %{schema_compliance: 0.80}
      callback.(results)

      assert_receive {:failed, _failed_assessments}
    end
  end

  describe "report/2" do
    test "generates passing report" do
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      report = QualityValidator.report(results)

      assert String.contains?(report, "PASSED")
      assert String.contains?(report, "4/4")
      assert String.contains?(report, "[OK]")
    end

    test "generates failing report" do
      results = %{
        schema_compliance: 0.80
      }

      report = QualityValidator.report(results)

      assert String.contains?(report, "FAILED")
      assert String.contains?(report, "[FAIL]")
    end
  end
end
