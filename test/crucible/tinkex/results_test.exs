defmodule Crucible.Tinkex.ResultsTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.Results

  doctest Results

  describe "new/1" do
    test "creates empty results container" do
      results = Results.new(experiment_id: "exp-123")

      assert results.experiment_id == "exp-123"
      assert results.training_metrics == []
      assert results.evaluation_metrics == []
    end
  end

  describe "add_training_metric/2" do
    test "adds training metric to results" do
      results = Results.new(experiment_id: "exp-123")

      metric = %{step: 1, loss: 1.0, citation_invalid_rate: 0.0}
      results = Results.add_training_metric(results, metric)

      assert length(results.training_metrics) == 1
      assert hd(results.training_metrics).step == 1
    end

    test "adds multiple metrics" do
      results = Results.new(experiment_id: "exp-123")

      results =
        results
        |> Results.add_training_metric(%{step: 1, loss: 1.0})
        |> Results.add_training_metric(%{step: 2, loss: 0.8})
        |> Results.add_training_metric(%{step: 3, loss: 0.6})

      assert length(results.training_metrics) == 3
    end
  end

  describe "add_evaluation_result/2" do
    test "adds evaluation result to results" do
      results = Results.new(experiment_id: "exp-123")

      eval = %{
        adapter_name: "test-adapter",
        metrics: %{schema_compliance: 0.96, citation_accuracy: 0.97}
      }

      results = Results.add_evaluation_result(results, eval)

      assert length(results.evaluation_metrics) == 1
      assert hd(results.evaluation_metrics).metrics.schema_compliance == 0.96
    end
  end

  describe "summarize_training/1" do
    test "calculates training summary statistics" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_training_metric(%{step: 1, loss: 1.0, citation_invalid_rate: 0.1})
        |> Results.add_training_metric(%{step: 2, loss: 0.8, citation_invalid_rate: 0.05})
        |> Results.add_training_metric(%{step: 3, loss: 0.6, citation_invalid_rate: 0.0})

      summary = Results.summarize_training(results)

      assert_in_delta summary.mean_loss, 0.8, 0.01
      assert_in_delta summary.final_loss, 0.6, 0.01
      assert_in_delta summary.loss_reduction, 0.4, 0.01
      assert summary.total_steps == 3
    end

    test "handles empty training metrics" do
      results = Results.new(experiment_id: "exp-123")
      summary = Results.summarize_training(results)

      assert summary.mean_loss == nil
      assert summary.total_steps == 0
    end
  end

  describe "summarize_evaluation/1" do
    test "calculates evaluation summary" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          metrics: %{schema_compliance: 0.95, citation_accuracy: 0.96}
        })
        |> Results.add_evaluation_result(%{
          adapter_name: "v2",
          metrics: %{schema_compliance: 0.97, citation_accuracy: 0.98}
        })

      summary = Results.summarize_evaluation(results)

      assert summary.count == 2
      assert_in_delta summary.mean_schema_compliance, 0.96, 0.01
      assert_in_delta summary.mean_citation_accuracy, 0.97, 0.01
    end

    test "handles empty evaluation metrics" do
      results = Results.new(experiment_id: "exp-123")
      summary = Results.summarize_evaluation(results)

      assert summary.count == 0
    end
  end

  describe "best_run/1" do
    test "finds best run by target metric" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          run_id: "run-1",
          metrics: %{overall_pass_rate: 0.40}
        })
        |> Results.add_evaluation_result(%{
          adapter_name: "v2",
          run_id: "run-2",
          metrics: %{overall_pass_rate: 0.50}
        })
        |> Results.add_evaluation_result(%{
          adapter_name: "v3",
          run_id: "run-3",
          metrics: %{overall_pass_rate: 0.45}
        })

      best = Results.best_run(results, :overall_pass_rate)

      assert best.run_id == "run-2"
      assert best.metrics.overall_pass_rate == 0.50
    end

    test "returns nil for empty results" do
      results = Results.new(experiment_id: "exp-123")
      assert Results.best_run(results, :overall_pass_rate) == nil
    end
  end

  describe "to_report_data/1" do
    test "generates report-ready data structure" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_training_metric(%{step: 1, loss: 1.0})
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          metrics: %{schema_compliance: 0.96}
        })

      report_data = Results.to_report_data(results)

      assert report_data.experiment_id == "exp-123"
      assert Map.has_key?(report_data, :training_summary)
      assert Map.has_key?(report_data, :evaluation_summary)
    end
  end

  describe "to_csv/2" do
    test "exports training metrics to CSV format" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_training_metric(%{step: 1, loss: 1.0, citation_invalid_rate: 0.1})
        |> Results.add_training_metric(%{step: 2, loss: 0.8, citation_invalid_rate: 0.0})

      csv = Results.to_csv(results, :training)

      assert String.contains?(csv, "step,loss,citation_invalid_rate")
      assert String.contains?(csv, "1,1.0,0.1")
      assert String.contains?(csv, "2,0.8,0.0")
    end

    test "exports evaluation metrics to CSV format" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          metrics: %{schema_compliance: 0.96, citation_accuracy: 0.97}
        })

      csv = Results.to_csv(results, :evaluation)

      assert String.contains?(csv, "adapter_name")
      assert String.contains?(csv, "v1")
    end
  end

  describe "validate_against_targets/2" do
    test "validates results against quality targets" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          metrics: %{
            schema_compliance: 0.96,
            citation_accuracy: 0.97,
            mean_entailment: 0.55,
            overall_pass_rate: 0.48
          }
        })

      targets = %{
        schema_compliance: 0.95,
        citation_accuracy: 0.95,
        mean_entailment: 0.50,
        overall_pass_rate: 0.45
      }

      validation = Results.validate_against_targets(results, targets)

      assert validation.passed == true
      assert validation.passed_count == 4
    end

    test "detects failed quality targets" do
      results =
        Results.new(experiment_id: "exp-123")
        |> Results.add_evaluation_result(%{
          adapter_name: "v1",
          metrics: %{
            schema_compliance: 0.90,
            citation_accuracy: 0.85,
            mean_entailment: 0.40,
            overall_pass_rate: 0.30
          }
        })

      targets = %{
        schema_compliance: 0.95,
        citation_accuracy: 0.95,
        mean_entailment: 0.50,
        overall_pass_rate: 0.45
      }

      validation = Results.validate_against_targets(results, targets)

      assert validation.passed == false
      assert validation.passed_count == 0
    end
  end
end
