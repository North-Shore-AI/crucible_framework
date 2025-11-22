defmodule Crucible.Harness.TrainingReporterTest do
  use ExUnit.Case, async: true

  alias Crucible.Harness.TrainingReporter

  @sample_results %{
    experiment_id: "exp-123",
    experiment_name: "Test Experiment",
    train: %{
      total_steps: 100,
      epochs_completed: 5,
      metrics: %{
        avg_loss: 0.5,
        final_loss: 0.3
      }
    },
    eval: %{
      schema_compliance: 0.98,
      citation_accuracy: 0.96,
      mean_entailment: 0.55,
      overall_pass_rate: 0.48
    },
    quality_targets: %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      overall_pass_rate: 0.45
    }
  }

  describe "generate/2" do
    test "generates report with all sections" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      assert report.experiment_id == "exp-123"
      assert is_list(report.sections)
      assert length(report.sections) >= 3
    end

    test "includes training section" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      training_section = Enum.find(report.sections, &(&1.name == :training))
      assert training_section != nil
      assert training_section.data.total_steps == 100
    end

    test "includes evaluation section" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      eval_section = Enum.find(report.sections, &(&1.name == :evaluation))
      assert eval_section != nil
      assert eval_section.data.schema_compliance == 0.98
    end

    test "includes quality assessment section" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      quality_section = Enum.find(report.sections, &(&1.name == :quality))
      assert quality_section != nil
    end

    test "includes recommendations" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      assert Map.has_key?(report, :recommendations) or
               Enum.any?(report.sections, &(&1.name == :recommendations))
    end
  end

  describe "format_training_metrics/1" do
    test "formats training metrics" do
      metrics = %{
        avg_loss: 0.5,
        final_loss: 0.3,
        grad_norm: 1.2
      }

      formatted = TrainingReporter.format_training_metrics(metrics)

      assert is_map(formatted)
      assert Map.has_key?(formatted, :avg_loss)
    end

    test "handles missing metrics" do
      metrics = %{}

      formatted = TrainingReporter.format_training_metrics(metrics)

      assert is_map(formatted)
    end
  end

  describe "format_evaluation_metrics/1" do
    test "formats evaluation metrics" do
      metrics = %{
        schema_compliance: 0.98,
        citation_accuracy: 0.96
      }

      formatted = TrainingReporter.format_evaluation_metrics(metrics)

      assert is_map(formatted)
    end
  end

  describe "format_quality_assessment/1" do
    test "assesses quality targets" do
      evaluation = %{
        schema_compliance: 0.98,
        citation_accuracy: 0.96,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      targets = %{
        schema_compliance: 0.95,
        citation_accuracy: 0.95,
        mean_entailment: 0.50,
        overall_pass_rate: 0.45
      }

      assessment = TrainingReporter.format_quality_assessment(evaluation, targets)

      assert is_list(assessment)

      Enum.each(assessment, fn item ->
        assert Map.has_key?(item, :metric)
        assert Map.has_key?(item, :target)
        assert Map.has_key?(item, :actual)
        assert Map.has_key?(item, :passed)
      end)
    end

    test "marks targets as passed or failed" do
      evaluation = %{
        schema_compliance: 0.90,
        citation_accuracy: 0.98
      }

      targets = %{
        schema_compliance: 0.95,
        citation_accuracy: 0.95
      }

      assessment = TrainingReporter.format_quality_assessment(evaluation, targets)

      schema_item = Enum.find(assessment, &(&1.metric == :schema_compliance))
      citation_item = Enum.find(assessment, &(&1.metric == :citation_accuracy))

      assert schema_item.passed == false
      assert citation_item.passed == true
    end
  end

  describe "to_markdown/1" do
    test "formats as markdown" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      markdown = TrainingReporter.to_markdown(report)

      assert is_binary(markdown)
      assert String.contains?(markdown, "# ")
      assert String.contains?(markdown, "Test Experiment")
    end

    test "includes tables and charts" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      markdown = TrainingReporter.to_markdown(report)

      # Check for table formatting
      assert String.contains?(markdown, "|")
      assert String.contains?(markdown, "---")
    end

    test "includes metrics table" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      markdown = TrainingReporter.to_markdown(report)

      assert String.contains?(markdown, "Metric")
      assert String.contains?(markdown, "Target") or String.contains?(markdown, "Value")
    end
  end

  describe "to_latex/1" do
    test "formats as latex" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      latex = TrainingReporter.to_latex(report)

      assert is_binary(latex)
      assert String.contains?(latex, "\\")
    end

    test "includes proper document structure" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      latex = TrainingReporter.to_latex(report)

      assert String.contains?(latex, "\\section") or String.contains?(latex, "\\begin")
    end
  end

  describe "to_html/1" do
    test "formats as html" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      html = TrainingReporter.to_html(report)

      assert is_binary(html)
      assert String.contains?(html, "<")
      assert String.contains?(html, ">")
    end

    test "includes proper HTML structure" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      html = TrainingReporter.to_html(report)

      assert String.contains?(html, "<h1>") or String.contains?(html, "<div")
      assert String.contains?(html, "<table") or String.contains?(html, "<p>")
    end
  end

  describe "to_json/1" do
    test "formats as json" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      json = TrainingReporter.to_json(report)

      assert is_binary(json)
      assert {:ok, _} = Jason.decode(json)
    end

    test "includes all report data" do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      json = TrainingReporter.to_json(report)
      {:ok, decoded} = Jason.decode(json)

      assert Map.has_key?(decoded, "experiment_id")
      assert Map.has_key?(decoded, "sections")
    end
  end

  describe "export/3" do
    @tag :tmp_dir
    test "exports to file", %{tmp_dir: tmp_dir} do
      {:ok, report} = TrainingReporter.generate(@sample_results)

      path = Path.join(tmp_dir, "report.md")
      :ok = TrainingReporter.export(report, :markdown, path)

      assert File.exists?(path)
      content = File.read!(path)
      assert String.contains?(content, "Test Experiment")
    end
  end
end
