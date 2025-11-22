defmodule Crucible.Harness.TrainingReporter do
  @moduledoc """
  Generates reports from ML training experiments.

  Supports multiple output formats including Markdown, LaTeX, HTML, and JSON
  for publication-ready reports.

  ## Example

      {:ok, report} = TrainingReporter.generate(experiment_results)

      markdown = TrainingReporter.to_markdown(report)
      latex = TrainingReporter.to_latex(report)
      html = TrainingReporter.to_html(report)
      json = TrainingReporter.to_json(report)

      :ok = TrainingReporter.export(report, :markdown, "report.md")

  """

  @type format :: :markdown | :latex | :html | :json

  @type report :: %{
          experiment_id: String.t(),
          experiment_name: String.t(),
          sections: [map()],
          recommendations: [String.t()],
          generated_at: DateTime.t()
        }

  @doc """
  Generates a report from experiment results.

  ## Options

    * `:title` - Custom report title
    * `:include_recommendations` - Include recommendations section (default: true)

  ## Examples

      {:ok, report} = TrainingReporter.generate(results)

  """
  @spec generate(map(), keyword()) :: {:ok, report()} | {:error, term()}
  def generate(results, opts \\ []) do
    experiment_id = Map.get(results, :experiment_id, "unknown")
    experiment_name = Map.get(results, :experiment_name, "Unknown Experiment")

    sections = [
      build_summary_section(results),
      build_training_section(results),
      build_evaluation_section(results),
      build_quality_section(results)
    ]

    recommendations =
      if Keyword.get(opts, :include_recommendations, true) do
        build_recommendations(results)
      else
        []
      end

    report = %{
      experiment_id: experiment_id,
      experiment_name: experiment_name,
      sections: sections,
      recommendations: recommendations,
      generated_at: DateTime.utc_now()
    }

    {:ok, report}
  end

  @doc """
  Formats training metrics for display.

  ## Examples

      formatted = TrainingReporter.format_training_metrics(%{avg_loss: 0.5})

  """
  @spec format_training_metrics(map()) :: map()
  def format_training_metrics(metrics) when is_map(metrics) do
    Map.new(metrics, fn {key, value} ->
      formatted_value =
        if is_float(value) do
          Float.round(value, 4)
        else
          value
        end

      {key, formatted_value}
    end)
  end

  @doc """
  Formats evaluation metrics for display.

  ## Examples

      formatted = TrainingReporter.format_evaluation_metrics(%{accuracy: 0.95})

  """
  @spec format_evaluation_metrics(map()) :: map()
  def format_evaluation_metrics(metrics) when is_map(metrics) do
    Map.new(metrics, fn {key, value} ->
      formatted_value =
        if is_float(value) do
          Float.round(value, 4)
        else
          value
        end

      {key, formatted_value}
    end)
  end

  @doc """
  Assesses quality targets and returns pass/fail status for each.

  ## Examples

      assessment = TrainingReporter.format_quality_assessment(
        %{accuracy: 0.98},
        %{accuracy: 0.95}
      )
      # => [%{metric: :accuracy, target: 0.95, actual: 0.98, passed: true, delta: 0.03}]

  """
  @spec format_quality_assessment(map(), map()) :: [map()]
  def format_quality_assessment(evaluation, targets)
      when is_map(evaluation) and is_map(targets) do
    Enum.map(targets, fn {metric, target} ->
      actual = Map.get(evaluation, metric, 0)
      passed = actual >= target
      delta = Float.round(actual - target, 4)

      %{
        metric: metric,
        target: target,
        actual: Float.round(actual, 4),
        passed: passed,
        delta: delta
      }
    end)
  end

  @doc """
  Converts report to Markdown format.

  ## Examples

      markdown = TrainingReporter.to_markdown(report)

  """
  @spec to_markdown(report()) :: String.t()
  def to_markdown(report) do
    title = report.experiment_name

    header = """
    # #{title}

    **Experiment ID:** #{report.experiment_id}
    **Generated:** #{DateTime.to_iso8601(report.generated_at)}

    ---

    """

    sections_content =
      Enum.map_join(report.sections, "\n\n", fn section ->
        section_to_markdown(section)
      end)

    recommendations_content =
      if report.recommendations != [] do
        """

        ## Recommendations

        #{Enum.map_join(report.recommendations, "\n", fn rec -> "- #{rec}" end)}
        """
      else
        ""
      end

    header <> sections_content <> recommendations_content
  end

  @doc """
  Converts report to LaTeX format.

  ## Examples

      latex = TrainingReporter.to_latex(report)

  """
  @spec to_latex(report()) :: String.t()
  def to_latex(report) do
    title = escape_latex(report.experiment_name)

    """
    \\documentclass{article}
    \\usepackage{booktabs}
    \\usepackage{graphicx}

    \\title{#{title}}
    \\date{#{DateTime.to_iso8601(report.generated_at)}}

    \\begin{document}

    \\maketitle

    #{Enum.map_join(report.sections, "\n\n", &section_to_latex/1)}

    \\end{document}
    """
  end

  @doc """
  Converts report to HTML format.

  ## Examples

      html = TrainingReporter.to_html(report)

  """
  @spec to_html(report()) :: String.t()
  def to_html(report) do
    title = escape_html(report.experiment_name)

    """
    <!DOCTYPE html>
    <html>
    <head>
      <title>#{title}</title>
      <style>
        body { font-family: sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f4f4f4; }
        .passed { color: green; }
        .failed { color: red; }
      </style>
    </head>
    <body>
      <h1>#{title}</h1>
      <p><strong>Experiment ID:</strong> #{report.experiment_id}</p>
      <p><strong>Generated:</strong> #{DateTime.to_iso8601(report.generated_at)}</p>

      #{Enum.map_join(report.sections, "\n", &section_to_html/1)}
    </body>
    </html>
    """
  end

  @doc """
  Converts report to JSON format.

  ## Examples

      json = TrainingReporter.to_json(report)

  """
  @spec to_json(report()) :: String.t()
  def to_json(report) do
    json_data = %{
      experiment_id: report.experiment_id,
      experiment_name: report.experiment_name,
      sections:
        Enum.map(report.sections, fn section ->
          %{
            name: Atom.to_string(section.name),
            data: section.data
          }
        end),
      recommendations: report.recommendations,
      generated_at: DateTime.to_iso8601(report.generated_at)
    }

    Jason.encode!(json_data, pretty: true)
  end

  @doc """
  Exports report to a file in the specified format.

  ## Examples

      :ok = TrainingReporter.export(report, :markdown, "report.md")

  """
  @spec export(report(), format(), String.t()) :: :ok | {:error, term()}
  def export(report, format, path) do
    content =
      case format do
        :markdown -> to_markdown(report)
        :latex -> to_latex(report)
        :html -> to_html(report)
        :json -> to_json(report)
      end

    File.write(path, content)
  end

  # Private functions

  defp build_summary_section(results) do
    %{
      name: :summary,
      data: %{
        experiment_id: Map.get(results, :experiment_id),
        experiment_name: Map.get(results, :experiment_name)
      }
    }
  end

  defp build_training_section(results) do
    train = Map.get(results, :train, %{})

    %{
      name: :training,
      data: %{
        total_steps: Map.get(train, :total_steps, 0),
        epochs_completed: Map.get(train, :epochs_completed, 0),
        metrics: Map.get(train, :metrics, %{})
      }
    }
  end

  defp build_evaluation_section(results) do
    eval = Map.get(results, :eval, %{})

    %{
      name: :evaluation,
      data: format_evaluation_metrics(eval)
    }
  end

  defp build_quality_section(results) do
    eval = Map.get(results, :eval, %{})
    targets = Map.get(results, :quality_targets, %{})

    %{
      name: :quality,
      data: format_quality_assessment(eval, targets)
    }
  end

  defp build_recommendations(results) do
    eval = Map.get(results, :eval, %{})
    targets = Map.get(results, :quality_targets, %{})

    recommendations = []

    # Check each quality target
    recommendations =
      Enum.reduce(targets, recommendations, fn {metric, target}, acc ->
        actual = Map.get(eval, metric, 0)

        if actual < target do
          [
            "Consider improving #{metric}: current #{Float.round(actual, 3)}, target #{target}"
            | acc
          ]
        else
          acc
        end
      end)

    Enum.reverse(recommendations)
  end

  defp section_to_markdown(%{name: :summary, data: data}) do
    """
    ## Summary

    - **Experiment ID:** #{data.experiment_id}
    - **Name:** #{data.experiment_name}
    """
  end

  defp section_to_markdown(%{name: :training, data: data}) do
    metrics_table =
      data.metrics
      |> Enum.map(fn {k, v} -> "| #{k} | #{v} |" end)
      |> Enum.join("\n")

    """
    ## Training

    - **Total Steps:** #{data.total_steps}
    - **Epochs Completed:** #{data.epochs_completed}

    ### Metrics

    | Metric | Value |
    |--------|-------|
    #{metrics_table}
    """
  end

  defp section_to_markdown(%{name: :evaluation, data: data}) do
    rows =
      data
      |> Enum.map(fn {k, v} -> "| #{k} | #{v} |" end)
      |> Enum.join("\n")

    """
    ## Evaluation

    | Metric | Value |
    |--------|-------|
    #{rows}
    """
  end

  defp section_to_markdown(%{name: :quality, data: data}) do
    rows =
      data
      |> Enum.map(fn item ->
        status = if item.passed, do: "PASS", else: "FAIL"
        "| #{item.metric} | #{item.target} | #{item.actual} | #{status} |"
      end)
      |> Enum.join("\n")

    """
    ## Quality Assessment

    | Metric | Target | Actual | Status |
    |--------|--------|--------|--------|
    #{rows}
    """
  end

  defp section_to_markdown(%{name: name, data: data}) do
    """
    ## #{String.capitalize(to_string(name))}

    #{inspect(data)}
    """
  end

  defp section_to_latex(%{name: :training, data: data}) do
    """
    \\section{Training}

    Total Steps: #{data.total_steps}

    Epochs Completed: #{data.epochs_completed}

    \\begin{table}[h]
    \\centering
    \\begin{tabular}{lr}
    \\toprule
    Metric & Value \\\\
    \\midrule
    #{Enum.map_join(data.metrics, " \\\\\n", fn {k, v} -> "#{k} & #{v}" end)} \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """
  end

  defp section_to_latex(%{name: name, data: _data}) do
    "\\section{#{String.capitalize(to_string(name))}}\n"
  end

  defp section_to_html(%{name: :training, data: data}) do
    rows =
      data.metrics
      |> Enum.map(fn {k, v} -> "<tr><td>#{k}</td><td>#{v}</td></tr>" end)
      |> Enum.join("\n")

    """
    <h2>Training</h2>
    <p><strong>Total Steps:</strong> #{data.total_steps}</p>
    <p><strong>Epochs Completed:</strong> #{data.epochs_completed}</p>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      #{rows}
    </table>
    """
  end

  defp section_to_html(%{name: :quality, data: data}) do
    rows =
      data
      |> Enum.map(fn item ->
        class = if item.passed, do: "passed", else: "failed"
        status = if item.passed, do: "PASS", else: "FAIL"

        "<tr><td>#{item.metric}</td><td>#{item.target}</td><td>#{item.actual}</td><td class=\"#{class}\">#{status}</td></tr>"
      end)
      |> Enum.join("\n")

    """
    <h2>Quality Assessment</h2>
    <table>
      <tr><th>Metric</th><th>Target</th><th>Actual</th><th>Status</th></tr>
      #{rows}
    </table>
    """
  end

  defp section_to_html(%{name: name, data: _data}) do
    "<h2>#{String.capitalize(to_string(name))}</h2>\n"
  end

  defp escape_latex(text) do
    text
    |> String.replace("\\", "\\textbackslash{}")
    |> String.replace("&", "\\&")
    |> String.replace("%", "\\%")
    |> String.replace("$", "\\$")
    |> String.replace("#", "\\#")
    |> String.replace("_", "\\_")
    |> String.replace("{", "\\{")
    |> String.replace("}", "\\}")
    |> String.replace("~", "\\textasciitilde{}")
    |> String.replace("^", "\\textasciicircum{}")
  end

  defp escape_html(text) do
    text
    |> String.replace("&", "&amp;")
    |> String.replace("<", "&lt;")
    |> String.replace(">", "&gt;")
    |> String.replace("\"", "&quot;")
    |> String.replace("'", "&#39;")
  end
end
