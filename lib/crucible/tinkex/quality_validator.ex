defmodule Crucible.Tinkex.QualityValidator do
  @moduledoc """
  Quality target validation for Tinkex experiments.

  This module validates evaluation results against predefined quality targets
  based on CNS3 experimental research. It supports continuous monitoring
  during training and final validation after evaluation.

  ## Default Quality Targets

  Based on CNS3 experiments:

  | Metric | Target | Description |
  |--------|--------|-------------|
  | schema_compliance | >= 0.95 | Outputs conform to expected structure |
  | citation_accuracy | >= 0.95 | Only valid citations from source corpus |
  | mean_entailment | >= 0.50 | Claims supported by evidence |
  | overall_pass_rate | >= 0.45 | Combined semantic quality |

  ## Examples

      # Validate evaluation results
      results = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55,
        overall_pass_rate: 0.48
      }

      validation = Crucible.Tinkex.QualityValidator.validate(results)
      # => %{passed: true, assessments: [...]}

      # Check if all targets pass
      Crucible.Tinkex.QualityValidator.passed?(results)
      # => true

  """

  @default_targets %{
    schema_compliance: 0.95,
    citation_accuracy: 0.95,
    mean_entailment: 0.50,
    overall_pass_rate: 0.45
  }

  @secondary_targets %{
    entailment_pass_rate: 0.45,
    similarity_pass_rate: 0.35
  }

  @training_targets %{
    loss_reduction: 0.95,
    citation_invalid_rate: 0.0,
    convergence_steps: 500
  }

  @doc """
  Validates evaluation results against quality targets.

  ## Options

  - `:targets` - Custom quality targets map (overrides defaults)
  - `:include_secondary` - Include secondary targets (default: false)

  ## Examples

      iex> results = %{schema_compliance: 0.96, citation_accuracy: 0.97, mean_entailment: 0.55, overall_pass_rate: 0.48}
      iex> validation = Crucible.Tinkex.QualityValidator.validate(results)
      iex> validation.passed
      true
  """
  @spec validate(map(), keyword()) :: map()
  def validate(results, opts \\ []) when is_map(results) do
    custom_targets = Keyword.get(opts, :targets)

    targets =
      if custom_targets do
        # Use only custom targets when provided
        custom_targets
      else
        if Keyword.get(opts, :include_secondary, false) do
          Map.merge(@default_targets, @secondary_targets)
        else
          @default_targets
        end
      end

    assessments =
      Enum.map(targets, fn {metric, target} ->
        actual = Map.get(results, metric, 0)
        passed = actual >= target

        %{
          metric: metric,
          target: target,
          actual: actual,
          passed: passed,
          delta: Float.round(actual - target, 4)
        }
      end)

    passed_assessments = Enum.filter(assessments, & &1.passed)
    failed_assessments = Enum.reject(assessments, & &1.passed)

    %{
      passed: length(failed_assessments) == 0,
      passed_count: length(passed_assessments),
      failed_count: length(failed_assessments),
      total_count: length(assessments),
      assessments: assessments,
      passed_assessments: passed_assessments,
      failed_assessments: failed_assessments,
      summary: generate_summary(passed_assessments, failed_assessments)
    }
  end

  @doc """
  Checks if all quality targets pass.

  ## Examples

      iex> results = %{schema_compliance: 0.96, citation_accuracy: 0.97, mean_entailment: 0.55, overall_pass_rate: 0.48}
      iex> Crucible.Tinkex.QualityValidator.passed?(results)
      true

      iex> results = %{schema_compliance: 0.80}
      iex> Crucible.Tinkex.QualityValidator.passed?(results)
      false
  """
  @spec passed?(map(), keyword()) :: boolean()
  def passed?(results, opts \\ []) do
    validate(results, opts).passed
  end

  @doc """
  Validates training metrics against training targets.

  ## Examples

      iex> metrics = %{loss_reduction: 0.98, citation_invalid_rate: 0.0, final_step: 320}
      iex> validation = Crucible.Tinkex.QualityValidator.validate_training(metrics)
      iex> validation.passed
      true
  """
  @spec validate_training(map(), keyword()) :: map()
  def validate_training(metrics, opts \\ []) when is_map(metrics) do
    custom_targets = Keyword.get(opts, :targets, %{})
    targets = Map.merge(@training_targets, custom_targets)

    assessments =
      Enum.map(targets, fn {metric, target} ->
        actual = Map.get(metrics, metric, 0)

        # For convergence_steps, lower is better
        passed =
          if metric == :convergence_steps do
            actual <= target
          else
            actual >= target
          end

        %{
          metric: metric,
          target: target,
          actual: actual,
          passed: passed
        }
      end)

    passed_count = Enum.count(assessments, & &1.passed)

    %{
      passed: passed_count == length(assessments),
      passed_count: passed_count,
      total_count: length(assessments),
      assessments: assessments
    }
  end

  @doc """
  Returns the default quality targets.

  ## Examples

      iex> targets = Crucible.Tinkex.QualityValidator.default_targets()
      iex> targets.schema_compliance
      0.95
  """
  @spec default_targets() :: map()
  def default_targets, do: @default_targets

  @doc """
  Returns secondary quality targets.

  ## Examples

      iex> targets = Crucible.Tinkex.QualityValidator.secondary_targets()
      iex> targets.entailment_pass_rate
      0.45
  """
  @spec secondary_targets() :: map()
  def secondary_targets, do: @secondary_targets

  @doc """
  Returns training quality targets.

  ## Examples

      iex> targets = Crucible.Tinkex.QualityValidator.training_targets()
      iex> targets.loss_reduction
      0.95
  """
  @spec training_targets() :: map()
  def training_targets, do: @training_targets

  @doc """
  Creates a monitoring callback for continuous quality checking.

  ## Options

  - `:warn_threshold` - Threshold below target to trigger warning
  - `:on_warning` - Callback function for warnings
  - `:on_failure` - Callback function for failures

  ## Examples

      callback = Crucible.Tinkex.QualityValidator.monitor_callback(
        warn_threshold: 0.1,
        on_warning: fn metrics -> Logger.warn("Warning: " <> inspect(metrics)) end
      )
  """
  @spec monitor_callback(keyword()) :: (map() -> :ok | :warn | :fail)
  def monitor_callback(opts \\ []) do
    warn_threshold = Keyword.get(opts, :warn_threshold, 0.1)
    on_warning = Keyword.get(opts, :on_warning, fn _ -> :ok end)
    on_failure = Keyword.get(opts, :on_failure, fn _ -> :ok end)

    fn metrics ->
      validation = validate(metrics)

      cond do
        validation.passed ->
          :ok

        Enum.any?(validation.assessments, fn a ->
          not a.passed and a.delta > -warn_threshold
        end) ->
          on_warning.(validation.failed_assessments)
          :warn

        true ->
          on_failure.(validation.failed_assessments)
          :fail
      end
    end
  end

  @doc """
  Generates a human-readable report of validation results.

  ## Examples

      iex> results = %{schema_compliance: 0.96, citation_accuracy: 0.97, mean_entailment: 0.55, overall_pass_rate: 0.48}
      iex> report = Crucible.Tinkex.QualityValidator.report(results)
      iex> String.contains?(report, "PASSED")
      true
  """
  @spec report(map(), keyword()) :: String.t()
  def report(results, opts \\ []) do
    validation = validate(results, opts)

    status = if validation.passed, do: "PASSED", else: "FAILED"

    header = """
    # Quality Validation Report

    **Status**: #{status}
    **Passed**: #{validation.passed_count}/#{validation.total_count}

    ## Assessments
    """

    assessments_text =
      validation.assessments
      |> Enum.map(fn a ->
        status_icon = if a.passed, do: "[OK]", else: "[FAIL]"
        delta_text = if a.delta >= 0, do: "+#{a.delta}", else: "#{a.delta}"

        "- #{status_icon} #{a.metric}: #{a.actual} (target: #{a.target}, delta: #{delta_text})"
      end)
      |> Enum.join("\n")

    header <> assessments_text
  end

  # Private helpers

  defp generate_summary(passed, failed) do
    total = length(passed) + length(failed)

    if length(failed) == 0 do
      "All #{total} quality targets met"
    else
      failed_names =
        failed
        |> Enum.map(& &1.metric)
        |> Enum.join(", ")

      "#{length(passed)}/#{total} targets met. Failed: #{failed_names}"
    end
  end
end
