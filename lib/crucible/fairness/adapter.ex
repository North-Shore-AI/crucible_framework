defmodule Crucible.Fairness.Adapter do
  @moduledoc """
  Behaviour describing the interface for fairness evaluation in Crucible experiments.

  Implementations bridge Crucible's experiment pipeline to fairness libraries like ExFairness.
  The adapter receives predictions, labels, and sensitive attributes, then returns a
  comprehensive fairness evaluation.

  ## Data Format

  The adapter expects data in a format that can be converted to Nx tensors:

    * `predictions` - Binary predictions (0 or 1), or probability scores
    * `labels` - Ground truth binary labels (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute values (0 or 1) for group identification

  ## Metrics

  Standard fairness metrics that implementations should support:

    * `:demographic_parity` - Equal positive prediction rates across groups
    * `:equalized_odds` - Equal TPR and FPR across groups
    * `:equal_opportunity` - Equal TPR across groups (for positive class)
    * `:predictive_parity` - Equal precision/PPV across groups

  ## Example Implementation

      defmodule MyApp.FairnessAdapter do
        @behaviour Crucible.Fairness.Adapter

        @impl true
        def evaluate(predictions, labels, sensitive_attr, opts) do
          metrics = Keyword.get(opts, :metrics, [:demographic_parity])
          threshold = Keyword.get(opts, :threshold, 0.1)

          # Convert to tensors and compute metrics...
          {:ok, %{
            metrics: %{demographic_parity: %{disparity: 0.05, passes: true}},
            overall_passes: true,
            passed_count: 1,
            failed_count: 0
          }}
        end
      end

  """

  @type predictions :: [number()] | Nx.Tensor.t()
  @type labels :: [number()] | Nx.Tensor.t()
  @type sensitive_attr :: [number()] | Nx.Tensor.t()
  @type opts :: keyword()

  @type metric_result :: %{
          required(:disparity) => float(),
          required(:passes) => boolean(),
          required(:threshold) => float(),
          required(:interpretation) => String.t(),
          optional(:error) => term()
        }

  @type evaluation_result :: %{
          required(:metrics) => %{atom() => metric_result()},
          required(:overall_passes) => boolean(),
          required(:passed_count) => non_neg_integer(),
          required(:failed_count) => non_neg_integer(),
          required(:total_count) => non_neg_integer(),
          optional(:violations) => [map()],
          optional(:report) => String.t()
        }

  @doc """
  Evaluates fairness across specified metrics.

  ## Parameters

    * `predictions` - Model predictions (binary or probability scores)
    * `labels` - Ground truth labels
    * `sensitive_attr` - Sensitive attribute for group identification
    * `opts` - Options:
      * `:metrics` - List of metrics to evaluate (default: all)
      * `:threshold` - Fairness threshold (default: 0.1)
      * `:min_per_group` - Minimum samples per group (default: 10)
      * `:fail_fast` - Stop on first failure (default: false)

  ## Returns

    * `{:ok, evaluation_result}` - Successful evaluation with metrics
    * `{:error, reason}` - Evaluation failed

  """
  @callback evaluate(predictions, labels, sensitive_attr, opts) ::
              {:ok, evaluation_result()} | {:error, term()}

  @doc """
  Generates a formatted fairness report.

  ## Parameters

    * `evaluation_result` - Result from `evaluate/4`
    * `format` - Output format (`:markdown`, `:json`, `:text`)

  ## Returns

    * `{:ok, report_string}` - Formatted report
    * `{:error, reason}` - Report generation failed

  """
  @callback generate_report(evaluation_result(), format :: atom()) ::
              {:ok, String.t()} | {:error, term()}

  @optional_callbacks generate_report: 2
end
