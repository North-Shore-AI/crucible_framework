defmodule Crucible.Fairness.Noop do
  @moduledoc """
  No-op fairness adapter for testing and when fairness evaluation is disabled.

  Returns a skipped status with no actual fairness evaluation performed.
  Used as the default adapter in crucible_framework; real implementations
  should be configured in integration apps.
  """

  @behaviour Crucible.Fairness.Adapter

  @impl true
  def evaluate(_predictions, _labels, _sensitive_attr, _opts) do
    {:ok,
     %{
       status: :skipped,
       metrics: %{},
       overall_passes: true,
       passed_count: 0,
       failed_count: 0,
       total_count: 0,
       message: "Fairness evaluation skipped (noop adapter)"
     }}
  end

  @impl true
  def generate_report(_evaluation_result, _format) do
    {:ok, "Fairness evaluation was skipped (noop adapter configured)."}
  end
end
