defmodule Crucible.Stage.FairnessTest do
  use ExUnit.Case, async: true

  alias Crucible.Context
  alias Crucible.Stage.Fairness
  alias Crucible.IR.{Experiment, BackendRef, ReliabilityConfig, FairnessConfig}

  describe "run/2 with fairness disabled" do
    test "skips evaluation when fairness is disabled" do
      ctx = build_context(fairness_enabled: false)

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      assert new_ctx.metrics.fairness.status == :disabled
    end

    test "skips evaluation when fairness config is nil" do
      ctx = build_context(fairness_config: nil)

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      assert new_ctx.metrics.fairness.status == :disabled
    end
  end

  describe "run/2 with noop adapter" do
    test "returns skipped status with noop adapter" do
      ctx = build_context(fairness_enabled: true)

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      # With noop adapter and no data, should return no_data or skipped
      assert new_ctx.metrics.fairness.status in [:skipped, :no_data, :completed]
    end
  end

  describe "run/2 with explicit data" do
    test "evaluates fairness when data is provided in opts" do
      ctx = build_context(fairness_enabled: true)

      predictions = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
      labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
      sensitive = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

      opts = %{
        predictions: predictions,
        labels: labels,
        sensitive_attr: sensitive
      }

      {:ok, new_ctx} = Fairness.run(ctx, opts)

      # With noop adapter, should complete with skipped status
      assert new_ctx.metrics.fairness.status in [:skipped, :completed]
    end
  end

  describe "run/2 with data in context assigns" do
    test "extracts data from context assigns" do
      ctx =
        build_context(fairness_enabled: true)
        |> put_in_assigns(:fairness_predictions, [1, 0, 1, 0, 1])
        |> put_in_assigns(:fairness_labels, [1, 0, 1, 0, 1])
        |> put_in_assigns(:fairness_sensitive_attr, [0, 0, 1, 1, 1])

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      assert new_ctx.metrics.fairness.status in [:skipped, :completed]
    end
  end

  describe "run/2 with data in outputs" do
    test "extracts data from context outputs" do
      outputs = [
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 0},
        %{prediction: 1, label: 1, gender: 0},
        %{prediction: 0, label: 0, gender: 1},
        %{prediction: 1, label: 1, gender: 1}
      ]

      ctx =
        build_context(fairness_enabled: true, group_by: :gender)
        |> Map.put(:outputs, outputs)

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      assert new_ctx.metrics.fairness.status in [:skipped, :completed, :no_data]
    end

    test "handles missing data gracefully" do
      ctx =
        build_context(fairness_enabled: true)
        |> Map.put(:outputs, [])

      {:ok, new_ctx} = Fairness.run(ctx, %{})

      assert new_ctx.metrics.fairness.status == :no_data
    end
  end

  describe "describe/1" do
    test "returns stage description" do
      desc = Fairness.describe(%{metrics: [:demographic_parity], threshold: 0.15})

      assert desc.stage == :fairness
      assert desc.metrics == [:demographic_parity]
      assert desc.threshold == 0.15
    end

    test "uses defaults when no options provided" do
      desc = Fairness.describe(%{})

      assert desc.stage == :fairness
      assert is_list(desc.metrics)
      assert desc.threshold == 0.1
    end
  end

  # Test helpers

  defp build_context(opts) do
    fairness_config = build_fairness_config(opts)

    %Context{
      experiment_id: "test_exp",
      run_id: "test_run",
      experiment: %Experiment{
        id: "test_exp",
        backend: %BackendRef{id: :tinkex},
        pipeline: [],
        reliability: %ReliabilityConfig{
          fairness: fairness_config,
          ensemble: %Crucible.IR.EnsembleConfig{strategy: :none, members: []},
          hedging: %Crucible.IR.HedgingConfig{strategy: :off},
          guardrails: %Crucible.IR.GuardrailConfig{profiles: []},
          stats: %Crucible.IR.StatsConfig{tests: [], alpha: 0.05, options: %{}}
        }
      },
      outputs: [],
      metrics: %{},
      assigns: %{}
    }
  end

  defp build_fairness_config(opts) do
    case Keyword.get(opts, :fairness_config, :default) do
      nil ->
        nil

      :default ->
        %FairnessConfig{
          enabled: Keyword.get(opts, :fairness_enabled, false),
          group_by: Keyword.get(opts, :group_by, :sensitive_attr),
          metrics: Keyword.get(opts, :metrics, [:demographic_parity]),
          options: Keyword.get(opts, :fairness_options, %{threshold: 0.1})
        }

      config ->
        config
    end
  end

  defp put_in_assigns(ctx, key, value) do
    %Context{ctx | assigns: Map.put(ctx.assigns, key, value)}
  end
end
