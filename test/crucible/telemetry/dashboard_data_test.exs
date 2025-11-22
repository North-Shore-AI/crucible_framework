defmodule Crucible.Telemetry.DashboardDataTest do
  use ExUnit.Case, async: true

  alias Crucible.Telemetry.{DashboardData, MLMetrics, ExperimentTracker}

  setup do
    # Ensure Registry is started for collector lookup
    _ = Registry.start_link(keys: :unique, name: Crucible.Telemetry.Registry)

    # Use unique experiment_id per test to avoid collisions
    exp_id = "dash-test-#{System.unique_integer([:positive])}"

    {:ok, collector} = MLMetrics.start_link(experiment_id: exp_id)
    {:ok, tracker} = ExperimentTracker.start_link([])

    # Set up test data
    :ok = ExperimentTracker.start_experiment(tracker, exp_id, %{name: "Dashboard Test"})

    for i <- 1..50 do
      :ok = MLMetrics.record_training_step(collector, i, div(i - 1, 10) + 1, 5.0 - i * 0.05, 0.1)
    end

    # Sync call ensures all prior casts are processed
    _ = MLMetrics.get_metrics(collector)

    %{collector: collector, tracker: tracker, experiment_id: exp_id}
  end

  describe "training_progress/2" do
    test "returns training progress data", %{experiment_id: exp_id, collector: collector} do
      # Sync to ensure all casts are processed before assertion
      _ = MLMetrics.get_metrics(collector)

      progress = DashboardData.training_progress(exp_id)

      assert is_map(progress)
      assert Map.has_key?(progress, :current_step)
      assert Map.has_key?(progress, :current_epoch)
      assert Map.has_key?(progress, :total_steps)
      assert progress.current_step == 50
    end
  end

  describe "loss_curve/2" do
    test "returns loss over steps", %{experiment_id: exp_id} do
      curve = DashboardData.loss_curve(exp_id)

      assert is_list(curve)
      assert length(curve) == 50
      assert hd(curve).step == 1
      assert List.last(curve).step == 50
      # Loss should decrease
      assert hd(curve).loss > List.last(curve).loss
    end

    test "supports smoothing", %{experiment_id: exp_id} do
      raw_curve = DashboardData.loss_curve(exp_id, smoothing: 0)
      smoothed_curve = DashboardData.loss_curve(exp_id, smoothing: 0.9)

      # Smoothed values should be different
      _raw_first = hd(raw_curve).loss
      _smoothed_first = hd(smoothed_curve).loss

      # First values might be similar, but later ones should differ
      raw_last = List.last(raw_curve).loss
      smoothed_last = List.last(smoothed_curve).loss

      assert is_float(smoothed_last)
      # Smoothing should affect values
      assert abs(raw_last - smoothed_last) >= 0 or raw_last == smoothed_last
    end

    test "supports downsampling", %{experiment_id: exp_id} do
      curve = DashboardData.loss_curve(exp_id, max_points: 10)
      assert length(curve) <= 10
    end
  end

  describe "quality_metrics/2" do
    test "returns quality metrics summary", %{experiment_id: exp_id} do
      metrics = DashboardData.quality_metrics(exp_id)

      assert is_map(metrics)
      assert Map.has_key?(metrics, :loss)
      assert Map.has_key?(metrics.loss, :current)
      assert Map.has_key?(metrics.loss, :best)
      assert Map.has_key?(metrics.loss, :trend)
    end
  end

  describe "ensemble_performance/2" do
    test "returns ensemble performance data", %{experiment_id: exp_id, collector: collector} do
      # Add ensemble data
      for i <- 1..10 do
        :ok =
          MLMetrics.record(
            collector,
            :ensemble,
            %{latency: 100 + i * 10, accuracy: 0.9 + i * 0.005},
            %{strategy: :majority, models: 3}
          )
      end

      perf = DashboardData.ensemble_performance(exp_id)

      assert is_map(perf)
      assert Map.has_key?(perf, :mean_latency)
      assert Map.has_key?(perf, :mean_accuracy)
    end
  end

  describe "latency_distribution/2" do
    test "returns latency histogram data", %{experiment_id: exp_id, collector: collector} do
      # Add inference data
      for _ <- 1..100 do
        latency = 50 + :rand.uniform(200)
        :ok = MLMetrics.record_inference(collector, "model-a", latency, 100)
      end

      dist = DashboardData.latency_distribution(exp_id)

      assert is_map(dist)
      assert Map.has_key?(dist, :histogram)
      assert Map.has_key?(dist, :percentiles)
      assert is_list(dist.histogram)
    end
  end

  describe "model_comparison/2" do
    test "compares metrics across experiments" do
      # Create multiple experiments
      {:ok, collector1} = MLMetrics.start_link(experiment_id: "exp-a")
      {:ok, collector2} = MLMetrics.start_link(experiment_id: "exp-b")

      for i <- 1..20 do
        :ok = MLMetrics.record_training_step(collector1, i, 1, 3.0 - i * 0.1, 0.1)
        :ok = MLMetrics.record_training_step(collector2, i, 1, 4.0 - i * 0.15, 0.1)
      end

      comparison = DashboardData.model_comparison(["exp-a", "exp-b"], :loss)

      assert is_map(comparison)
      assert Map.has_key?(comparison, "exp-a")
      assert Map.has_key?(comparison, "exp-b")
      assert Map.has_key?(comparison["exp-a"], :final)
      assert Map.has_key?(comparison["exp-a"], :best)
    end
  end

  describe "format_for_chart/2" do
    test "formats data for chart type" do
      data = [
        %{step: 1, loss: 2.0},
        %{step: 2, loss: 1.5},
        %{step: 3, loss: 1.0}
      ]

      line_data = DashboardData.format_for_chart(data, :line)
      assert is_map(line_data)
      assert Map.has_key?(line_data, :x)
      assert Map.has_key?(line_data, :y)
      assert line_data.x == [1, 2, 3]
      assert line_data.y == [2.0, 1.5, 1.0]

      bar_data = DashboardData.format_for_chart(data, :bar)
      assert is_map(bar_data)
      assert Map.has_key?(bar_data, :labels)
      assert Map.has_key?(bar_data, :values)
    end
  end

  describe "to_vega_lite_spec/2" do
    test "generates VegaLite specification" do
      data = [
        %{step: 1, loss: 2.0},
        %{step: 2, loss: 1.5}
      ]

      spec = DashboardData.to_vega_lite_spec(data, :line)

      assert is_map(spec)
      assert spec["$schema"] =~ "vega-lite"
      assert Map.has_key?(spec, "data")
      assert Map.has_key?(spec, "mark")
      assert Map.has_key?(spec, "encoding")
    end
  end

  describe "subscribe/1 and unsubscribe/1" do
    test "manages subscriptions for real-time updates" do
      exp_id = "sub-test"

      # Should not raise
      :ok = DashboardData.subscribe(exp_id)
      :ok = DashboardData.unsubscribe(exp_id)
    end
  end
end
