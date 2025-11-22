defmodule Crucible.Telemetry.MLMetricsTest do
  use ExUnit.Case, async: true

  alias Crucible.Telemetry.MLMetrics

  setup do
    {:ok, collector} = MLMetrics.start_link(experiment_id: "test-exp-123")
    %{collector: collector}
  end

  describe "record_training_step/5" do
    test "records training metrics", %{collector: collector} do
      :ok = MLMetrics.record_training_step(collector, 1, 1, 2.5, 0.1)
      :ok = MLMetrics.record_training_step(collector, 2, 1, 2.3, 0.09)

      metrics = MLMetrics.get_metrics(collector)
      assert length(metrics) == 2
      assert hd(metrics).step == 2
      assert hd(metrics).loss == 2.3
    end

    test "calculates running aggregates", %{collector: collector} do
      :ok = MLMetrics.record_training_step(collector, 1, 1, 2.0, 0.1)
      :ok = MLMetrics.record_training_step(collector, 2, 1, 3.0, 0.2)
      :ok = MLMetrics.record_training_step(collector, 3, 1, 4.0, 0.3)

      aggregates = MLMetrics.get_aggregates(collector, :loss)
      assert aggregates.mean == 3.0
      assert aggregates.min == 2.0
      assert aggregates.max == 4.0
    end

    test "tracks epoch boundaries", %{collector: collector} do
      :ok = MLMetrics.record_training_step(collector, 1, 1, 2.5, 0.1)
      :ok = MLMetrics.record_training_step(collector, 2, 2, 2.3, 0.09)

      metrics = MLMetrics.get_metrics(collector, epoch: 2)
      assert length(metrics) == 1
      assert hd(metrics).epoch == 2
    end
  end

  describe "get_aggregates/2" do
    test "returns mean, min, max, percentiles", %{collector: collector} do
      for i <- 1..100 do
        :ok = MLMetrics.record_training_step(collector, i, 1, i * 0.01, 0.1)
      end

      aggregates = MLMetrics.get_aggregates(collector, :loss)
      assert aggregates.mean == 0.505
      assert aggregates.min == 0.01
      assert aggregates.max == 1.0
      assert Map.has_key?(aggregates, :p50)
      assert Map.has_key?(aggregates, :p95)
      assert Map.has_key?(aggregates, :p99)
    end

    test "returns empty aggregates for no data", %{collector: collector} do
      aggregates = MLMetrics.get_aggregates(collector, :loss)
      assert aggregates == %{count: 0}
    end
  end

  describe "record_checkpoint/3" do
    test "records checkpoint with metrics", %{collector: collector} do
      metrics = %{loss: 2.1, accuracy: 0.95}
      :ok = MLMetrics.record_checkpoint(collector, 100, metrics)

      summary = MLMetrics.get_training_summary(collector)
      assert length(summary.checkpoints) == 1
      assert hd(summary.checkpoints).step == 100
      assert hd(summary.checkpoints).metrics.accuracy == 0.95
    end
  end

  describe "record_inference/4" do
    test "records inference metrics", %{collector: collector} do
      :ok = MLMetrics.record_inference(collector, "model-a", 150, 100)
      :ok = MLMetrics.record_inference(collector, "model-a", 200, 120)

      stats = MLMetrics.get_inference_stats(collector, "model-a")
      assert stats.count == 2
      assert stats.mean_latency == 175.0
      assert stats.total_tokens == 220
    end

    test "tracks multiple models separately", %{collector: collector} do
      :ok = MLMetrics.record_inference(collector, "model-a", 100, 50)
      :ok = MLMetrics.record_inference(collector, "model-b", 200, 100)

      stats_a = MLMetrics.get_inference_stats(collector, "model-a")
      stats_b = MLMetrics.get_inference_stats(collector, "model-b")

      assert stats_a.mean_latency == 100.0
      assert stats_b.mean_latency == 200.0
    end
  end

  describe "record/4" do
    test "records generic events", %{collector: collector} do
      :ok = MLMetrics.record(collector, :ensemble, %{latency: 250}, %{strategy: :majority})

      metrics = MLMetrics.get_metrics(collector, type: :ensemble)
      assert length(metrics) == 1
      assert hd(metrics).measurements.latency == 250
    end
  end

  describe "flush/1" do
    test "clears all metrics", %{collector: collector} do
      :ok = MLMetrics.record_training_step(collector, 1, 1, 2.5, 0.1)
      :ok = MLMetrics.flush(collector)

      metrics = MLMetrics.get_metrics(collector)
      assert metrics == []
    end
  end

  describe "reset/1" do
    test "resets collector state", %{collector: collector} do
      :ok = MLMetrics.record_training_step(collector, 1, 1, 2.5, 0.1)
      :ok = MLMetrics.reset(collector)

      metrics = MLMetrics.get_metrics(collector)
      assert metrics == []

      aggregates = MLMetrics.get_aggregates(collector, :loss)
      assert aggregates == %{count: 0}
    end
  end
end
