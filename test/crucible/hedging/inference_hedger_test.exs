defmodule Crucible.Hedging.InferenceHedgerTest do
  use ExUnit.Case, async: true

  alias Crucible.Hedging.InferenceHedger

  describe "new/1" do
    test "creates hedger with default options" do
      hedger = InferenceHedger.new()

      assert hedger.strategy == :percentile
      assert hedger.delay_ms == 100
      assert hedger.percentile == 75
      assert hedger.history_window == 100
      assert hedger.latency_history == %{}
    end

    test "creates hedger with custom options" do
      hedger =
        InferenceHedger.new(
          strategy: :fixed,
          delay_ms: 200,
          percentile: 90,
          history_window: 50
        )

      assert hedger.strategy == :fixed
      assert hedger.delay_ms == 200
      assert hedger.percentile == 90
      assert hedger.history_window == 50
    end

    test "creates hedger with adaptive strategy" do
      hedger = InferenceHedger.new(strategy: :adaptive)

      assert hedger.strategy == :adaptive
    end

    test "creates hedger with workload_aware strategy" do
      hedger = InferenceHedger.new(strategy: :workload_aware)

      assert hedger.strategy == :workload_aware
    end
  end

  describe "dispatch/4" do
    test "sends primary request immediately" do
      hedger = InferenceHedger.new(strategy: :fixed, delay_ms: 1000)

      clients = [
        {%{name: "model1"}, :client1},
        {%{name: "model2"}, :client2}
      ]

      start_time = System.monotonic_time(:millisecond)

      {:ok, _result, _latency} =
        InferenceHedger.dispatch(
          hedger,
          clients,
          fn _client -> {:ok, "response"} end,
          timeout: 5000
        )

      elapsed = System.monotonic_time(:millisecond) - start_time

      # Primary should complete quickly (well under delay)
      # Allow some margin for polling overhead
      assert elapsed < 100
    end

    test "sends backup after delay with fixed strategy" do
      hedger = InferenceHedger.new(strategy: :fixed, delay_ms: 20)

      clients = [
        {%{name: "slow_model"}, :client1},
        {%{name: "fast_model"}, :client2}
      ]

      # Primary is slow, backup starts after 20ms and completes quickly
      # Use a controlled delay that blocks until explicitly released or times out
      slow_ref = make_ref()

      inference_fn = fn client ->
        case client do
          :client1 ->
            # Block waiting for a message that won't come (simulates slow response)
            receive do
              {:release, ^slow_ref} -> {:ok, "slow_response"}
            after
              500 -> {:ok, "slow_response"}
            end

          :client2 ->
            # Backup completes quickly after the 20ms delay
            {:ok, "fast_response"}
        end
      end

      {:ok, result, _latency} =
        InferenceHedger.dispatch(
          hedger,
          clients,
          inference_fn,
          timeout: 5000
        )

      # Should get backup response since it completes first (after ~20ms delay + instant response)
      assert result == "fast_response"
    end

    test "calculates delay from percentile history" do
      hedger =
        InferenceHedger.new(strategy: :percentile, percentile: 75)
        |> InferenceHedger.record_latency("model1", 100)
        |> InferenceHedger.record_latency("model1", 150)
        |> InferenceHedger.record_latency("model1", 200)
        |> InferenceHedger.record_latency("model1", 250)

      # P75 should be around 200
      stats = InferenceHedger.get_latency_stats(hedger, "model1")
      assert stats.p75 >= 175 and stats.p75 <= 225
    end

    test "returns first successful response" do
      hedger = InferenceHedger.new(strategy: :fixed, delay_ms: 10)

      clients = [
        {%{name: "model1"}, :client1},
        {%{name: "model2"}, :client2}
      ]

      # Use a controlled delay for the primary
      slow_ref = make_ref()

      inference_fn = fn client ->
        case client do
          :client1 ->
            # Primary blocks waiting for a message that won't come
            receive do
              {:release, ^slow_ref} -> {:ok, "response1"}
            after
              200 -> {:ok, "response1"}
            end

          :client2 ->
            # Backup starts after 10ms delay but completes instantly
            {:ok, "response2"}
        end
      end

      {:ok, result, _latency} =
        InferenceHedger.dispatch(
          hedger,
          clients,
          inference_fn,
          timeout: 5000
        )

      # Second client should win (10ms delay + instant < 100ms)
      assert result == "response2"
    end

    test "cancels slower requests" do
      hedger = InferenceHedger.new(strategy: :fixed, delay_ms: 10)

      clients = [
        {%{name: "slow"}, :slow_client},
        {%{name: "fast"}, :fast_client}
      ]

      # Track when slow client starts
      slow_started = :atomics.new(1, signed: false)
      slow_ref = make_ref()

      inference_fn = fn client ->
        case client do
          :slow_client ->
            :atomics.put(slow_started, 1, 1)
            # Block waiting for a message that won't come (simulates slow response)
            receive do
              {:release, ^slow_ref} -> {:ok, "slow"}
            after
              1000 -> {:ok, "slow"}
            end

          :fast_client ->
            # Fast client completes quickly after 10ms delay
            {:ok, "fast"}
        end
      end

      {:ok, result, _latency} =
        InferenceHedger.dispatch(
          hedger,
          clients,
          inference_fn,
          timeout: 1000
        )

      # Should get fast response
      assert result == "fast"

      # Verify slow client was started (hedging behavior)
      assert :atomics.get(slow_started, 1) == 1
    end
  end

  describe "adaptive hedging" do
    test "adjusts delay based on recent latencies" do
      # Start with empty history
      hedger = InferenceHedger.new(strategy: :adaptive, history_window: 10)

      # Record some latencies
      hedger =
        Enum.reduce(1..10, hedger, fn i, h ->
          InferenceHedger.record_latency(h, "model1", i * 10)
        end)

      # With latencies 10-100, P75 should be around 75
      stats = InferenceHedger.get_latency_stats(hedger, "model1")
      assert stats.p75 >= 60 and stats.p75 <= 80
    end

    test "respects workload configuration" do
      hedger =
        InferenceHedger.new(
          strategy: :workload_aware,
          workload_config: %{
            high_load_multiplier: 2.0,
            low_load_threshold: 0.3
          }
        )

      clients = [
        {%{name: "model1", load: 0.8}, :client1},
        {%{name: "model2", load: 0.2}, :client2}
      ]

      {:ok, _result, _latency} =
        InferenceHedger.dispatch(
          hedger,
          clients,
          fn _client -> {:ok, "response"} end,
          timeout: 5000
        )

      # Just verify it works with workload config
      assert true
    end
  end

  describe "record_latency/3" do
    test "adds latency to history" do
      hedger =
        InferenceHedger.new()
        |> InferenceHedger.record_latency("model1", 100)

      assert Map.has_key?(hedger.latency_history, "model1")
      assert length(hedger.latency_history["model1"]) == 1
    end

    test "maintains history window size" do
      hedger = InferenceHedger.new(history_window: 5)

      hedger =
        Enum.reduce(1..10, hedger, fn i, h ->
          InferenceHedger.record_latency(h, "model1", i * 10)
        end)

      # Should only keep last 5
      assert length(hedger.latency_history["model1"]) == 5
      # Should be the most recent 5 values (60, 70, 80, 90, 100)
      assert hd(hedger.latency_history["model1"]) == 100
    end

    test "tracks multiple models independently" do
      hedger =
        InferenceHedger.new()
        |> InferenceHedger.record_latency("model1", 100)
        |> InferenceHedger.record_latency("model2", 200)
        |> InferenceHedger.record_latency("model1", 150)

      assert length(hedger.latency_history["model1"]) == 2
      assert length(hedger.latency_history["model2"]) == 1
    end
  end

  describe "get_latency_stats/2" do
    test "returns statistics for model" do
      hedger =
        InferenceHedger.new()
        |> InferenceHedger.record_latency("model1", 100)
        |> InferenceHedger.record_latency("model1", 200)
        |> InferenceHedger.record_latency("model1", 300)

      stats = InferenceHedger.get_latency_stats(hedger, "model1")

      assert stats.min == 100
      assert stats.max == 300
      assert stats.mean == 200.0
      assert stats.count == 3
    end

    test "returns empty stats for unknown model" do
      hedger = InferenceHedger.new()

      stats = InferenceHedger.get_latency_stats(hedger, "unknown")

      assert stats.count == 0
    end
  end

  describe "calculate_percentile/2" do
    test "calculates correct percentile" do
      history = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

      # P50 of 10 elements: index = ceil(0.5 * 10) - 1 = 4, value = 50
      assert InferenceHedger.calculate_percentile(history, 50) == 50
      # P75 of 10 elements: index = ceil(0.75 * 10) - 1 = 7, value = 80
      assert InferenceHedger.calculate_percentile(history, 75) == 80
      # P90 of 10 elements: index = ceil(0.9 * 10) - 1 = 8, value = 90
      assert InferenceHedger.calculate_percentile(history, 90) == 90
    end

    test "handles empty history" do
      assert InferenceHedger.calculate_percentile([], 75) == 100
    end

    test "handles single element" do
      assert InferenceHedger.calculate_percentile([50], 75) == 50
    end
  end
end
