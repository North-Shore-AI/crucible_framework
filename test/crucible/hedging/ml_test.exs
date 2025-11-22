defmodule Crucible.Hedging.MLTest do
  use ExUnit.Case, async: true

  alias Crucible.Hedging.ML

  describe "dispatch/3" do
    test "uses hedger for inference" do
      clients = [
        {%{name: "model1", weight: 0.6}, :client1},
        {%{name: "model2", weight: 0.4}, :client2}
      ]

      inference_fn = fn _client -> {:ok, "test_response"} end

      config = %{strategy: :fixed, delay_ms: 100}

      {:ok, result, latency_ms} = ML.dispatch(clients, inference_fn, config)

      assert result == "test_response"
      assert is_integer(latency_ms)
      assert latency_ms >= 0
    end

    test "returns successful response" do
      clients = [
        {%{name: "model1"}, :client1}
      ]

      inference_fn = fn _client -> {:ok, %{text: "generated text", confidence: 0.95}} end

      {:ok, result, _latency} = ML.dispatch(clients, inference_fn, %{})

      assert result.text == "generated text"
      assert result.confidence == 0.95
    end

    test "handles inference errors" do
      clients = [
        {%{name: "model1"}, :client1}
      ]

      inference_fn = fn _client -> {:error, :timeout} end

      result = ML.dispatch(clients, inference_fn, %{timeout: 100})

      assert {:error, _} = result
    end

    test "supports percentile strategy" do
      clients = [
        {%{name: "model1"}, :client1},
        {%{name: "model2"}, :client2}
      ]

      inference_fn = fn _client -> {:ok, "response"} end

      config = %{strategy: :percentile_75}

      {:ok, _result, _latency} = ML.dispatch(clients, inference_fn, config)
    end

    test "supports adaptive strategy" do
      clients = [
        {%{name: "model1"}, :client1},
        {%{name: "model2"}, :client2}
      ]

      inference_fn = fn _client -> {:ok, "response"} end

      config = %{strategy: :adaptive, window_size: 50}

      {:ok, _result, _latency} = ML.dispatch(clients, inference_fn, config)
    end
  end

  describe "create_hedger/2" do
    test "creates hedger for fixed strategy" do
      hedger = ML.create_hedger(:fixed, delay_ms: 150)

      assert hedger.strategy == :fixed
      assert hedger.delay_ms == 150
    end

    test "creates hedger for percentile strategy" do
      hedger = ML.create_hedger(:percentile, percentile: 90)

      assert hedger.strategy == :percentile
      assert hedger.percentile == 90
    end

    test "creates hedger for adaptive strategy" do
      hedger = ML.create_hedger(:adaptive, history_window: 200)

      assert hedger.strategy == :adaptive
      assert hedger.history_window == 200
    end

    test "creates hedger for workload_aware strategy" do
      hedger = ML.create_hedger(:workload_aware, [])

      assert hedger.strategy == :workload_aware
    end
  end

  describe "create_router/2" do
    test "creates router with adaptive strategy" do
      models = [
        %{name: "model1", weight: 0.5},
        %{name: "model2", weight: 0.5}
      ]

      {:ok, router} = ML.create_router(models, strategy: :best_performing)

      assert Process.alive?(router)
      GenServer.stop(router)
    end

    test "creates router with default round_robin" do
      models = [
        %{name: "model1", weight: 1.0}
      ]

      {:ok, router} = ML.create_router(models)

      assert Process.alive?(router)
      GenServer.stop(router)
    end
  end

  describe "integration" do
    test "hedger and router work together" do
      models = [
        %{name: "model1", weight: 0.6},
        %{name: "model2", weight: 0.4}
      ]

      {:ok, router} = ML.create_router(models, strategy: :best_performing)
      hedger = ML.create_hedger(:fixed, delay_ms: 50)

      # Get clients from router (simulated)
      clients = Enum.map(models, fn model -> {model, make_ref()} end)

      # Run inference with hedging
      {:ok, _result, _latency} =
        ML.dispatch(
          clients,
          fn _client -> {:ok, "response"} end,
          %{strategy: hedger.strategy, delay_ms: hedger.delay_ms}
        )

      GenServer.stop(router)
    end
  end
end
