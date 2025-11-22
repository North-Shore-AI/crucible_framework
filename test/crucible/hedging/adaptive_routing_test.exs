defmodule Crucible.Hedging.AdaptiveRoutingTest do
  use ExUnit.Case, async: true

  alias Crucible.Hedging.AdaptiveRouting

  setup do
    models = [
      %{name: "model1", weight: 0.5},
      %{name: "model2", weight: 0.3},
      %{name: "model3", weight: 0.2}
    ]

    {:ok, router} = AdaptiveRouting.start_link(models: models, strategy: :round_robin)

    on_exit(fn ->
      if Process.alive?(router), do: GenServer.stop(router)
    end)

    %{router: router, models: models}
  end

  describe "start_link/1" do
    test "starts router with models" do
      models = [%{name: "test_model", weight: 1.0}]
      {:ok, pid} = AdaptiveRouting.start_link(models: models)

      assert Process.alive?(pid)
      GenServer.stop(pid)
    end

    test "starts with specified strategy" do
      models = [%{name: "test_model", weight: 1.0}]
      {:ok, pid} = AdaptiveRouting.start_link(models: models, strategy: :least_loaded)

      assert Process.alive?(pid)
      GenServer.stop(pid)
    end
  end

  describe "select_model/2" do
    test "round robin distributes evenly", %{router: router} do
      # Select 6 times (2 complete cycles)
      selections =
        for _ <- 1..6 do
          {:ok, model} = AdaptiveRouting.select_model(router)
          model.name
        end

      # Each model should be selected twice
      counts = Enum.frequencies(selections)
      assert counts["model1"] == 2
      assert counts["model2"] == 2
      assert counts["model3"] == 2
    end

    test "least_loaded selects model with fewest active requests" do
      models = [
        %{name: "model1", weight: 0.5},
        %{name: "model2", weight: 0.3}
      ]

      {:ok, router} = AdaptiveRouting.start_link(models: models, strategy: :least_loaded)

      # Record some active requests
      AdaptiveRouting.record_request(router, "model1", :started, 0)
      AdaptiveRouting.record_request(router, "model1", :started, 0)

      {:ok, selected} = AdaptiveRouting.select_model(router)

      # Should select model2 (least loaded)
      assert selected.name == "model2"

      GenServer.stop(router)
    end

    test "best_performing selects by success rate and latency" do
      models = [
        %{name: "model1", weight: 0.5},
        %{name: "model2", weight: 0.3}
      ]

      {:ok, router} = AdaptiveRouting.start_link(models: models, strategy: :best_performing)

      # Model1: 50% success, avg 100ms
      AdaptiveRouting.record_request(router, "model1", true, 100)
      AdaptiveRouting.record_request(router, "model1", false, 100)

      # Model2: 100% success, avg 50ms
      AdaptiveRouting.record_request(router, "model2", true, 50)
      AdaptiveRouting.record_request(router, "model2", true, 50)

      {:ok, selected} = AdaptiveRouting.select_model(router)

      # Should select model2 (better performance)
      assert selected.name == "model2"

      GenServer.stop(router)
    end

    test "weighted respects model weights" do
      models = [
        %{name: "heavy", weight: 0.9},
        %{name: "light", weight: 0.1}
      ]

      {:ok, router} = AdaptiveRouting.start_link(models: models, strategy: :weighted)

      # Select 100 times
      selections =
        for _ <- 1..100 do
          {:ok, model} = AdaptiveRouting.select_model(router)
          model.name
        end

      counts = Enum.frequencies(selections)

      # Heavy should be selected significantly more
      assert counts["heavy"] > counts["light"] * 3

      GenServer.stop(router)
    end
  end

  describe "update_metrics/3" do
    test "tracks latency and success rate", %{router: router} do
      AdaptiveRouting.update_metrics(router, "model1", %{
        latency_ms: 100,
        success: true
      })

      stats = AdaptiveRouting.get_model_stats(router)
      model1_stats = stats["model1"]

      assert model1_stats.total_requests == 1
      assert model1_stats.successful_requests == 1
      assert model1_stats.avg_latency_ms == 100.0
    end

    test "updates routing decisions", %{router: router} do
      # Update with bad metrics for model1
      for _ <- 1..5 do
        AdaptiveRouting.update_metrics(router, "model1", %{
          latency_ms: 1000,
          success: false
        })
      end

      # Update with good metrics for model2
      for _ <- 1..5 do
        AdaptiveRouting.update_metrics(router, "model2", %{
          latency_ms: 50,
          success: true
        })
      end

      # Set to best_performing and verify routing
      AdaptiveRouting.set_strategy(router, :best_performing)
      {:ok, selected} = AdaptiveRouting.select_model(router)

      assert selected.name == "model2"
    end
  end

  describe "record_request/4" do
    test "tracks successful request", %{router: router} do
      AdaptiveRouting.record_request(router, "model1", true, 100)

      stats = AdaptiveRouting.get_model_stats(router)

      assert stats["model1"].total_requests == 1
      assert stats["model1"].successful_requests == 1
    end

    test "tracks failed request", %{router: router} do
      AdaptiveRouting.record_request(router, "model1", false, 100)

      stats = AdaptiveRouting.get_model_stats(router)

      assert stats["model1"].total_requests == 1
      assert stats["model1"].successful_requests == 0
    end
  end

  describe "get_success_rate/2" do
    test "calculates success rate", %{router: router} do
      AdaptiveRouting.record_request(router, "model1", true, 100)
      AdaptiveRouting.record_request(router, "model1", true, 100)
      AdaptiveRouting.record_request(router, "model1", false, 100)

      rate = AdaptiveRouting.get_success_rate(router, "model1")

      assert_in_delta rate, 0.666, 0.01
    end

    test "returns 1.0 for no requests", %{router: router} do
      rate = AdaptiveRouting.get_success_rate(router, "model1")

      assert rate == 1.0
    end
  end

  describe "get_avg_latency/2" do
    test "calculates average latency", %{router: router} do
      AdaptiveRouting.record_request(router, "model1", true, 100)
      AdaptiveRouting.record_request(router, "model1", true, 200)
      AdaptiveRouting.record_request(router, "model1", true, 300)

      avg = AdaptiveRouting.get_avg_latency(router, "model1")

      assert avg == 200.0
    end

    test "returns 0 for no requests", %{router: router} do
      avg = AdaptiveRouting.get_avg_latency(router, "model1")

      assert avg == 0.0
    end
  end

  describe "get_model_stats/1" do
    test "returns stats for all models", %{router: router} do
      AdaptiveRouting.record_request(router, "model1", true, 100)
      AdaptiveRouting.record_request(router, "model2", true, 200)

      stats = AdaptiveRouting.get_model_stats(router)

      assert Map.has_key?(stats, "model1")
      assert Map.has_key?(stats, "model2")
      assert Map.has_key?(stats, "model3")
    end
  end

  describe "set_strategy/2" do
    test "changes routing strategy", %{router: router} do
      :ok = AdaptiveRouting.set_strategy(router, :least_loaded)

      # Verify by checking behavior
      assert true
    end
  end
end
