defmodule Crucible.Lora.LossRegistryTest do
  use ExUnit.Case, async: true

  alias Crucible.Lora.LossRegistry

  setup do
    {:ok, registry} = LossRegistry.start_link()
    %{registry: registry}
  end

  describe "start_link/1" do
    test "starts loss registry GenServer" do
      {:ok, pid} = LossRegistry.start_link()
      assert is_pid(pid)
      GenServer.stop(pid)
    end

    test "initializes with builtin losses" do
      {:ok, registry} = LossRegistry.start_link()

      losses = LossRegistry.list(registry)
      assert :cross_entropy in losses
      assert :mse in losses

      GenServer.stop(registry)
    end
  end

  describe "register/3" do
    test "registers custom loss function", %{registry: registry} do
      custom_fn = fn outputs, targets ->
        %{loss: abs(outputs - targets), components: %{}}
      end

      :ok = LossRegistry.register(registry, :custom_loss, custom_fn)

      {:ok, retrieved} = LossRegistry.get(registry, :custom_loss)
      assert is_function(retrieved)
    end

    test "registers with options", %{registry: registry} do
      custom_fn = fn _, _ -> %{loss: 0.0} end

      :ok = LossRegistry.register(registry, :weighted_loss, custom_fn, weight: 0.5)

      {:ok, _} = LossRegistry.get(registry, :weighted_loss)
    end

    test "overwrites existing registration", %{registry: registry} do
      fn1 = fn _, _ -> %{loss: 1.0} end
      fn2 = fn _, _ -> %{loss: 2.0} end

      :ok = LossRegistry.register(registry, :test_loss, fn1)
      :ok = LossRegistry.register(registry, :test_loss, fn2)

      {:ok, retrieved} = LossRegistry.get(registry, :test_loss)
      result = retrieved.(nil, nil)
      assert result.loss == 2.0
    end
  end

  describe "get/2" do
    test "retrieves registered loss", %{registry: registry} do
      custom_fn = fn _, _ -> %{loss: 0.5} end
      :ok = LossRegistry.register(registry, :my_loss, custom_fn)

      {:ok, fn_ref} = LossRegistry.get(registry, :my_loss)
      assert is_function(fn_ref)
    end

    test "returns builtin loss functions", %{registry: registry} do
      {:ok, ce_fn} = LossRegistry.get(registry, :cross_entropy)
      assert is_function(ce_fn)
    end

    test "returns error for unregistered loss", %{registry: registry} do
      assert {:error, :not_found} = LossRegistry.get(registry, :nonexistent)
    end
  end

  describe "list/1" do
    test "lists all available losses", %{registry: registry} do
      losses = LossRegistry.list(registry)

      assert is_list(losses)
      assert :cross_entropy in losses
      assert :mse in losses
    end

    test "includes registered custom losses", %{registry: registry} do
      :ok = LossRegistry.register(registry, :custom, fn _, _ -> %{loss: 0.0} end)

      losses = LossRegistry.list(registry)
      assert :custom in losses
    end
  end

  describe "cross_entropy_loss/2" do
    test "computes cross entropy loss" do
      outputs = [0.9, 0.1, 0.8]
      targets = [1, 0, 1]

      result = LossRegistry.cross_entropy_loss(outputs, targets)

      assert is_map(result)
      assert Map.has_key?(result, :loss)
      assert is_number(result.loss)
      assert result.loss >= 0
    end
  end

  describe "mse_loss/2" do
    test "computes mean squared error loss" do
      outputs = [0.9, 0.2, 0.7]
      targets = [1.0, 0.0, 1.0]

      result = LossRegistry.mse_loss(outputs, targets)

      assert is_map(result)
      assert Map.has_key?(result, :loss)
      assert result.loss >= 0
    end
  end

  describe "composite/2" do
    test "combines multiple losses with weights", %{registry: registry} do
      :ok = LossRegistry.register(registry, :loss_a, fn _, _ -> %{loss: 1.0} end)
      :ok = LossRegistry.register(registry, :loss_b, fn _, _ -> %{loss: 2.0} end)

      composite =
        LossRegistry.composite(
          registry,
          [:loss_a, :loss_b],
          [0.3, 0.7]
        )

      result = composite.(nil, nil)

      # 1.0 * 0.3 + 2.0 * 0.7 = 1.7
      assert_in_delta result.loss, 1.7, 0.001
      assert Map.has_key?(result, :components)
    end

    test "validates losses and weights have same length", %{registry: registry} do
      :ok = LossRegistry.register(registry, :a, fn _, _ -> %{loss: 0.0} end)

      assert_raise ArgumentError, fn ->
        LossRegistry.composite(registry, [:a], [0.5, 0.5])
      end
    end
  end

  describe "CNS-specific losses" do
    test "register_topological_loss/1", %{registry: registry} do
      :ok = LossRegistry.register_topological_loss(registry)

      {:ok, loss_fn} = LossRegistry.get(registry, :topological)
      result = loss_fn.(nil, nil)

      assert Map.has_key?(result, :loss)
      assert Map.has_key?(result, :betti_0)
    end

    test "register_chirality_loss/1", %{registry: registry} do
      :ok = LossRegistry.register_chirality_loss(registry)

      {:ok, loss_fn} = LossRegistry.get(registry, :chirality)
      result = loss_fn.(nil, nil)

      assert Map.has_key?(result, :loss)
      assert Map.has_key?(result, :balance_score)
    end

    test "register_citation_loss/1", %{registry: registry} do
      :ok = LossRegistry.register_citation_loss(registry)

      {:ok, loss_fn} = LossRegistry.get(registry, :citation_validity)
      result = loss_fn.(nil, nil)

      assert Map.has_key?(result, :loss)
      assert Map.has_key?(result, :precision)
    end
  end
end
