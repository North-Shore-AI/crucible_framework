defmodule Crucible.Lora.GradientHooksTest do
  use ExUnit.Case, async: true

  alias Crucible.Lora.GradientHooks

  setup do
    {:ok, hooks} = GradientHooks.start_link()
    %{hooks: hooks}
  end

  describe "start_link/1" do
    test "starts gradient hooks GenServer" do
      {:ok, pid} = GradientHooks.start_link()
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "register_hook/3" do
    test "registers a named hook", %{hooks: hooks} do
      callback = fn info -> {:ok, info} end

      :ok = GradientHooks.register_hook(hooks, :test_hook, callback)
    end

    test "multiple hooks can be registered", %{hooks: hooks} do
      :ok = GradientHooks.register_hook(hooks, :hook1, fn info -> {:ok, info} end)
      :ok = GradientHooks.register_hook(hooks, :hook2, fn info -> {:ok, info} end)
    end
  end

  describe "run_hooks/2" do
    test "executes all registered hooks", %{hooks: hooks} do
      test_pid = self()

      :ok =
        GradientHooks.register_hook(hooks, :test, fn info ->
          send(test_pid, {:hook_called, info})
          {:ok, info}
        end)

      step_result = %{
        step: 1,
        loss: 0.5,
        gradients: %{"layer1" => [1.0, 2.0]}
      }

      GradientHooks.run_hooks(hooks, step_result)

      assert_receive {:hook_called, ^step_result}, 1000
    end

    test "hooks can modify step result", %{hooks: hooks} do
      :ok =
        GradientHooks.register_hook(hooks, :modifier, fn info ->
          {:ok, Map.put(info, :modified, true)}
        end)

      result = GradientHooks.run_hooks(hooks, %{step: 1})

      assert result.modified == true
    end

    test "hooks run in registration order", %{hooks: hooks} do
      :ok =
        GradientHooks.register_hook(hooks, :first, fn info ->
          {:ok, Map.update(info, :order, [1], &[1 | &1])}
        end)

      :ok =
        GradientHooks.register_hook(hooks, :second, fn info ->
          {:ok, Map.update(info, :order, [2], &[2 | &1])}
        end)

      result = GradientHooks.run_hooks(hooks, %{})

      assert result.order == [2, 1]
    end
  end

  describe "gradient_norm_hook/0" do
    test "computes gradient norms" do
      hook = GradientHooks.gradient_norm_hook()

      info = %{
        gradients: %{
          "layer1" => [1.0, 2.0, 3.0],
          "layer2" => [4.0, 5.0]
        }
      }

      {:ok, result} = hook.(info)

      assert Map.has_key?(result, :gradient_norms)
      assert is_map(result.gradient_norms)
    end
  end

  describe "gradient_distribution_hook/0" do
    test "computes gradient statistics" do
      hook = GradientHooks.gradient_distribution_hook()

      info = %{
        gradients: %{
          "layer1" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
      }

      {:ok, result} = hook.(info)

      assert Map.has_key?(result, :gradient_stats)
      stats = result.gradient_stats["layer1"]

      assert Map.has_key?(stats, :mean)
      assert Map.has_key?(stats, :std)
      assert Map.has_key?(stats, :min)
      assert Map.has_key?(stats, :max)
      assert Map.has_key?(stats, :sparsity)
    end
  end

  describe "gradient_flow_hook/0" do
    test "tracks gradient flow through layers" do
      hook = GradientHooks.gradient_flow_hook()

      info = %{
        gradients: %{
          "layer1" => [1.0],
          "layer2" => [2.0],
          "layer3" => [3.0]
        }
      }

      {:ok, result} = hook.(info)

      assert Map.has_key?(result, :gradient_flow)
      assert is_list(result.gradient_flow)
    end
  end

  describe "gradient_health_hook/1" do
    test "detects healthy gradients" do
      hook = GradientHooks.gradient_health_hook()

      info = %{
        gradients: %{
          "layer1" => [0.1, 0.2, 0.3]
        }
      }

      {:ok, result} = hook.(info)

      assert result.gradient_health["layer1"] == :healthy
      assert result.gradient_issues == []
    end

    test "detects vanishing gradients" do
      hook = GradientHooks.gradient_health_hook(%{vanishing: 1.0e-5})

      info = %{
        gradients: %{
          "layer1" => [1.0e-8, 1.0e-9]
        }
      }

      {:ok, result} = hook.(info)

      assert result.gradient_health["layer1"] == :vanishing
      assert length(result.gradient_issues) > 0
    end

    test "detects exploding gradients" do
      hook = GradientHooks.gradient_health_hook(%{exploding: 100.0})

      info = %{
        gradients: %{
          "layer1" => [1000.0, 2000.0]
        }
      }

      {:ok, result} = hook.(info)

      assert result.gradient_health["layer1"] == :exploding
      assert length(result.gradient_issues) > 0
    end

    test "uses custom thresholds" do
      hook =
        GradientHooks.gradient_health_hook(%{
          vanishing: 0.001,
          exploding: 10.0
        })

      # Value that would be healthy with defaults but vanishing with custom
      info = %{
        gradients: %{
          "layer1" => [0.0001]
        }
      }

      {:ok, result} = hook.(info)
      assert result.gradient_health["layer1"] == :vanishing
    end
  end

  describe "integration" do
    test "multiple predefined hooks work together", %{hooks: hooks} do
      :ok = GradientHooks.register_hook(hooks, :norm, GradientHooks.gradient_norm_hook())
      :ok = GradientHooks.register_hook(hooks, :dist, GradientHooks.gradient_distribution_hook())
      :ok = GradientHooks.register_hook(hooks, :health, GradientHooks.gradient_health_hook())

      info = %{
        gradients: %{
          "layer1" => [0.1, 0.2, 0.3]
        }
      }

      result = GradientHooks.run_hooks(hooks, info)

      assert Map.has_key?(result, :gradient_norms)
      assert Map.has_key?(result, :gradient_stats)
      assert Map.has_key?(result, :gradient_health)
    end
  end
end
