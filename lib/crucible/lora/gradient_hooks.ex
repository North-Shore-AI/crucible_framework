defmodule Crucible.Lora.GradientHooks do
  @moduledoc """
  Research-grade gradient analysis hooks for LoRA training.

  Provides hooks for monitoring gradient behavior during training, including
  norm tracking, distribution analysis, flow visualization, and health checks.

  ## Example

      {:ok, hooks} = GradientHooks.start_link()

      # Register predefined hooks
      :ok = GradientHooks.register_hook(hooks, :norm, GradientHooks.gradient_norm_hook())
      :ok = GradientHooks.register_hook(hooks, :health, GradientHooks.gradient_health_hook())

      # Run hooks on step result
      enriched = GradientHooks.run_hooks(hooks, step_result)

  """

  use GenServer

  @type hook :: (map() -> {:ok, map()} | :ok)

  # Client API

  @doc """
  Starts the gradient hooks GenServer.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @doc """
  Registers a gradient analysis hook.

  ## Examples

      :ok = GradientHooks.register_hook(hooks, :custom, fn info ->
        IO.inspect(info.gradients)
        {:ok, info}
      end)

  """
  @spec register_hook(GenServer.server(), atom(), hook()) :: :ok
  def register_hook(hooks_pid, name, callback) do
    GenServer.call(hooks_pid, {:register, name, callback})
  end

  @doc """
  Runs all registered hooks on a step result.

  Hooks are executed in registration order, with each hook receiving
  the result from the previous hook.

  ## Examples

      result = GradientHooks.run_hooks(hooks, %{
        step: 1,
        gradients: %{"layer1" => [0.1, 0.2]}
      })

  """
  @spec run_hooks(GenServer.server(), map()) :: map()
  def run_hooks(hooks_pid, step_result) do
    GenServer.call(hooks_pid, {:run, step_result})
  end

  @doc """
  Creates a hook that computes gradient norms per layer.

  ## Examples

      hook = GradientHooks.gradient_norm_hook()
      {:ok, result} = hook.(%{gradients: %{"layer1" => [1.0, 2.0]}})
      result.gradient_norms["layer1"]
      #=> 2.236...

  """
  @spec gradient_norm_hook() :: hook()
  def gradient_norm_hook do
    fn %{gradients: grads} = info ->
      norms =
        grads
        |> Enum.map(fn {layer, grad} ->
          {layer, compute_norm(grad)}
        end)
        |> Map.new()

      {:ok, Map.put(info, :gradient_norms, norms)}
    end
  end

  @doc """
  Creates a hook that analyzes gradient distributions.

  Computes mean, std, min, max, and sparsity for each layer.
  """
  @spec gradient_distribution_hook() :: hook()
  def gradient_distribution_hook do
    fn %{gradients: grads} = info ->
      stats =
        grads
        |> Enum.map(fn {layer, grad} ->
          {layer,
           %{
             mean: compute_mean(grad),
             std: compute_std(grad),
             min: compute_min(grad),
             max: compute_max(grad),
             sparsity: compute_sparsity(grad)
           }}
        end)
        |> Map.new()

      {:ok, Map.put(info, :gradient_stats, stats)}
    end
  end

  @doc """
  Creates a hook that tracks gradient flow through layers.

  Useful for visualizing gradient magnitude across the network depth.
  """
  @spec gradient_flow_hook() :: hook()
  def gradient_flow_hook do
    fn %{gradients: grads} = info ->
      flow =
        grads
        |> Enum.sort_by(fn {layer, _} -> layer_order(layer) end)
        |> Enum.map(fn {layer, grad} ->
          %{layer: layer, magnitude: compute_norm(grad)}
        end)

      {:ok, Map.put(info, :gradient_flow, flow)}
    end
  end

  @doc """
  Creates a hook that detects vanishing/exploding gradients.

  ## Options

    * `:vanishing` - Threshold below which gradients are vanishing (default: 1.0e-7)
    * `:exploding` - Threshold above which gradients are exploding (default: 1.0e3)

  ## Examples

      hook = GradientHooks.gradient_health_hook(%{vanishing: 1.0e-5, exploding: 100.0})

  """
  @spec gradient_health_hook(map()) :: hook()
  def gradient_health_hook(thresholds \\ %{}) do
    vanishing = Map.get(thresholds, :vanishing, 1.0e-7)
    exploding = Map.get(thresholds, :exploding, 1.0e3)

    fn %{gradients: grads} = info ->
      health =
        grads
        |> Enum.map(fn {layer, grad} ->
          norm = compute_norm(grad)

          status =
            cond do
              norm < vanishing -> :vanishing
              norm > exploding -> :exploding
              true -> :healthy
            end

          {layer, status}
        end)
        |> Map.new()

      issues =
        health
        |> Enum.filter(fn {_, status} -> status != :healthy end)
        |> Enum.map(fn {layer, status} -> {layer, status} end)

      {:ok,
       Map.merge(info, %{
         gradient_health: health,
         gradient_issues: issues
       })}
    end
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    {:ok, %{hooks: []}}
  end

  @impl true
  def handle_call({:register, name, callback}, _from, state) do
    hooks = state.hooks ++ [{name, callback}]
    {:reply, :ok, %{state | hooks: hooks}}
  end

  @impl true
  def handle_call({:run, step_result}, _from, state) do
    result =
      Enum.reduce(state.hooks, step_result, fn {_name, callback}, acc ->
        case callback.(acc) do
          {:ok, new_acc} -> new_acc
          :ok -> acc
        end
      end)

    {:reply, result, state}
  end

  # Private helper functions

  defp compute_norm(grad) when is_list(grad) do
    grad
    |> Enum.map(&(&1 * &1))
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp compute_norm(_), do: 0.0

  defp compute_mean(grad) when is_list(grad) and length(grad) > 0 do
    Enum.sum(grad) / length(grad)
  end

  defp compute_mean(_), do: 0.0

  defp compute_std(grad) when is_list(grad) and length(grad) > 1 do
    mean = compute_mean(grad)

    variance =
      grad
      |> Enum.map(&:math.pow(&1 - mean, 2))
      |> Enum.sum()
      |> Kernel./(length(grad) - 1)

    :math.sqrt(variance)
  end

  defp compute_std(_), do: 0.0

  defp compute_min(grad) when is_list(grad) and length(grad) > 0 do
    Enum.min(grad)
  end

  defp compute_min(_), do: 0.0

  defp compute_max(grad) when is_list(grad) and length(grad) > 0 do
    Enum.max(grad)
  end

  defp compute_max(_), do: 0.0

  defp compute_sparsity(grad) when is_list(grad) and length(grad) > 0 do
    zero_count = Enum.count(grad, &(&1 == 0))
    zero_count / length(grad)
  end

  defp compute_sparsity(_), do: 0.0

  defp layer_order(layer) when is_binary(layer) do
    # Extract layer number from name like "layer_1", "layer_2", etc.
    case Regex.run(~r/(\d+)/, layer) do
      [_, num] -> String.to_integer(num)
      _ -> 0
    end
  end

  defp layer_order(_), do: 0
end
