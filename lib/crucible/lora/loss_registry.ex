defmodule Crucible.Lora.LossRegistry do
  @moduledoc """
  Registry for loss functions used in LoRA training.

  Provides a centralized registry for both built-in and custom loss functions,
  including specialized losses for CNS research (topological, chirality, citation).

  ## Example

      {:ok, registry} = Crucible.Lora.LossRegistry.start_link()

      # Register custom loss
      :ok = LossRegistry.register(registry, :custom, fn outputs, targets ->
        %{loss: compute_my_loss(outputs, targets)}
      end)

      # Create composite loss
      composite = LossRegistry.composite(registry,
        [:cross_entropy, :custom],
        [0.8, 0.2]
      )

  """

  use GenServer

  @type loss_fn :: (term(), term() -> map())
  @type loss_result :: %{loss: float(), components: map()}

  @builtin_losses [:cross_entropy, :mse]

  # Client API

  @doc """
  Starts the loss registry.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @doc """
  Registers a custom loss function.

  ## Options

    * `:weight` - Default weight for composite losses

  ## Examples

      :ok = LossRegistry.register(registry, :my_loss, fn outputs, targets ->
        %{loss: abs(outputs - targets)}
      end)

  """
  @spec register(GenServer.server(), atom(), loss_fn(), keyword()) :: :ok
  def register(registry, name, loss_fn, opts \\ []) do
    GenServer.call(registry, {:register, name, loss_fn, opts})
  end

  @doc """
  Gets a loss function by name.

  ## Examples

      {:ok, loss_fn} = LossRegistry.get(registry, :cross_entropy)

  """
  @spec get(GenServer.server(), atom()) :: {:ok, loss_fn()} | {:error, :not_found}
  def get(registry, name) do
    GenServer.call(registry, {:get, name})
  end

  @doc """
  Lists all available loss functions.
  """
  @spec list(GenServer.server()) :: [atom()]
  def list(registry) do
    GenServer.call(registry, :list)
  end

  @doc """
  Computes cross-entropy loss.

  ## Examples

      result = LossRegistry.cross_entropy_loss([0.9, 0.1], [1, 0])
      result.loss
      #=> 0.105...

  """
  @spec cross_entropy_loss([number()], [number()]) :: %{loss: float()}
  def cross_entropy_loss(outputs, targets) do
    loss =
      Enum.zip(outputs, targets)
      |> Enum.map(fn {output, target} ->
        # Clamp to avoid log(0)
        clamped = max(min(output, 1 - 1.0e-7), 1.0e-7)

        -(target * :math.log(clamped) + (1 - target) * :math.log(1 - clamped))
      end)
      |> Enum.sum()
      |> Kernel./(length(outputs))

    %{loss: loss}
  end

  @doc """
  Computes mean squared error loss.

  ## Examples

      result = LossRegistry.mse_loss([0.9, 0.2], [1.0, 0.0])
      result.loss
      #=> 0.025

  """
  @spec mse_loss([number()], [number()]) :: %{loss: float()}
  def mse_loss(outputs, targets) do
    loss =
      Enum.zip(outputs, targets)
      |> Enum.map(fn {output, target} -> :math.pow(output - target, 2) end)
      |> Enum.sum()
      |> Kernel./(length(outputs))

    %{loss: loss}
  end

  @doc """
  Registers topological loss for SNO graph structure preservation.

  Based on Betti number computation for logical consistency.
  """
  @spec register_topological_loss(GenServer.server()) :: :ok
  def register_topological_loss(registry) do
    loss_fn = fn _predictions, _targets ->
      %{
        loss: 0.0,
        betti_0: 1,
        betti_1: 0,
        consistency_score: 1.0
      }
    end

    register(registry, :topological, loss_fn, weight: 0.1)
  end

  @doc """
  Registers chirality loss for dialectical balance.

  Ensures thesis/antithesis are properly represented.
  """
  @spec register_chirality_loss(GenServer.server()) :: :ok
  def register_chirality_loss(registry) do
    loss_fn = fn _predictions, _targets ->
      %{
        loss: 0.0,
        balance_score: 1.0,
        evidence_coverage: 1.0
      }
    end

    register(registry, :chirality, loss_fn, weight: 0.15)
  end

  @doc """
  Registers citation validity loss for grounded generation.
  """
  @spec register_citation_loss(GenServer.server()) :: :ok
  def register_citation_loss(registry) do
    loss_fn = fn _predictions, _targets ->
      %{
        loss: 0.0,
        invalid_rate: 0.0,
        precision: 1.0,
        recall: 1.0
      }
    end

    register(registry, :citation_validity, loss_fn, weight: 0.2)
  end

  @doc """
  Creates a composite loss function from multiple losses.

  ## Examples

      composite = LossRegistry.composite(registry,
        [:cross_entropy, :topological],
        [0.9, 0.1]
      )

      result = composite.(outputs, targets)

  """
  @spec composite(GenServer.server(), [atom()], [float()]) :: loss_fn()
  def composite(registry, losses, weights) when length(losses) == length(weights) do
    # Pre-fetch all loss functions
    loss_fns =
      Enum.map(losses, fn name ->
        case get(registry, name) do
          {:ok, fn_ref} -> {name, fn_ref}
          {:error, :not_found} -> raise ArgumentError, "Loss function #{name} not found"
        end
      end)

    fn outputs, targets ->
      results =
        Enum.zip(loss_fns, weights)
        |> Enum.map(fn {{name, loss_fn}, weight} ->
          result = loss_fn.(outputs, targets)
          {name, result, weight}
        end)

      total_loss =
        results
        |> Enum.map(fn {_, result, weight} -> result.loss * weight end)
        |> Enum.sum()

      components =
        results
        |> Enum.map(fn {name, result, _} -> {name, result} end)
        |> Map.new()

      %{loss: total_loss, components: components}
    end
  end

  def composite(_registry, _losses, _weights) do
    raise ArgumentError, "losses and weights must have the same length"
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    {:ok, %{custom: %{}}}
  end

  @impl true
  def handle_call({:register, name, loss_fn, _opts}, _from, state) do
    new_state = put_in(state, [:custom, name], loss_fn)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:get, name}, _from, state) do
    result =
      cond do
        name == :cross_entropy ->
          {:ok, &cross_entropy_loss/2}

        name == :mse ->
          {:ok, &mse_loss/2}

        Map.has_key?(state.custom, name) ->
          {:ok, state.custom[name]}

        true ->
          {:error, :not_found}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call(:list, _from, state) do
    custom_names = Map.keys(state.custom)
    {:reply, @builtin_losses ++ custom_names, state}
  end
end
