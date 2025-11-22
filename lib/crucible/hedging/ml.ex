defmodule Crucible.Hedging.ML do
  @moduledoc """
  High-level ML hedging integration for ensemble inference.

  Provides a unified interface for hedging strategies and adaptive routing
  in ML inference scenarios with Tinkex.

  ## Examples

      # Simple dispatch with hedging
      {:ok, result, latency} = Crucible.Hedging.ML.dispatch(
        clients,
        fn client -> Tinkex.SamplingClient.generate(client, prompt, params) end,
        %{strategy: :percentile_75}
      )

      # Create hedger for reuse
      hedger = Crucible.Hedging.ML.create_hedger(:adaptive, history_window: 200)

      # Create router for model selection
      {:ok, router} = Crucible.Hedging.ML.create_router(models, strategy: :best_performing)
  """

  alias Crucible.Hedging.{InferenceHedger, AdaptiveRouting}

  require Logger

  @doc """
  Dispatches inference with hedging strategy.

  ## Config Options
    * `:strategy` - Hedging strategy (`:fixed`, `:percentile_75`, `:percentile_90`, `:adaptive`)
    * `:delay_ms` - Delay in ms for fixed strategy
    * `:window_size` - History window for adaptive strategy
  """
  @spec dispatch([{map(), term()}], (term() -> {:ok, term()} | {:error, term()}), map()) ::
          {:ok, term(), non_neg_integer()} | {:error, term()}
  def dispatch(clients, inference_fn, config \\ %{}) do
    strategy = Map.get(config, :strategy, :percentile_75)

    hedger =
      case strategy do
        :fixed ->
          InferenceHedger.new(
            strategy: :fixed,
            delay_ms: Map.get(config, :delay_ms, 100)
          )

        :percentile_75 ->
          InferenceHedger.new(
            strategy: :percentile,
            percentile: 75
          )

        :percentile_90 ->
          InferenceHedger.new(
            strategy: :percentile,
            percentile: 90
          )

        :adaptive ->
          InferenceHedger.new(
            strategy: :adaptive,
            history_window: Map.get(config, :window_size, 100)
          )

        other ->
          Logger.warning(
            "Unknown hedging strategy: #{inspect(other)}, falling back to percentile_75"
          )

          InferenceHedger.new(strategy: :percentile, percentile: 75)
      end

    InferenceHedger.dispatch(hedger, clients, inference_fn,
      timeout: Map.get(config, :timeout, 30_000)
    )
  end

  @doc """
  Creates a hedger for a specific strategy.

  ## Strategies
    * `:fixed` - Fixed delay hedging
    * `:percentile` - Percentile-based delay
    * `:adaptive` - Adaptive delay based on history
    * `:workload_aware` - Consider model load

  ## Options
    * `:delay_ms` - Delay for fixed strategy
    * `:percentile` - Percentile for percentile strategy
    * `:history_window` - Window size for adaptive strategy
  """
  @spec create_hedger(atom(), keyword()) :: InferenceHedger.t()
  def create_hedger(strategy, opts \\ []) do
    InferenceHedger.new([{:strategy, strategy} | opts])
  end

  @doc """
  Creates a router with adaptive strategy.

  ## Options
    * `:strategy` - Routing strategy (default: `:round_robin`)
  """
  @spec create_router([map()], keyword()) :: {:ok, pid()} | {:error, term()}
  def create_router(models, opts \\ []) do
    AdaptiveRouting.start_link([{:models, models} | opts])
  end
end
