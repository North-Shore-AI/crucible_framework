defmodule Crucible.Ensemble.MLVoting do
  @moduledoc """
  ML-specific voting strategies using Tinkex adapters.

  Provides multiple voting strategies (majority, weighted, best confidence, unanimous)
  and execution modes (parallel, sequential, hedged, cascade) for ensemble inference.

  ## Examples

      # Run parallel inference with weighted voting
      {:ok, result} = MLVoting.infer(pool, "What is 2+2?",
        strategy: :weighted,
        execution: :parallel,
        timeout: 5000
      )

      # Vote on existing responses
      {:ok, result} = MLVoting.vote(responses, :majority, [])
  """

  alias Crucible.Ensemble.AdapterPool

  require Logger

  @type strategy :: :majority | :weighted | :best_confidence | :unanimous | :custom
  @type execution :: :parallel | :sequential | :hedged | :cascade

  # Main API

  @doc """
  Runs ensemble inference with multiple adapters.

  ## Options
    * `:strategy` - Voting strategy (default: `:weighted`)
    * `:execution` - Execution mode (default: `:parallel`)
    * `:timeout` - Timeout in milliseconds (default: 30_000)
    * `:hedging` - Hedging configuration for `:hedged` execution
    * `:confidence_threshold` - Threshold for `:cascade` mode (default: 0.8)
  """
  @spec infer(AdapterPool.t(), String.t(), keyword()) :: {:ok, term()} | {:error, term()}
  def infer(pool, prompt, opts \\ []) do
    strategy = Keyword.get(opts, :strategy, :weighted)
    execution = Keyword.get(opts, :execution, :parallel)
    hedging = Keyword.get(opts, :hedging, nil)

    clients = AdapterPool.all_clients(pool)

    start_time = System.monotonic_time(:millisecond)

    responses =
      case execution do
        :parallel -> parallel_inference(clients, prompt, opts)
        :sequential -> sequential_inference(clients, prompt, opts)
        :hedged -> hedged_inference(clients, prompt, opts, hedging)
        :cascade -> cascade_inference(clients, prompt, opts)
      end

    duration = System.monotonic_time(:millisecond) - start_time

    emit_telemetry(:infer, %{
      execution: execution,
      strategy: strategy,
      client_count: length(clients),
      duration_ms: duration
    })

    vote(responses, strategy, opts)
  end

  @doc """
  Applies voting strategy to responses.

  Responses should be a list of `{adapter_spec, {:ok, result} | {:error, reason}, latency_ms}` tuples.
  """
  @spec vote([tuple()], strategy(), keyword()) :: {:ok, term()} | {:error, term()}
  def vote(responses, strategy, opts) do
    # Filter out errors
    successful =
      Enum.filter(responses, fn
        {_, {:ok, _}, _} -> true
        _ -> false
      end)

    if successful == [] do
      {:error, :all_requests_failed}
    else
      result =
        case strategy do
          :majority ->
            majority_vote(successful)

          :weighted ->
            weighted_vote(successful, & &1.weight)

          :best_confidence ->
            best_confidence_vote(successful)

          :unanimous ->
            unanimous_vote(successful)

          :custom ->
            custom_vote = Keyword.fetch!(opts, :vote_fn)
            custom_vote.(successful)
        end

      emit_telemetry(:vote, %{
        strategy: strategy,
        response_count: length(successful)
      })

      result
    end
  end

  @doc """
  Extracts confidence from a result.

  Handles multiple formats:
  - Map with `:confidence` key
  - Map with `:logprobs` key (computes average probability)
  - Returns 0.5 default otherwise
  """
  @spec extract_confidence(term()) :: float()
  def extract_confidence(result) do
    case result do
      %{confidence: conf} when is_number(conf) ->
        conf

      %{logprobs: logprobs} when is_list(logprobs) ->
        compute_confidence_from_logprobs(logprobs)

      _ ->
        0.5
    end
  end

  # Execution Modes

  defp parallel_inference(clients, prompt, opts) do
    tasks =
      Enum.map(clients, fn {adapter, client} ->
        Task.async(fn ->
          params = get_sampling_params(opts)
          start_time = System.monotonic_time(:millisecond)

          result = do_generate(client, prompt, params)

          latency = System.monotonic_time(:millisecond) - start_time

          {adapter, result, latency}
        end)
      end)

    timeout = Keyword.get(opts, :timeout, 30_000)
    Task.await_many(tasks, timeout)
  end

  defp sequential_inference(clients, prompt, opts) do
    Enum.map(clients, fn {adapter, client} ->
      params = get_sampling_params(opts)
      start_time = System.monotonic_time(:millisecond)

      result = do_generate(client, prompt, params)

      latency = System.monotonic_time(:millisecond) - start_time

      {adapter, result, latency}
    end)
  end

  defp hedged_inference(clients, prompt, opts, hedging_config) do
    # Use Crucible.Hedging if available, otherwise fall back to basic implementation
    config = hedging_config || %{strategy: :percentile_75, delay_ms: 100}

    case Code.ensure_loaded(Crucible.Hedging.ML) do
      {:module, _} ->
        Crucible.Hedging.ML.dispatch(
          clients,
          fn client ->
            params = get_sampling_params(opts)
            do_generate(client, prompt, params)
          end,
          config
        )

      {:error, _} ->
        # Fallback: use basic fixed delay hedging
        basic_hedged_inference(clients, prompt, opts, config)
    end
  end

  defp basic_hedged_inference(clients, prompt, opts, config) do
    delay = Map.get(config, :delay_ms, 100)
    [{primary_adapter, primary_client} | rest] = clients

    primary_task =
      Task.async(fn ->
        params = get_sampling_params(opts)
        start_time = System.monotonic_time(:millisecond)
        result = do_generate(primary_client, prompt, params)
        latency = System.monotonic_time(:millisecond) - start_time
        {primary_adapter, result, latency}
      end)

    backup_task =
      if rest != [] do
        Task.async(fn ->
          Process.sleep(delay)
          {backup_adapter, backup_client} = hd(rest)
          params = get_sampling_params(opts)
          start_time = System.monotonic_time(:millisecond)
          result = do_generate(backup_client, prompt, params)
          latency = System.monotonic_time(:millisecond) - start_time
          {backup_adapter, result, latency}
        end)
      end

    tasks = [primary_task | if(backup_task, do: [backup_task], else: [])]
    timeout = Keyword.get(opts, :timeout, 30_000)

    case Task.yield_many(tasks, timeout) do
      [{_, {:ok, result}} | _] ->
        Enum.each(tasks, &Task.shutdown(&1, :brutal_kill))
        [result]

      _ ->
        []
    end
  end

  defp cascade_inference(clients, prompt, opts) do
    confidence_threshold = Keyword.get(opts, :confidence_threshold, 0.8)

    Enum.reduce_while(clients, [], fn {adapter, client}, acc ->
      params = get_sampling_params(opts)
      start_time = System.monotonic_time(:millisecond)

      result = do_generate(client, prompt, params)

      latency = System.monotonic_time(:millisecond) - start_time

      case result do
        {:ok, output} ->
          confidence = extract_confidence(output)

          if confidence >= confidence_threshold do
            {:halt, [{adapter, {:ok, output}, latency} | acc]}
          else
            {:cont, [{adapter, {:ok, output}, latency} | acc]}
          end

        {:error, _} = error ->
          {:cont, [{adapter, error, latency} | acc]}
      end
    end)
    |> Enum.reverse()
  end

  # Voting Strategies

  defp majority_vote(responses) do
    responses
    |> Enum.map(fn {_adapter, {:ok, result}, _latency} -> normalize_result(result) end)
    |> Enum.frequencies()
    |> Enum.max_by(fn {_result, count} -> count end)
    |> elem(0)
    |> then(&{:ok, &1})
  end

  defp weighted_vote(responses, weight_fn) do
    responses
    |> Enum.map(fn {adapter, {:ok, result}, _latency} ->
      {normalize_result(result), weight_fn.(adapter)}
    end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {result, weights} -> {result, Enum.sum(weights)} end)
    |> Enum.max_by(&elem(&1, 1))
    |> elem(0)
    |> then(&{:ok, &1})
  end

  defp best_confidence_vote(responses) do
    responses
    |> Enum.map(fn {adapter, {:ok, result}, latency} ->
      {adapter, result, extract_confidence(result), latency}
    end)
    |> Enum.max_by(&elem(&1, 2))
    |> elem(1)
    |> then(&{:ok, &1})
  end

  defp unanimous_vote(responses) do
    results = Enum.map(responses, fn {_, {:ok, result}, _} -> normalize_result(result) end)
    first = hd(results)

    if Enum.all?(results, &(&1 == first)) do
      # Return the original (non-normalized) result
      {_, {:ok, original}, _} = hd(responses)
      {:ok, original}
    else
      {:error, :no_consensus}
    end
  end

  # Helper Functions

  defp normalize_result(result) when is_binary(result), do: result

  defp normalize_result(%{text: text}), do: text

  defp normalize_result(result), do: result

  defp compute_confidence_from_logprobs(logprobs) when is_list(logprobs) and logprobs != [] do
    logprobs
    |> Enum.map(&:math.exp/1)
    |> Enum.sum()
    |> Kernel./(length(logprobs))
  end

  defp compute_confidence_from_logprobs(_), do: 0.5

  defp get_sampling_params(opts) do
    case Code.ensure_loaded(Crucible.Tinkex) do
      {:module, _} ->
        Crucible.Tinkex.sampling_params(opts)

      {:error, _} ->
        # Fallback defaults
        %{
          temperature: Keyword.get(opts, :temperature, 0.7),
          max_tokens: Keyword.get(opts, :max_tokens, 512),
          top_p: Keyword.get(opts, :top_p, 0.9)
        }
    end
  end

  defp do_generate(client, prompt, params) do
    # Call the sampling client
    try do
      case Code.ensure_loaded(Tinkex.SamplingClient) do
        {:module, _} ->
          result = apply(Tinkex.SamplingClient, :sample, [client, prompt, params, []])

          case result do
            {:ok, task} ->
              case Task.await(task, 30_000) do
                {:ok, response} -> {:ok, response}
                {:error, _} = error -> error
              end

            {:error, _} = error ->
              error

            other ->
              {:error, {:unexpected_result, other}}
          end

        {:error, _} ->
          # Tinkex not available, return mock response
          {:ok, %{text: "mock response", confidence: 0.8}}
      end
    rescue
      e ->
        Logger.error("Generation failed: #{Exception.message(e)}")
        {:error, Exception.message(e)}
    end
  end

  defp emit_telemetry(event, metadata) do
    :telemetry.execute(
      [:crucible, :ensemble, event],
      %{timestamp: System.system_time(:millisecond)},
      metadata
    )
  end
end
