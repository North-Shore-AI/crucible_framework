# Ensemble ML Integration Design

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

This document describes how to integrate crucible_ensemble with ML models via Tinkex, enabling multi-model voting for critics, synthesis, and inference with hedging strategies.

## Architecture

```
+------------------------------------------------------------------+
|                      Crucible.Ensemble                            |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+    +-------------------+                    |
|  | Voting Strategies |    | Execution Modes   |                    |
|  | - Majority        |    | - Parallel        |                    |
|  | - Weighted        |    | - Sequential      |                    |
|  | - Best Confidence |    | - Hedged          |                    |
|  | - Unanimous       |    | - Cascade         |                    |
|  +-------------------+    +-------------------+                    |
|                                                                    |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                    ML Adapter Layer                               |
+------------------------------------------------------------------+
|  +------------------------+    +------------------------+         |
|  | Crucible.Ensemble.ML   |    | Crucible.Hedging.ML   |         |
|  | - Multi-adapter infer  |    | - Request timing      |         |
|  | - Critic voting        |    | - Backup dispatch     |         |
|  | - Synthesis blend      |    | - Cancel slow         |         |
|  +------------------------+    +------------------------+         |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                         Tinkex                                    |
+------------------------------------------------------------------+
|  SamplingClient[]  - Multiple adapters loaded                     |
+------------------------------------------------------------------+
```

## Module Structure

```
lib/crucible/
  ensemble/
    ml.ex                      # ML-specific ensemble operations
    adapter_pool.ex            # Manage multiple sampling clients
    critic_ensemble.ex         # Ensemble for critic voting
    synthesis_ensemble.ex      # Multi-model synthesis
```

## Adapter Pool Management

```elixir
defmodule Crucible.Ensemble.AdapterPool do
  @moduledoc """
  Manages a pool of Tinkex sampling clients for ensemble inference.
  """

  use GenServer

  defstruct [
    :pool_name,
    :adapters,
    :sampling_clients,
    :config
  ]

  @type adapter_spec :: %{
    name: String.t(),
    checkpoint_path: String.t(),
    weight: float(),
    tags: [atom()]
  }

  @doc """
  Creates an adapter pool from registered models.
  """
  def create(opts) do
    adapters = Keyword.fetch!(opts, :adapters)
    session = Keyword.fetch!(opts, :session)

    # Start sampling clients for each adapter
    clients = Enum.map(adapters, fn adapter ->
      {:ok, client} = start_sampling_client(session, adapter)
      {adapter.name, client}
    end)

    GenServer.start_link(__MODULE__, %{
      adapters: adapters,
      sampling_clients: Map.new(clients),
      config: Keyword.get(opts, :config, %{})
    })
  end

  @doc """
  Gets a specific sampling client by adapter name.
  """
  def get_client(pool, adapter_name) do
    GenServer.call(pool, {:get_client, adapter_name})
  end

  @doc """
  Gets all clients for parallel inference.
  """
  def all_clients(pool) do
    GenServer.call(pool, :all_clients)
  end

  @doc """
  Gets clients matching specific tags.
  """
  def clients_by_tags(pool, tags) do
    GenServer.call(pool, {:clients_by_tags, tags})
  end

  defp start_sampling_client(session, adapter) do
    Tinkex.TrainingClient.create_sampling_client_async(
      session.training_client,
      adapter.checkpoint_path
    )
    |> Task.await(:infinity)
  end

  @impl true
  def handle_call({:get_client, name}, _from, state) do
    {:reply, Map.get(state.sampling_clients, name), state}
  end

  @impl true
  def handle_call(:all_clients, _from, state) do
    clients = Enum.map(state.adapters, fn adapter ->
      {adapter, Map.get(state.sampling_clients, adapter.name)}
    end)
    {:reply, clients, state}
  end

  @impl true
  def handle_call({:clients_by_tags, tags}, _from, state) do
    clients = state.adapters
    |> Enum.filter(fn adapter ->
      Enum.all?(tags, &(&1 in adapter.tags))
    end)
    |> Enum.map(fn adapter ->
      {adapter, Map.get(state.sampling_clients, adapter.name)}
    end)
    {:reply, clients, state}
  end
end
```

## ML-Aware Ensemble Operations

```elixir
defmodule Crucible.Ensemble.ML do
  @moduledoc """
  ML-specific ensemble operations using Tinkex adapters.
  """

  alias Crucible.Ensemble.AdapterPool

  @doc """
  Runs ensemble inference with multiple adapters.
  """
  def infer(pool, prompt, opts \\ []) do
    strategy = Keyword.get(opts, :strategy, :weighted_majority)
    execution = Keyword.get(opts, :execution, :parallel)
    hedging = Keyword.get(opts, :hedging, nil)

    # Get all clients
    clients = AdapterPool.all_clients(pool)

    # Execute based on mode
    responses = case execution do
      :parallel -> parallel_inference(clients, prompt, opts)
      :sequential -> sequential_inference(clients, prompt, opts)
      :hedged -> hedged_inference(clients, prompt, opts, hedging)
      :cascade -> cascade_inference(clients, prompt, opts)
    end

    # Apply voting strategy
    vote(responses, strategy, opts)
  end

  defp parallel_inference(clients, prompt, opts) do
    tasks = Enum.map(clients, fn {adapter, client} ->
      Task.async(fn ->
        params = Crucible.Tinkex.sampling_params(opts)
        start_time = System.monotonic_time(:millisecond)

        result = Tinkex.SamplingClient.generate(client, prompt, params)

        latency = System.monotonic_time(:millisecond) - start_time

        {adapter, result, latency}
      end)
    end)

    # Await all with timeout
    timeout = Keyword.get(opts, :timeout, 30_000)
    Task.await_many(tasks, timeout)
  end

  defp sequential_inference(clients, prompt, opts) do
    Enum.map(clients, fn {adapter, client} ->
      params = Crucible.Tinkex.sampling_params(opts)
      start_time = System.monotonic_time(:millisecond)

      result = Tinkex.SamplingClient.generate(client, prompt, params)

      latency = System.monotonic_time(:millisecond) - start_time

      {adapter, result, latency}
    end)
  end

  defp hedged_inference(clients, prompt, opts, hedging_config) do
    # Use crucible_hedging for tail latency optimization
    Crucible.Hedging.dispatch(clients, fn client ->
      params = Crucible.Tinkex.sampling_params(opts)
      Tinkex.SamplingClient.generate(client, prompt, params)
    end, hedging_config)
  end

  defp cascade_inference(clients, prompt, opts) do
    # Try each model until one succeeds with confidence
    confidence_threshold = Keyword.get(opts, :confidence_threshold, 0.8)

    Enum.reduce_while(clients, [], fn {adapter, client}, acc ->
      params = Crucible.Tinkex.sampling_params(opts)
      {:ok, result} = Tinkex.SamplingClient.generate(client, prompt, params)

      confidence = extract_confidence(result)

      if confidence >= confidence_threshold do
        {:halt, [{adapter, {:ok, result}, 0} | acc]}
      else
        {:cont, [{adapter, {:ok, result}, 0} | acc]}
      end
    end)
  end

  @doc """
  Applies voting strategy to responses.
  """
  def vote(responses, strategy, opts) do
    case strategy do
      :majority ->
        majority_vote(responses)

      :weighted_majority ->
        weighted_vote(responses, &(&1.weight))

      :best_confidence ->
        best_confidence_vote(responses)

      :unanimous ->
        unanimous_vote(responses)

      :custom ->
        custom_vote = Keyword.fetch!(opts, :vote_fn)
        custom_vote.(responses)
    end
  end

  defp majority_vote(responses) do
    responses
    |> Enum.map(fn {_adapter, {:ok, result}, _latency} -> result end)
    |> Enum.frequencies()
    |> Enum.max_by(fn {_result, count} -> count end)
    |> elem(0)
    |> then(&{:ok, &1})
  end

  defp weighted_vote(responses, weight_fn) do
    responses
    |> Enum.map(fn {adapter, {:ok, result}, _latency} ->
      {result, weight_fn.(adapter)}
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
    results = Enum.map(responses, fn {_, {:ok, result}, _} -> result end)

    if Enum.all?(results, &(&1 == hd(results))) do
      {:ok, hd(results)}
    else
      {:error, :no_consensus}
    end
  end

  defp extract_confidence(result) do
    # Extract confidence from model output
    # Could be based on logprobs, explicit confidence scores, etc.
    case result do
      %{confidence: conf} -> conf
      %{logprobs: logprobs} -> compute_confidence_from_logprobs(logprobs)
      _ -> 0.5
    end
  end

  defp compute_confidence_from_logprobs(logprobs) when is_list(logprobs) do
    # Average token probability
    logprobs
    |> Enum.map(&:math.exp/1)
    |> Enum.sum()
    |> Kernel./(length(logprobs))
  end
end
```

## Critic Ensemble for CNS

```elixir
defmodule Crucible.Ensemble.Critics do
  @moduledoc """
  Ensemble-based critic evaluation for CNS.
  """

  alias Crucible.Ensemble.{AdapterPool, ML}

  @type critic_type :: :logic | :grounding | :novelty | :causal | :bias

  @doc """
  Creates a critic ensemble from specialized adapters.
  """
  def create_critic_ensemble(session, critic_type, opts \\ []) do
    # Find adapters tagged for this critic type
    adapters = Crucible.Tinkex.ModelRegistry.find_for_ensemble(
      session.model_registry,
      %{tags: [critic_type], top_n: 3}
    )

    AdapterPool.create(
      adapters: adapters,
      session: session,
      config: critic_config(critic_type)
    )
  end

  @doc """
  Runs critic evaluation with ensemble voting.
  """
  def evaluate(critic_pool, sno, critic_type, opts \\ []) do
    prompt = format_critic_prompt(sno, critic_type)

    # Run ensemble inference
    {:ok, result} = ML.infer(critic_pool, prompt,
      strategy: :weighted_majority,
      execution: :parallel,
      timeout: 10_000
    )

    # Parse critic output
    parse_critic_result(result, critic_type)
  end

  @doc """
  Runs all critics in parallel with ensembles.
  """
  def evaluate_all(critic_pools, sno, opts \\ []) do
    tasks = Enum.map(critic_pools, fn {critic_type, pool} ->
      Task.async(fn ->
        {critic_type, evaluate(pool, sno, critic_type, opts)}
      end)
    end)

    results = Task.await_many(tasks, Keyword.get(opts, :timeout, 30_000))
    aggregate_critic_results(results)
  end

  defp format_critic_prompt(sno, :logic) do
    """
    Analyze the following structured narrative for logical consistency.

    Thesis: #{sno.thesis}
    Antithesis: #{sno.antithesis}
    Synthesis: #{sno.synthesis}
    Evidence: #{format_evidence(sno.evidence)}

    Identify any logical flaws, contradictions, or non-sequiturs.
    Rate the logical consistency from 0.0 to 1.0.
    """
  end

  defp format_critic_prompt(sno, :grounding) do
    """
    Verify that the following claims are properly grounded in evidence.

    Synthesis: #{sno.synthesis}
    Citations: #{format_citations(sno.citations)}
    Evidence Pool: #{format_evidence(sno.evidence)}

    For each citation, verify it exists in the evidence pool and supports the claim.
    Rate grounding from 0.0 to 1.0.
    """
  end

  defp format_critic_prompt(sno, :novelty) do
    """
    Assess the novelty of this synthesis compared to existing knowledge.

    Synthesis: #{sno.synthesis}
    Source Texts: #{format_sources(sno.sources)}

    Rate novelty from 0.0 (pure repetition) to 1.0 (significant new insight).
    """
  end

  defp format_critic_prompt(sno, :causal) do
    """
    Analyze causal relationships in this narrative.

    Synthesis: #{sno.synthesis}
    Causal Claims: #{format_causal_claims(sno)}

    Identify any causal fallacies or unsupported causal claims.
    Rate causal validity from 0.0 to 1.0.
    """
  end

  defp format_critic_prompt(sno, :bias) do
    """
    Detect potential biases in this synthesis.

    Thesis: #{sno.thesis}
    Antithesis: #{sno.antithesis}
    Synthesis: #{sno.synthesis}

    Identify confirmation bias, selection bias, or framing effects.
    Rate balance from 0.0 to 1.0.
    """
  end

  defp parse_critic_result(result, critic_type) do
    # Parse structured output from critic model
    %{
      critic_type: critic_type,
      score: extract_score(result),
      issues: extract_issues(result),
      explanation: extract_explanation(result),
      suggestions: extract_suggestions(result)
    }
  end

  defp aggregate_critic_results(results) do
    %{
      critics: Map.new(results),
      overall_score: compute_overall_score(results),
      critical_issues: find_critical_issues(results),
      passed: all_critics_passed?(results)
    }
  end

  defp critic_config(:logic), do: %{temperature: 0.3, max_tokens: 512}
  defp critic_config(:grounding), do: %{temperature: 0.1, max_tokens: 1024}
  defp critic_config(:novelty), do: %{temperature: 0.5, max_tokens: 256}
  defp critic_config(:causal), do: %{temperature: 0.3, max_tokens: 512}
  defp critic_config(:bias), do: %{temperature: 0.3, max_tokens: 512}

  # Placeholder implementations
  defp format_evidence(evidence), do: inspect(evidence)
  defp format_citations(citations), do: inspect(citations)
  defp format_sources(sources), do: inspect(sources)
  defp format_causal_claims(sno), do: inspect(sno)
  defp extract_score(_result), do: 0.85
  defp extract_issues(_result), do: []
  defp extract_explanation(_result), do: ""
  defp extract_suggestions(_result), do: []
  defp compute_overall_score(_results), do: 0.85
  defp find_critical_issues(_results), do: []
  defp all_critics_passed?(_results), do: true
end
```

## Multi-Model Synthesis

```elixir
defmodule Crucible.Ensemble.Synthesis do
  @moduledoc """
  Ensemble-based synthesis for CNS dialectical generation.
  """

  alias Crucible.Ensemble.{AdapterPool, ML}

  @doc """
  Generates synthesis using ensemble of models with blending.
  """
  def synthesize(pool, thesis, antithesis, evidence, opts \\ []) do
    blend_strategy = Keyword.get(opts, :blend_strategy, :best_of_n)

    prompt = format_synthesis_prompt(thesis, antithesis, evidence, opts)

    case blend_strategy do
      :best_of_n ->
        best_of_n_synthesis(pool, prompt, opts)

      :iterative_refinement ->
        iterative_synthesis(pool, prompt, opts)

      :parallel_then_merge ->
        parallel_merge_synthesis(pool, prompt, opts)
    end
  end

  defp best_of_n_synthesis(pool, prompt, opts) do
    n = Keyword.get(opts, :n, 3)

    # Generate N candidates from ensemble
    candidates = for _ <- 1..n do
      {:ok, result} = ML.infer(pool, prompt,
        strategy: :best_confidence,
        execution: :parallel
      )
      result
    end

    # Score and select best
    scored = Enum.map(candidates, fn candidate ->
      score = score_synthesis(candidate, opts)
      {candidate, score}
    end)

    best = Enum.max_by(scored, &elem(&1, 1))
    {:ok, elem(best, 0)}
  end

  defp iterative_synthesis(pool, prompt, opts) do
    iterations = Keyword.get(opts, :iterations, 3)

    Enum.reduce(1..iterations, {:ok, nil}, fn iteration, {:ok, prev} ->
      refined_prompt = if prev do
        """
        #{prompt}

        Previous synthesis (iteration #{iteration - 1}):
        #{prev}

        Improve this synthesis while maintaining accuracy and balance.
        """
      else
        prompt
      end

      ML.infer(pool, refined_prompt,
        strategy: :weighted_majority,
        execution: :parallel
      )
    end)
  end

  defp parallel_merge_synthesis(pool, prompt, opts) do
    # Get individual model outputs
    clients = AdapterPool.all_clients(pool)

    tasks = Enum.map(clients, fn {adapter, client} ->
      Task.async(fn ->
        params = Crucible.Tinkex.sampling_params(opts)
        {:ok, result} = Tinkex.SamplingClient.generate(client, prompt, params)
        {adapter, result}
      end)
    end)

    results = Task.await_many(tasks, 30_000)

    # Merge outputs
    merge_syntheses(results, opts)
  end

  defp merge_syntheses(results, opts) do
    # Use a merging model to combine syntheses
    syntheses = Enum.map(results, fn {adapter, result} ->
      "Model #{adapter.name}: #{result}"
    end)
    |> Enum.join("\n\n")

    merge_prompt = """
    The following are syntheses from different models. Create a unified synthesis that:
    1. Preserves accurate information from all sources
    2. Resolves contradictions by weighing evidence
    3. Maintains balanced dialectical structure

    #{syntheses}

    Unified synthesis:
    """

    # Use primary model for merging
    {:ok, _merged} = ML.infer(opts[:merge_pool], merge_prompt,
      strategy: :best_confidence,
      execution: :parallel
    )
  end

  defp format_synthesis_prompt(thesis, antithesis, evidence, opts) do
    constraints = Keyword.get(opts, :constraints, [])

    """
    Create a dialectical synthesis from the following:

    THESIS: #{thesis}

    ANTITHESIS: #{antithesis}

    EVIDENCE:
    #{format_evidence_list(evidence)}

    CONSTRAINTS:
    #{format_constraints(constraints)}

    Generate a synthesis that:
    1. Acknowledges valid points from both thesis and antithesis
    2. Cites specific evidence using [E1], [E2], etc.
    3. Provides a higher-level reconciliation
    4. Maintains logical consistency
    """
  end

  defp score_synthesis(candidate, opts) do
    # Score based on multiple criteria
    evidence_coverage = score_evidence_coverage(candidate, opts[:evidence])
    logical_coherence = score_logical_coherence(candidate)
    balance = score_dialectical_balance(candidate)

    evidence_coverage * 0.4 + logical_coherence * 0.4 + balance * 0.2
  end

  defp format_evidence_list(evidence) do
    evidence
    |> Enum.with_index(1)
    |> Enum.map(fn {e, i} -> "[E#{i}] #{e}" end)
    |> Enum.join("\n")
  end

  defp format_constraints(constraints) do
    constraints
    |> Enum.map(&"- #{&1}")
    |> Enum.join("\n")
  end

  defp score_evidence_coverage(_candidate, _evidence), do: 0.9
  defp score_logical_coherence(_candidate), do: 0.85
  defp score_dialectical_balance(_candidate), do: 0.8
end
```

## Hedging Strategies for ML

```elixir
defmodule Crucible.Hedging.ML do
  @moduledoc """
  ML-aware hedging strategies for Tinkex inference.
  """

  @doc """
  Dispatches requests with hedging based on model latency profiles.
  """
  def dispatch(clients, inference_fn, config) do
    strategy = Map.get(config, :strategy, :percentile_75)

    case strategy do
      :fixed ->
        fixed_hedging(clients, inference_fn, config)

      :percentile_75 ->
        percentile_hedging(clients, inference_fn, 75, config)

      :percentile_90 ->
        percentile_hedging(clients, inference_fn, 90, config)

      :adaptive ->
        adaptive_hedging(clients, inference_fn, config)
    end
  end

  defp fixed_hedging(clients, inference_fn, config) do
    delay = Map.get(config, :delay_ms, 100)

    primary = hd(clients)
    backups = tl(clients)

    # Start primary
    primary_task = Task.async(fn ->
      {primary, inference_fn.(elem(primary, 1))}
    end)

    # Start backup after delay
    backup_task = Task.async(fn ->
      Process.sleep(delay)
      backup = hd(backups)
      {backup, inference_fn.(elem(backup, 1))}
    end)

    # Return first to complete
    case Task.yield_many([primary_task, backup_task], 30_000) do
      [{^primary_task, {:ok, result}} | _] ->
        Task.shutdown(backup_task, :brutal_kill)
        [{elem(result, 0), {:ok, elem(result, 1)}, 0}]

      [{_, nil}, {^backup_task, {:ok, result}}] ->
        Task.shutdown(primary_task, :brutal_kill)
        [{elem(result, 0), {:ok, elem(result, 1)}, 0}]
    end
  end

  defp percentile_hedging(clients, inference_fn, percentile, _config) do
    # Get latency percentile from telemetry
    delay = get_latency_percentile(percentile)

    [{adapter, client} | rest] = clients

    # Start primary
    primary_task = Task.async(fn ->
      start = System.monotonic_time(:millisecond)
      result = inference_fn.(client)
      latency = System.monotonic_time(:millisecond) - start
      {adapter, result, latency}
    end)

    # Start backup at percentile
    backup_task = if rest != [] do
      Task.async(fn ->
        Process.sleep(delay)
        {b_adapter, b_client} = hd(rest)
        start = System.monotonic_time(:millisecond)
        result = inference_fn.(b_client)
        latency = System.monotonic_time(:millisecond) - start
        {b_adapter, result, latency}
      end)
    end

    # Await first
    tasks = [primary_task | (if backup_task, do: [backup_task], else: [])]

    case await_first(tasks) do
      {:ok, result} ->
        Enum.each(tasks, &Task.shutdown(&1, :brutal_kill))
        [result]

      {:error, _} = error ->
        error
    end
  end

  defp adaptive_hedging(clients, inference_fn, config) do
    # Adjust delay based on recent performance
    window_size = Map.get(config, :window_size, 100)
    base_percentile = Map.get(config, :base_percentile, 75)

    # Get recent latencies
    recent_latencies = get_recent_latencies(window_size)

    # Compute adaptive delay
    delay = if length(recent_latencies) > 10 do
      sorted = Enum.sort(recent_latencies)
      index = floor(length(sorted) * base_percentile / 100)
      Enum.at(sorted, index)
    else
      100  # Default fallback
    end

    percentile_hedging(clients, inference_fn, delay, config)
  end

  defp get_latency_percentile(percentile) do
    # Query telemetry for historical latency data
    # Placeholder: return fixed value
    case percentile do
      75 -> 500
      90 -> 1000
      95 -> 2000
      _ -> 500
    end
  end

  defp get_recent_latencies(_window_size) do
    # Query telemetry store
    []
  end

  defp await_first(tasks) do
    receive do
      {ref, result} when is_reference(ref) ->
        {:ok, result}
    after
      30_000 -> {:error, :timeout}
    end
  end
end
```

## Complete Usage Example

```elixir
alias Crucible.Ensemble.{AdapterPool, ML, Critics, Synthesis}

# 1. Create adapter pool from trained models
{:ok, pool} = AdapterPool.create(
  adapters: [
    %{name: "scifact-v1", checkpoint_path: "tinker://exp1/ckpt/1000", weight: 0.4},
    %{name: "scifact-v2", checkpoint_path: "tinker://exp2/ckpt/1500", weight: 0.35},
    %{name: "scifact-v3", checkpoint_path: "tinker://exp3/ckpt/800", weight: 0.25}
  ],
  session: training_session
)

# 2. Run ensemble inference with hedging
{:ok, result} = ML.infer(pool, "Claim: X causes Y...",
  strategy: :weighted_majority,
  execution: :hedged,
  hedging: %{strategy: :percentile_75},
  timeout: 5000
)

# 3. Create critic ensembles
logic_critics = Critics.create_critic_ensemble(session, :logic)
grounding_critics = Critics.create_critic_ensemble(session, :grounding)

# 4. Evaluate SNO with critic ensembles
critic_results = Critics.evaluate_all(
  %{logic: logic_critics, grounding: grounding_critics},
  sno
)

# 5. Generate synthesis with ensemble
{:ok, synthesis} = Synthesis.synthesize(pool,
  thesis: "Vaccine X is effective",
  antithesis: "Vaccine X has significant side effects",
  evidence: evidence_list,
  blend_strategy: :best_of_n,
  n: 5
)
```
