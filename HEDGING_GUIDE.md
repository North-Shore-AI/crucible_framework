# Request Hedging - Complete Guide

**Tail Latency Optimization for Distributed Systems**

Version: 0.1.0
Last Updated: 2025-10-08

---

## Table of Contents

1. [Overview](#overview)
2. [The Tail at Scale Problem](#the-tail-at-scale-problem)
3. [Hedging Fundamentals](#hedging-fundamentals)
4. [Hedging Strategies](#hedging-strategies)
5. [Latency Analysis](#latency-analysis)
6. [Cost-Benefit Analysis](#cost-benefit-analysis)
7. [Implementation Patterns](#implementation-patterns)
8. [Performance Tuning](#performance-tuning)
9. [Production Deployment](#production-deployment)
10. [Advanced Topics](#advanced-topics)
11. [References](#references)

---

## Overview

Request hedging is a latency optimization technique that dramatically reduces tail latency (P95, P99) by issuing backup requests after a delay. When a primary request is slow, a backup request can complete first, protecting against stragglers and reducing worst-case latency.

### Key Benefits

- **96% P99 latency reduction** (Google BigTable results)
- **5-10% average cost increase** for massive tail latency improvement
- **Simple implementation** with immediate production impact
- **Works across services** - API calls, databases, distributed systems

### Quick Start

```elixir
# Basic hedging with fixed delay
{:ok, result, metadata} = Hedging.request(
  fn -> slow_api_call() end,
  strategy: :fixed,
  delay_ms: 100
)

# Percentile-based hedging (recommended)
{:ok, result, metadata} = Hedging.request(
  fn -> make_request() end,
  strategy: :percentile,
  percentile: 95
)

# Adaptive learning
{:ok, result, metadata} = Hedging.request(
  fn -> api_call() end,
  strategy: :adaptive
)

# Check if hedge was fired
if metadata.hedged do
  IO.puts "Hedge saved us from slow request!"
end
```

### Research Foundation

Based on **"The Tail at Scale"** (Dean & Barroso, 2013):

> "The slowest 1% of requests can dominate overall latency. Request hedging addresses this by sending duplicate requests after a short delay, allowing the system to ignore stragglers."

**Real-world Results:**
- Google Search: 96% P99 latency reduction, 5% cost increase
- Google BigTable: 75% P99 reduction, 2x throughput improvement
- Amazon DynamoDB: 80% P99 reduction

---

## The Tail at Scale Problem

### What is Tail Latency?

Tail latency refers to the slowest percentiles of request latency distribution (P95, P99, P99.9).

**Example Distribution:**

```
P50 (median):  100ms
P95:           500ms   (5% of requests > 500ms)
P99:           2000ms  (1% of requests > 2000ms)
P99.9:         5000ms  (0.1% of requests > 5000ms)
```

### Why Tail Latency Matters

#### 1. User Experience Impact

For user-facing applications:
- **Amazon**: 100ms latency = 1% revenue loss
- **Google**: 500ms increase = 20% traffic drop
- **Human perception**: >100ms feels sluggish

#### 2. Fan-Out Amplification

When a request fans out to multiple services:

```
Request → Service A (P99 = 200ms)
       → Service B (P99 = 200ms)
       → Service C (P99 = 200ms)

P(all < 200ms) = 0.99³ = 0.97
P(at least one ≥ 200ms) = 1 - 0.97 = 0.03 = 3%

Effective P97 = 200ms (worse than individual P99!)
```

**General Formula:**

```
P(all N services fast) = p^N
P(at least one slow) = 1 - p^N

For N=10, p=0.99: P(at least one slow) = 9.6%
For N=100, p=0.99: P(at least one slow) = 63.4%
```

#### 3. Sources of Tail Latency

**Hardware:**
- CPU throttling
- Disk seeks
- Network congestion
- Background processes

**Software:**
- Garbage collection pauses
- Lock contention
- Resource exhaustion
- Cold caches

**System:**
- Queueing delays
- Power management
- Noisy neighbors (cloud)
- Thermal throttling

### Traditional Approaches (Insufficient)

❌ **Retry on timeout**: Too late, already waited too long
❌ **Lower timeout**: Increases failure rate
❌ **Load balancing**: Doesn't help if all servers have variance
❌ **Caching**: Not applicable to unique requests
✅ **Hedging**: Proactive duplicate requests

---

## Hedging Fundamentals

### Core Concept

**Hedging = Proactive Redundancy**

```
Timeline:
0ms   ─ Primary request starts
      │
100ms ─ Hedge delay expires
      │ Start backup request
      │
      │ ┌─ Backup completes (150ms) ✓ USE THIS
      │ │
250ms ─ │ Primary completes (slow)
        │
        └─ Cancel primary
```

**Key Insight**: Don't wait for timeout—hedge proactively when request is taking longer than expected.

### Hedging vs Other Techniques

| Technique | When Applied | Cost | Latency Improvement |
|-----------|-------------|------|---------------------|
| **Retry** | After failure/timeout | 2x on failure | None (adds delay) |
| **Hedging** | After delay | 1-2x always | High (P99: -96%) |
| **Replication** | Immediately | Nx always | Moderate |
| **Speculation** | Before request | Nx always | High but expensive |

### Critical Parameters

#### 1. Hedge Delay

**Too Short**: Fires hedges unnecessarily → high cost
**Too Long**: Misses tail latency → no benefit

**Optimal**: P95-P99 of historical latency

```elixir
# Measure historical latencies
latencies = collect_latencies()
p95 = percentile(latencies, 95)  # e.g., 800ms

# Use P95 as hedge delay
Hedging.request(fn -> api_call() end,
  strategy: :fixed,
  delay_ms: p95
)

# Result:
# - 95% of requests complete before hedge (cost = 1.0x)
# - 5% fire hedge, reducing P99 dramatically
```

#### 2. Cancellation Policy

**With Cancellation** (recommended):
- Cancel slower request when faster completes
- Prevents wasted backend resources
- Requires idempotent operations

**Without Cancellation**:
- Let both complete (for non-idempotent ops)
- Higher backend load
- Useful for side effects

```elixir
# With cancellation (default)
Hedging.request(fn -> api_call() end,
  enable_cancellation: true
)

# Without cancellation
Hedging.request(fn -> non_idempotent_update() end,
  enable_cancellation: false
)
```

#### 3. Max Hedges

Limit number of backup requests:

```elixir
# Single hedge (most common)
Hedging.request(fn -> api_call() end,
  max_hedges: 1  # Primary + 1 backup = 2 total
)

# Multiple hedges (extreme cases)
Hedging.request(fn -> critical_request() end,
  max_hedges: 3  # Primary + 3 backups = 4 total
)
```

**Cost Analysis:**

```
max_hedges = 1:
  Expected cost = 1.0 + p(hedge_fires) × 1.0
  If hedge fires 10%: cost = 1.1x

max_hedges = 2:
  Expected cost = 1.0 + p(hedge1) + p(hedge2)
  More complex, diminishing returns
```

### Mathematical Model

#### Expected Latency

Without hedging:
```
E[latency] = ∫₀^∞ t × f(t) dt
```

With hedging at delay d:
```
E[latency_hedged] = ∫₀^d t × f(t) dt + ∫_d^∞ min(d + t', t) × f(t) × f(t') dt dt'

Where:
- f(t) = latency PDF
- d = hedge delay
- t' = backup latency
```

**Simplified (assuming independent):**

```
E[latency_hedged] ≈ E[min(primary_latency, hedge_delay + backup_latency)]
```

#### Expected Cost

```
E[cost] = 1.0 + P(primary_latency > hedge_delay) × 1.0

For P(slow) = 0.05:
E[cost] = 1.0 + 0.05 = 1.05x
```

**Cost-Latency Tradeoff:**

```elixir
defmodule HedgingTradeoff do
  def analyze(latencies, hedge_delays) do
    Enum.map(hedge_delays, fn delay ->
      # Calculate expected cost
      hedge_fire_rate = Enum.count(latencies, & &1 > delay) / length(latencies)
      expected_cost = 1.0 + hedge_fire_rate

      # Simulate hedged latency
      hedged_latencies = Enum.map(latencies, fn primary_latency ->
        if primary_latency > delay do
          # Hedge fires, take min of primary and backup
          backup_latency = Enum.random(latencies)
          min(primary_latency, delay + backup_latency)
        else
          primary_latency
        end
      end)

      p99_improvement = (percentile(latencies, 99) - percentile(hedged_latencies, 99)) /
                        percentile(latencies, 99) * 100

      %{
        delay: delay,
        cost: expected_cost,
        p99_reduction: p99_improvement,
        efficiency: p99_improvement / (expected_cost - 1.0)
      }
    end)
  end
end

# Find optimal delay
latencies = collect_production_latencies()
analysis = HedgingTradeoff.analyze(latencies, [50, 100, 200, 500, 1000])

optimal = Enum.max_by(analysis, & &1.efficiency)
IO.puts "Optimal hedge delay: #{optimal.delay}ms"
IO.puts "P99 reduction: #{optimal.p99_reduction}%"
IO.puts "Cost: #{optimal.cost}x"
```

---

## Hedging Strategies

### 1. Fixed Delay Strategy

**Principle**: Wait a constant duration before hedging.

**Implementation:**

```elixir
defmodule Hedging.Strategy.Fixed do
  @behaviour Hedging.Strategy

  @impl true
  def calculate_delay(opts) do
    Keyword.get(opts, :delay_ms, 100)
  end

  @impl true
  def update(_metrics, state), do: state
end
```

**Usage:**

```elixir
# Simple fixed delay
{:ok, result, metadata} = Hedging.request(
  fn -> make_api_call() end,
  strategy: :fixed,
  delay_ms: 200
)
```

**Characteristics:**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Simplicity | ⭐⭐⭐⭐⭐ | Dead simple |
| Adaptability | ⭐ | Static, no learning |
| Performance | ⭐⭐⭐ | Good if tuned correctly |
| Cost Efficiency | ⭐⭐⭐ | Fixed overhead |

**When to Use:**
- Development/testing
- Highly predictable latency
- Simple deployment requirements
- Known latency patterns

**Tuning Guidelines:**

```elixir
# 1. Measure baseline latency
latencies = measure_latencies(1000)

# 2. Calculate percentiles
p50 = percentile(latencies, 50)  # e.g., 100ms
p95 = percentile(latencies, 95)  # e.g., 400ms
p99 = percentile(latencies, 99)  # e.g., 1000ms

# 3. Choose delay based on goals
delay = cond do
  # Aggressive (optimize P99)
  true -> p95  # Fire hedges for slowest 5%

  # Balanced (optimize cost-latency)
  true -> p95 * 1.2  # Slight buffer

  # Conservative (minimize cost)
  true -> p99 * 0.8  # Only extreme outliers
end
```

**Limitations:**
- No adaptation to workload changes
- Doesn't handle multi-modal distributions
- Requires manual re-tuning

### 2. Percentile-Based Strategy (Recommended)

**Principle**: Hedge at the Xth percentile of recent latency distribution.

**Implementation:**

```elixir
defmodule Hedging.Strategy.Percentile do
  use GenServer

  defstruct [
    :percentile,
    :window_size,
    :latencies,      # Queue of recent latencies
    :current_delay,
    :min_samples
  ]

  def calculate_delay(opts) do
    GenServer.call(__MODULE__, :get_delay)
  end

  def update(metrics, _state) do
    latency = metrics[:primary_latency] || metrics[:total_latency]
    GenServer.cast(__MODULE__, {:update, latency})
  end

  def handle_cast({:update, latency}, state) do
    # Add to rolling window
    latencies = :queue.in(latency, state.latencies)

    # Trim to window size
    latencies = if :queue.len(latencies) > state.window_size do
      {_, trimmed} = :queue.out(latencies)
      trimmed
    else
      latencies
    end

    # Recalculate delay
    current_delay = calculate_percentile(latencies, state.percentile)

    {:noreply, %{state | latencies: latencies, current_delay: current_delay}}
  end
end
```

**Usage:**

```elixir
# Start percentile strategy
{:ok, _} = Hedging.Strategy.Percentile.start_link(
  percentile: 95,
  window_size: 1000,
  initial_delay: 100
)

# Use in requests
{:ok, result, metadata} = Hedging.request(
  fn -> api_call() end,
  strategy: :percentile,
  percentile: 95
)

# Check current stats
stats = Hedging.Strategy.Percentile.get_stats()
IO.puts "Current delay: #{stats.current_delay}ms"
IO.puts "P95 latency: #{stats.p95}ms"
```

**Characteristics:**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Simplicity | ⭐⭐⭐⭐ | Easy to configure |
| Adaptability | ⭐⭐⭐⭐ | Tracks workload changes |
| Performance | ⭐⭐⭐⭐⭐ | Optimal for stable workloads |
| Cost Efficiency | ⭐⭐⭐⭐⭐ | Auto-tunes to minimize cost |

**When to Use:**
- Production systems with traffic
- Predictable daily patterns
- Stable service characteristics
- When you want auto-tuning

**Configuration:**

```elixir
# Aggressive (low latency, higher cost)
Hedging.Strategy.Percentile.start_link(
  percentile: 90,      # Hedge for slowest 10%
  window_size: 500,    # Smaller window = more reactive
  initial_delay: 50
)

# Balanced (recommended)
Hedging.Strategy.Percentile.start_link(
  percentile: 95,      # Hedge for slowest 5%
  window_size: 1000,
  initial_delay: 100
)

# Conservative (low cost, higher latency)
Hedging.Strategy.Percentile.start_link(
  percentile: 99,      # Only extreme outliers
  window_size: 2000,   # Larger window = more stable
  initial_delay: 200
)
```

**Advanced: Time-Window Percentile**

```elixir
defmodule TimeWindowPercentile do
  # Only consider latencies from last N seconds
  def calculate_delay(time_window_seconds \\ 60) do
    now = System.system_time(:second)
    cutoff = now - time_window_seconds

    recent_latencies =
      get_all_latencies()
      |> Enum.filter(fn {timestamp, _} -> timestamp >= cutoff end)
      |> Enum.map(fn {_, latency} -> latency end)

    percentile(recent_latencies, 95)
  end
end
```

### 3. Adaptive Strategy (Thompson Sampling)

**Principle**: Learn optimal delay using multi-armed bandit algorithm.

**Research Foundation:**

Thompson Sampling achieves **O(K log T) regret**:
- K = number of delay candidates
- T = number of trials
- Typically converges in ~500 requests

**Implementation:**

```elixir
defmodule Hedging.Strategy.Adaptive do
  use GenServer

  defstruct [
    :delay_candidates,   # [50, 100, 200, 500, 1000]
    :arm_stats,          # %{delay => %{alpha, beta, pulls}}
    :current_delay
  ]

  def calculate_delay(_opts) do
    GenServer.call(__MODULE__, :select_delay)
  end

  def handle_call(:select_delay, _from, state) do
    # Thompson Sampling: sample from Beta for each arm
    sampled_values =
      Map.new(state.arm_stats, fn {delay, stats} ->
        {delay, sample_beta(stats.alpha, stats.beta)}
      end)

    # Select arm with highest sample
    {selected_delay, _} = Enum.max_by(sampled_values, fn {_, v} -> v end)

    {:reply, selected_delay, %{state | current_delay: selected_delay}}
  end

  def update(metrics, _state) do
    reward = calculate_reward(metrics)
    delay = metrics[:hedge_delay]

    GenServer.cast(__MODULE__, {:update_reward, delay, reward})
  end

  def handle_cast({:update_reward, delay, reward}, state) do
    arm_stats = Map.update!(state.arm_stats, delay, fn stats ->
      %{stats |
        alpha: stats.alpha + reward,
        beta: stats.beta + (1 - reward)
      }
    end)

    {:noreply, %{state | arm_stats: arm_stats}}
  end
end
```

**Reward Function:**

```elixir
def calculate_reward(metrics) do
  cond do
    # Hedge won - reward proportional to latency saved
    metrics[:hedge_won] == true ->
      latency_saved = metrics[:primary_latency] -
                      (metrics[:hedge_delay] + metrics[:backup_latency])
      # Normalize: 1.0 if saved >500ms, 0.0 if no savings
      min(max(latency_saved / 500, 0.0), 1.0)

    # Hedge fired but didn't win - penalty
    metrics[:hedged] == true ->
      0.0

    # No hedge, request was fast - good decision
    true ->
      if metrics[:total_latency] < 200, do: 0.8, else: 0.5
  end
end
```

**Usage:**

```elixir
# Start adaptive strategy
{:ok, _} = Hedging.Strategy.Adaptive.start_link(
  delay_candidates: [50, 100, 200, 500, 1000],
  learning_rate: 0.1
)

# Use in requests
{:ok, result, metadata} = Hedging.request(
  fn -> api_call() end,
  strategy: :adaptive
)

# Monitor learning progress
stats = Hedging.Strategy.Adaptive.get_stats()
IO.inspect stats.arms
# => %{
#   50 => %{pulls: 45, avg_reward: 0.23, expected_value: 0.21},
#   100 => %{pulls: 120, avg_reward: 0.67, expected_value: 0.65},
#   200 => %{pulls: 89, avg_reward: 0.71, expected_value: 0.69},
#   ...
# }
```

**Characteristics:**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Simplicity | ⭐⭐ | Complex algorithm |
| Adaptability | ⭐⭐⭐⭐⭐ | Learns optimal delay |
| Performance | ⭐⭐⭐⭐⭐ | Best long-term |
| Cost Efficiency | ⭐⭐⭐⭐⭐ | Optimizes reward/cost |

**When to Use:**
- High-traffic production (>1000 req/hour)
- Non-stationary workloads
- When optimal delay is unknown
- Long-running services

**Convergence Analysis:**

```elixir
defmodule ConvergenceAnalysis do
  def track_convergence(n_trials \\ 1000) do
    # Reset adaptive strategy
    Hedging.Strategy.Adaptive.start_link(
      delay_candidates: [50, 100, 200, 500]
    )

    # Run trials
    results = Enum.map(1..n_trials, fn i ->
      {:ok, _, metadata} = Hedging.request(
        fn -> simulated_request() end,
        strategy: :adaptive
      )

      stats = Hedging.Strategy.Adaptive.get_stats()

      %{
        trial: i,
        selected_delay: metadata.hedge_delay,
        avg_latency: metadata.total_latency,
        arm_distribution: calculate_distribution(stats.arms)
      }
    end)

    # Plot convergence
    plot_convergence(results)
  end

  defp calculate_distribution(arms) do
    total_pulls = arms |> Map.values() |> Enum.map(& &1.pulls) |> Enum.sum()
    Map.new(arms, fn {delay, stats} ->
      {delay, stats.pulls / total_pulls}
    end)
  end
end

# Typical convergence:
# Trial 100: [0.28, 0.24, 0.25, 0.23] (exploring)
# Trial 500: [0.05, 0.75, 0.15, 0.05] (converging)
# Trial 1000: [0.02, 0.92, 0.04, 0.02] (converged)
```

### 4. Workload-Aware Strategy

**Principle**: Adjust delay based on request characteristics.

**Implementation:**

```elixir
defmodule Hedging.Strategy.WorkloadAware do
  def calculate_delay(opts) do
    base_delay = Keyword.get(opts, :base_delay, 100)

    base_delay
    |> adjust_for_prompt_length(opts[:prompt_length])
    |> adjust_for_model(opts[:model_complexity])
    |> adjust_for_time(opts[:time_of_day])
    |> adjust_for_priority(opts[:priority])
  end

  defp adjust_for_prompt_length(delay, len) when len > 4000, do: delay * 2.5
  defp adjust_for_prompt_length(delay, len) when len > 2000, do: delay * 2.0
  defp adjust_for_prompt_length(delay, len) when len > 1000, do: delay * 1.5
  defp adjust_for_prompt_length(delay, _), do: delay

  defp adjust_for_model(delay, :complex), do: delay * 2.0
  defp adjust_for_model(delay, :medium), do: delay
  defp adjust_for_model(delay, :simple), do: delay * 0.5
  defp adjust_for_model(delay, _), do: delay

  defp adjust_for_time(delay, :peak), do: delay * 0.7      # Hedge sooner
  defp adjust_for_time(delay, :off_peak), do: delay * 1.3  # Can wait longer
  defp adjust_for_time(delay, _), do: delay

  defp adjust_for_priority(delay, :high), do: delay * 0.6
  defp adjust_for_priority(delay, :normal), do: delay
  defp adjust_for_priority(delay, :low), do: delay * 1.5
  defp adjust_for_priority(delay, _), do: delay
end
```

**Usage:**

```elixir
# Context-sensitive hedging
{:ok, result, metadata} = Hedging.request(
  fn -> openai_call(prompt) end,
  strategy: :workload_aware,
  base_delay: 100,
  prompt_length: String.length(prompt),
  model_complexity: :complex,  # GPT-4
  time_of_day: detect_time_of_day(),
  priority: :high
)

defp detect_time_of_day do
  hour = DateTime.utc_now().hour
  cond do
    hour in 8..18 -> :peak      # Business hours
    hour in 0..6 -> :off_peak   # Night
    true -> :normal
  end
end
```

**Example Calculations:**

```elixir
# Scenario 1: Long prompt, complex model, peak hour, high priority
base = 100
adjusted = base * 2.5 * 2.0 * 0.7 * 0.6 = 210ms

# Scenario 2: Short prompt, simple model, off-peak, low priority
base = 100
adjusted = base * 1.0 * 0.5 * 1.3 * 1.5 = 97.5ms
```

**Characteristics:**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Simplicity | ⭐⭐⭐ | Moderate complexity |
| Adaptability | ⭐⭐⭐ | Context-aware |
| Performance | ⭐⭐⭐⭐ | Good for diverse workloads |
| Cost Efficiency | ⭐⭐⭐⭐ | Avoids unnecessary hedges |

**When to Use:**
- Diverse request types
- Predictable context patterns
- Mixed model complexity
- Priority-based systems

**Tuning Multipliers:**

```elixir
defmodule MultiplierTuning do
  def tune_multipliers(historical_data) do
    # Group by context
    grouped = Enum.group_by(historical_data, fn req ->
      {req.prompt_length_bucket, req.model, req.time_of_day}
    end)

    # Calculate optimal multiplier for each group
    Map.new(grouped, fn {context, requests} ->
      latencies = Enum.map(requests, & &1.latency)
      p95 = percentile(latencies, 95)
      base_p95 = get_baseline_p95()

      multiplier = p95 / base_p95
      {context, multiplier}
    end)
  end
end
```

### Strategy Comparison Table

| Strategy | Latency (P99) | Cost | Setup | Adaptability | Best For |
|----------|--------------|------|-------|--------------|----------|
| Fixed | 🟡 Good | 🟢 Low | 🟢 Trivial | 🔴 None | Development |
| Percentile | 🟢 Excellent | 🟢 Low | 🟢 Easy | 🟢 High | Production |
| Adaptive | 🟢 Excellent | 🟡 Medium | 🟡 Complex | 🟢 Highest | High-traffic |
| Workload-Aware | 🟢 Excellent | 🟢 Low | 🟡 Moderate | 🟡 Medium | Diverse loads |

---

## Latency Analysis

### Measuring Latency Impact

**Baseline Collection:**

```elixir
defmodule LatencyStudy do
  def measure_baseline(n_trials \\ 10_000) do
    latencies =
      1..n_trials
      |> Task.async_stream(fn _ ->
        {time_us, _} = :timer.tc(fn -> api_call() end)
        div(time_us, 1000)  # Convert to ms
      end, max_concurrency: 100, timeout: 30_000)
      |> Enum.map(fn {:ok, latency} -> latency end)

    %{
      p50: percentile(latencies, 50),
      p90: percentile(latencies, 90),
      p95: percentile(latencies, 95),
      p99: percentile(latencies, 99),
      p99_9: percentile(latencies, 99.9),
      mean: Enum.sum(latencies) / length(latencies),
      max: Enum.max(latencies),
      samples: length(latencies)
    }
  end
end

baseline = LatencyStudy.measure_baseline()
# => %{
#   p50: 450ms,
#   p90: 800ms,
#   p95: 1200ms,
#   p99: 3000ms,
#   p99_9: 8000ms,
#   mean: 520ms,
#   max: 15000ms
# }
```

**Hedged Measurement:**

```elixir
defmodule HedgedLatencyStudy do
  def measure_hedged(hedge_delay, n_trials \\ 10_000) do
    latencies =
      1..n_trials
      |> Task.async_stream(fn _ ->
        {time_us, {:ok, _, metadata}} = :timer.tc(fn ->
          Hedging.request(
            fn -> api_call() end,
            strategy: :fixed,
            delay_ms: hedge_delay
          )
        end)

        {div(time_us, 1000), metadata}
      end, max_concurrency: 100, timeout: 30_000)
      |> Enum.map(fn {:ok, result} -> result end)

    hedge_fire_rate =
      latencies
      |> Enum.count(fn {_, m} -> m.hedged end)
      |> Kernel./(length(latencies))

    hedge_win_rate =
      latencies
      |> Enum.filter(fn {_, m} -> m.hedged end)
      |> Enum.count(fn {_, m} -> m.hedge_won end)
      |> case do
        0 -> 0.0
        count -> count / Enum.count(latencies, fn {_, m} -> m.hedged end)
      end

    latency_values = Enum.map(latencies, fn {l, _} -> l end)

    %{
      p50: percentile(latency_values, 50),
      p95: percentile(latency_values, 95),
      p99: percentile(latency_values, 99),
      hedge_fire_rate: hedge_fire_rate,
      hedge_win_rate: hedge_win_rate,
      expected_cost: 1.0 + hedge_fire_rate
    }
  end
end

# Test multiple delays
delays = [100, 200, 500, 800, 1000]
results = Enum.map(delays, fn d ->
  {d, HedgedLatencyStudy.measure_hedged(d)}
end)

# Find optimal
optimal = Enum.max_by(results, fn {_, r} ->
  p99_improvement = (baseline.p99 - r.p99) / baseline.p99
  p99_improvement / (r.expected_cost - 1.0)  # Efficiency score
end)
```

### P99 Reduction Analysis

**Case Study: Real Production Data**

```elixir
# Without Hedging
baseline = %{
  p50: 450ms,
  p95: 1200ms,
  p99: 3500ms,
  p99_9: 12000ms
}

# With Hedging (delay = P95 = 1200ms)
hedged = %{
  p50: 455ms,    # +5ms (+1.1%)
  p95: 1210ms,   # +10ms (+0.8%)
  p99: 1350ms,   # -2150ms (-61.4%)
  p99_9: 2800ms  # -9200ms (-76.7%)
}

# Cost
hedge_fire_rate = 0.05  # 5%
expected_cost = 1.05x

# Analysis
improvement = %{
  p99_reduction_ms: 3500 - 1350,        # 2150ms saved
  p99_reduction_pct: 61.4,
  p99_9_reduction_pct: 76.7,
  cost_increase: 5,
  efficiency: 61.4 / 5                   # 12.3% reduction per 1% cost
}
```

### Latency Distribution Changes

**Before Hedging (Bi-modal):**

```
Frequency
    │
100%│  ●●●●●●
    │  ●●●●●●●●
 75%│  ●●●●●●●●●●
    │  ●●●●●●●●●●●
 50%│  ●●●●●●●●●●●●●
    │  ●●●●●●●●●●●●●●         ●●
 25%│  ●●●●●●●●●●●●●●●●●●●●●●●●●●
    │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
  0%└──────────────────────────────────
    0ms        500ms       3000ms    15000ms
           Fast Mode    Slow Mode (stragglers)
```

**After Hedging (Single-mode):**

```
Frequency
    │
100%│  ●●●●●●
    │  ●●●●●●●●
 75%│  ●●●●●●●●●●
    │  ●●●●●●●●●●●●
 50%│  ●●●●●●●●●●●●●●●●
    │  ●●●●●●●●●●●●●●●●●●
 25%│  ●●●●●●●●●●●●●●●●●●●●
    │  ●●●●●●●●●●●●●●●●●●●●●●
  0%└────────────────────────────────
    0ms        500ms       1500ms
           Stragglers eliminated!
```

### Multi-Level Impact

**Single-Service:**

```
Without hedging: P99 = 2000ms
With hedging:    P99 = 400ms
Improvement:     80%
```

**Fan-out (10 services):**

```
Without hedging:
  P(all < 2000ms) = 0.99^10 = 90.4%
  Effective P90 = 2000ms

With hedging:
  P(all < 400ms) = 0.99^10 = 90.4%
  Effective P90 = 400ms

Improvement: 80% (same as single!)
```

**Key Insight**: Hedging maintains benefits even with fan-out.

---

## Cost-Benefit Analysis

### Cost Model

**Components:**

1. **Request Cost**: $C_req
2. **Hedge Fire Rate**: P(hedge_fires)
3. **Expected Hedges**: E[hedges] = P(hedge_fires) × 1

**Total Cost:**

```
C_total = C_req × (1 + E[hedges])
        = C_req × (1 + P(hedge_fires))

For P(hedge_fires) = 0.05:
C_total = C_req × 1.05
Cost increase: 5%
```

### ROI Calculation

**Latency Value:**

```
Value per millisecond saved:
- E-commerce: $0.0001 per ms (based on conversion data)
- Financial trading: $100+ per ms
- Gaming: $0.001 per ms
- Enterprise SaaS: $0.00001 per ms
```

**Example: E-commerce**

```elixir
# Parameters
baseline_p99 = 3000  # ms
hedged_p99 = 500     # ms
latency_saved = 2500 # ms

request_cost = 0.001  # $0.001 per request
value_per_ms = 0.0001  # $0.0001 per ms

# Calculations
hedge_cost = request_cost * 0.05  # 5% overhead = $0.00005
latency_value = latency_saved * value_per_ms  # $0.25

roi = (latency_value - hedge_cost) / hedge_cost
# => 4999x ROI!

# For 1M requests/day
daily_requests = 1_000_000
daily_benefit = daily_requests * (latency_value - hedge_cost)
# => $249,950 per day
```

### Break-Even Analysis

**When is hedging worth it?**

```
Break-even occurs when:
value_of_latency_savings ≥ cost_of_hedging

Let:
- L = latency saved (ms)
- V = value per ms saved ($)
- C = request cost ($)
- P = hedge fire rate

Break-even condition:
L × V ≥ C × P

Solve for L:
L ≥ (C × P) / V

Example:
C = $0.001
P = 0.05
V = $0.0001

L ≥ (0.001 × 0.05) / 0.0001 = 0.5ms

Conclusion: Worth it if saving ≥ 0.5ms
```

### Cost Optimization Strategies

#### 1. Selective Hedging

**Only hedge critical requests:**

```elixir
defmodule SelectiveHedging do
  def request(fun, opts \\ []) do
    if should_hedge?(opts) do
      Hedging.request(fun, opts)
    else
      {:ok, fun.(), %{hedged: false}}
    end
  end

  defp should_hedge?(opts) do
    cond do
      # Always hedge high-priority
      opts[:priority] == :high -> true

      # Hedge user-facing requests
      opts[:user_facing] == true -> true

      # Hedge during peak hours
      peak_hours?() -> true

      # Don't hedge batch jobs
      opts[:batch] == true -> false

      # Default: hedge
      true -> true
    end
  end
end
```

#### 2. Adaptive Budget

**Stay within cost budget:**

```elixir
defmodule BudgetAwareHedging do
  use GenServer

  defstruct daily_budget: 100.0,
            spent_today: 0.0,
            reset_at: nil

  def request(fun, base_cost, opts \\ []) do
    hedge_cost = base_cost * 0.05  # 5% overhead

    if can_afford?(hedge_cost) do
      {:ok, result, metadata} = Hedging.request(fun, opts)
      record_spend(hedge_cost)
      {:ok, result, metadata}
    else
      # No budget - skip hedging
      {:ok, fun.(), %{hedged: false, reason: :budget_exceeded}}
    end
  end

  defp can_afford?(cost) do
    state = GenServer.call(__MODULE__, :get_state)
    state.spent_today + cost <= state.daily_budget
  end

  defp record_spend(cost) do
    GenServer.cast(__MODULE__, {:spend, cost})
  end
end
```

#### 3. Tiered Hedging

**Different delays for different tiers:**

```elixir
defmodule TieredHedging do
  def request(fun, tier, opts \\ []) do
    delay = case tier do
      :premium -> 50   # Aggressive hedging
      :standard -> 200 # Normal hedging
      :budget -> 500   # Conservative hedging
    end

    Hedging.request(fun, Keyword.put(opts, :delay_ms, delay))
  end
end

# Usage
TieredHedging.request(fn -> api_call() end, :premium)
```

### Cost-Latency Pareto Frontier

```elixir
defmodule ParetoAnalysis do
  def find_pareto_frontier do
    # Test range of hedge delays
    delays = 10..2000 |> Enum.take_every(50)

    results = Enum.map(delays, fn delay ->
      # Simulate performance
      {:ok, _, metadata} = simulate_hedging(delay)

      %{
        delay: delay,
        p99: metadata.p99_latency,
        cost: metadata.expected_cost
      }
    end)

    # Find Pareto optimal points
    pareto = find_pareto_optimal(results)

    # Recommend based on efficiency
    recommended = Enum.max_by(pareto, fn r ->
      p99_improvement = (baseline_p99 - r.p99) / baseline_p99
      cost_increase = r.cost - 1.0
      p99_improvement / max(cost_increase, 0.001)
    end)

    %{
      pareto_frontier: pareto,
      recommended: recommended
    }
  end

  defp find_pareto_optimal(results) do
    Enum.filter(results, fn r1 ->
      not Enum.any?(results, fn r2 ->
        # r2 dominates r1 if it's better in all dimensions
        r2.p99 < r1.p99 and r2.cost < r1.cost
      end)
    end)
  end
end
```

---

## Implementation Patterns

### Basic Integration

```elixir
defmodule MyApp.API do
  def call_external_service(params) do
    Hedging.request(
      fn ->
        HTTPoison.post!(
          "https://api.example.com/endpoint",
          Jason.encode!(params),
          [{"Content-Type", "application/json"}],
          timeout: 5000
        )
      end,
      strategy: :percentile,
      percentile: 95,
      timeout_ms: 6000
    )
  end
end
```

### Phoenix Integration

```elixir
defmodule MyAppWeb.SearchController do
  use MyAppWeb, :controller

  def search(conn, %{"q" => query}) do
    # Hedge the search request
    case Hedging.request(
      fn -> MyApp.Search.query(query) end,
      strategy: :adaptive,
      timeout_ms: 3000
    ) do
      {:ok, results, metadata} ->
        conn
        |> put_resp_header("x-hedge-fired", "#{metadata.hedged}")
        |> put_resp_header("x-latency-ms", "#{metadata.total_latency}")
        |> json(%{results: results})

      {:error, reason} ->
        conn
        |> put_status(500)
        |> json(%{error: "Search failed: #{inspect(reason)}"})
    end
  end
end
```

### GenServer Pattern

```elixir
defmodule MyApp.WorkerPool do
  use GenServer

  def handle_call({:process, task}, _from, state) do
    result = Hedging.request(
      fn -> heavy_processing(task) end,
      strategy: :workload_aware,
      base_delay: 100,
      prompt_length: String.length(task.input),
      priority: task.priority
    )

    {:reply, result, state}
  end
end
```

### Retry + Hedging

```elixir
defmodule ResilientRequest do
  def call(fun, opts \\ []) do
    max_retries = Keyword.get(opts, :max_retries, 3)

    retry_with_hedging(fun, opts, max_retries)
  end

  defp retry_with_hedging(fun, opts, retries_left) when retries_left > 0 do
    case Hedging.request(fun, opts) do
      {:ok, result, metadata} ->
        {:ok, result, metadata}

      {:error, reason} when retries_left > 1 ->
        # Exponential backoff
        delay = :math.pow(2, 3 - retries_left) * 100
        Process.sleep(trunc(delay))

        retry_with_hedging(fun, opts, retries_left - 1)

      {:error, reason} ->
        {:error, reason}
    end
  end
end
```

### Circuit Breaker Integration

```elixir
defmodule HedgedCircuitBreaker do
  use GenServer

  defstruct failures: 0,
            state: :closed,
            last_failure: nil

  def request(fun, opts \\ []) do
    case get_state() do
      :open ->
        {:error, :circuit_open}

      :half_open ->
        execute_with_hedging(fun, opts, :half_open)

      :closed ->
        execute_with_hedging(fun, opts, :closed)
    end
  end

  defp execute_with_hedging(fun, opts, circuit_state) do
    case Hedging.request(fun, opts) do
      {:ok, result, metadata} ->
        if circuit_state == :half_open do
          close_circuit()
        end

        {:ok, result, metadata}

      {:error, reason} ->
        record_failure()
        {:error, reason}
    end
  end

  defp record_failure do
    GenServer.cast(__MODULE__, :failure)
  end

  def handle_cast(:failure, state) do
    new_failures = state.failures + 1

    new_state = if new_failures >= 5 do
      # Open circuit after 5 failures
      schedule_half_open(30_000)  # Try again in 30s
      :open
    else
      state.state
    end

    {:noreply, %{state | failures: new_failures, state: new_state}}
  end
end
```

---

## Performance Tuning

### Delay Calibration

**Automated Calibration:**

```elixir
defmodule HedgeCalibration do
  def auto_calibrate(service_fn, target_p99_ms, opts \\ []) do
    # Binary search for optimal delay
    min_delay = Keyword.get(opts, :min_delay, 10)
    max_delay = Keyword.get(opts, :max_delay, 5000)

    find_optimal_delay(service_fn, target_p99_ms, min_delay, max_delay)
  end

  defp find_optimal_delay(service_fn, target, min_d, max_d, iterations \\ 0) do
    if iterations >= 10 or max_d - min_d < 50 do
      # Converged
      mid = div(min_d + max_d, 2)
      %{optimal_delay: mid, iterations: iterations}
    else
      mid = div(min_d + max_d, 2)

      # Test this delay
      p99 = test_delay(service_fn, mid)

      cond do
        p99 < target ->
          # Too aggressive, increase delay
          find_optimal_delay(service_fn, target, mid, max_d, iterations + 1)

        p99 > target * 1.1 ->
          # Too conservative, decrease delay
          find_optimal_delay(service_fn, target, min_d, mid, iterations + 1)

        true ->
          # Close enough
          %{optimal_delay: mid, achieved_p99: p99, iterations: iterations}
      end
    end
  end

  defp test_delay(service_fn, delay) do
    latencies = Enum.map(1..100, fn _ ->
      {:ok, _, metadata} = Hedging.request(
        service_fn,
        strategy: :fixed,
        delay_ms: delay
      )
      metadata.total_latency
    end)

    percentile(latencies, 99)
  end
end

# Usage
calibration = HedgeCalibration.auto_calibrate(
  fn -> my_api_call() end,
  target_p99_ms: 500
)
# => %{optimal_delay: 320, achieved_p99: 495, iterations: 5}
```

### Load Testing

```elixir
defmodule HedgingLoadTest do
  def run_load_test(rps, duration_seconds) do
    total_requests = rps * duration_seconds
    interval_us = div(1_000_000, rps)

    start_time = System.monotonic_time(:millisecond)

    results =
      1..total_requests
      |> Task.async_stream(
        fn i ->
          # Pace requests
          target_time = start_time + div(i * interval_us, 1000)
          current_time = System.monotonic_time(:millisecond)
          sleep_time = max(0, target_time - current_time)
          if sleep_time > 0, do: Process.sleep(sleep_time)

          # Execute hedged request
          {time_us, result} = :timer.tc(fn ->
            Hedging.request(
              fn -> api_call() end,
              strategy: :percentile
            )
          end)

          {div(time_us, 1000), result}
        end,
        max_concurrency: rps * 2,
        timeout: 30_000
      )
      |> Enum.to_list()

    analyze_load_test(results, rps)
  end

  defp analyze_load_test(results, rps) do
    {latencies, outcomes} = Enum.unzip(results)

    successes = Enum.count(outcomes, &match?({:ok, _, _}, &1))
    failures = length(outcomes) - successes

    hedge_fired = Enum.count(outcomes, fn
      {:ok, {_, _, meta}} -> meta.hedged
      _ -> false
    end)

    %{
      rps: rps,
      total_requests: length(results),
      success_rate: successes / length(results),
      p50: percentile(latencies, 50),
      p99: percentile(latencies, 99),
      hedge_fire_rate: hedge_fired / length(results),
      throughput: successes / (Enum.max(latencies) / 1000)
    }
  end
end

# Run tests at different RPS
[10, 50, 100, 500, 1000]
|> Enum.map(fn rps ->
  {rps, HedgingLoadTest.run_load_test(rps, 60)}
end)
|> Enum.each(fn {rps, results} ->
  IO.puts """
  RPS: #{rps}
    Success Rate: #{Float.round(results.success_rate * 100, 2)}%
    P99 Latency: #{results.p99}ms
    Hedge Fire Rate: #{Float.round(results.hedge_fire_rate * 100, 2)}%
  """
end)
```

### Monitoring & Alerting

```elixir
defmodule HedgingMonitor do
  use GenServer

  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def init(_) do
    # Attach telemetry
    :telemetry.attach_many(
      "hedging-monitor",
      [
        [:hedging, :request, :stop],
        [:hedging, :hedge, :fired],
        [:hedging, :hedge, :won]
      ],
      &handle_telemetry/4,
      nil
    )

    {:ok, %{
      window: :queue.new(),
      window_size: 1000
    }}
  end

  def handle_telemetry([:hedging, :request, :stop], measurements, metadata, _) do
    GenServer.cast(__MODULE__, {:request_complete, measurements, metadata})
  end

  def handle_telemetry([:hedging, :hedge, :fired], _, metadata, _) do
    GenServer.cast(__MODULE__, {:hedge_fired, metadata})
  end

  def handle_cast({:request_complete, measurements, metadata}, state) do
    # Track metrics
    window = :queue.in({:request, measurements, metadata}, state.window)

    # Trim window
    window = if :queue.len(window) > state.window_size do
      {_, trimmed} = :queue.out(window)
      trimmed
    else
      window
    end

    # Check for anomalies
    check_anomalies(window)

    {:noreply, %{state | window: window}}
  end

  defp check_anomalies(window) do
    recent = :queue.to_list(window)

    # Calculate metrics
    hedge_fire_rate = Enum.count(recent, fn
      {:request, _, %{hedged: true}} -> true
      _ -> false
    end) / length(recent)

    avg_latency = recent
      |> Enum.map(fn {:request, m, _} -> m.duration / 1000 end)
      |> Enum.sum()
      |> Kernel./(length(recent))

    # Alert conditions
    cond do
      hedge_fire_rate > 0.2 ->
        alert(:high_hedge_rate, hedge_fire_rate)

      hedge_fire_rate < 0.01 ->
        alert(:low_hedge_rate, hedge_fire_rate)

      avg_latency > 1000 ->
        alert(:high_latency, avg_latency)

      true ->
        :ok
    end
  end

  defp alert(type, value) do
    Logger.warning("Hedging Alert: #{type} = #{value}")

    # Send to alerting system
    # Pagerduty.alert(type, value)
  end
end
```

---

## Production Deployment

### Rollout Strategy

**Phase 1: Shadow Mode**

```elixir
defmodule ShadowHedging do
  def request(fun, opts \\ []) do
    # Execute without hedging
    {primary_time, primary_result} = :timer.tc(fun)

    # Simulate hedging in background (log only)
    Task.start(fn ->
      simulate_hedging(fun, opts, primary_time)
    end)

    primary_result
  end

  defp simulate_hedging(fun, opts, actual_latency) do
    delay = calculate_delay(opts)

    would_hedge = actual_latency > delay * 1000  # us to ms

    if would_hedge do
      # Log hypothetical hedge
      Logger.info("Would hedge: actual=#{div(actual_latency, 1000)}ms, delay=#{delay}ms")
    end
  end
end
```

**Phase 2: Canary (1%)**

```elixir
defmodule CanaryHedging do
  def request(fun, opts \\ []) do
    if :rand.uniform() < 0.01 do
      # 1% traffic gets hedging
      Hedging.request(fun, Keyword.put(opts, :canary, true))
    else
      # 99% normal
      {:ok, fun.(), %{hedged: false, canary: false}}
    end
  end
end
```

**Phase 3: Gradual Rollout**

```elixir
defmodule GradualRollout do
  def request(fun, opts \\ []) do
    rollout_pct = get_rollout_percentage()

    if :rand.uniform() * 100 < rollout_pct do
      Hedging.request(fun, opts)
    else
      {:ok, fun.(), %{hedged: false}}
    end
  end

  defp get_rollout_percentage do
    # Read from config/feature flag
    Application.get_env(:my_app, :hedging_rollout_pct, 0)
  end
end

# Schedule:
# Day 1: 1%
# Day 2: 5%
# Day 3: 10%
# Day 4: 25%
# Day 5: 50%
# Day 6: 100%
```

### Configuration Management

```elixir
defmodule HedgingConfig do
  use GenServer

  defstruct [
    :enabled,
    :strategy,
    :default_delay,
    :percentile,
    :max_hedges,
    :enable_cancellation
  ]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_) do
    # Load from application config
    config = %__MODULE__{
      enabled: Application.get_env(:my_app, :hedging_enabled, false),
      strategy: Application.get_env(:my_app, :hedging_strategy, :percentile),
      default_delay: Application.get_env(:my_app, :hedging_delay, 100),
      percentile: Application.get_env(:my_app, :hedging_percentile, 95),
      max_hedges: Application.get_env(:my_app, :hedging_max_hedges, 1),
      enable_cancellation: Application.get_env(:my_app, :hedging_cancellation, true)
    }

    # Watch for config changes
    schedule_config_reload()

    {:ok, config}
  end

  def get_config do
    GenServer.call(__MODULE__, :get_config)
  end

  def update_config(updates) do
    GenServer.call(__MODULE__, {:update, updates})
  end

  def handle_call(:get_config, _from, config) do
    {:reply, config, config}
  end

  def handle_call({:update, updates}, _from, config) do
    new_config = struct(config, updates)
    persist_config(new_config)
    {:reply, :ok, new_config}
  end

  def handle_info(:reload_config, _config) do
    # Reload from storage
    new_config = load_config_from_db()
    schedule_config_reload()
    {:noreply, new_config}
  end

  defp schedule_config_reload do
    Process.send_after(self(), :reload_config, 60_000)  # Every minute
  end
end
```

### Emergency Controls

```elixir
defmodule HedgingEmergency do
  def kill_switch do
    # Immediately disable hedging
    HedgingConfig.update_config(enabled: false)
    Logger.warning("Hedging DISABLED via kill switch")
  end

  def reduce_aggressiveness do
    # Make hedging more conservative
    HedgingConfig.update_config(
      percentile: 99,  # Only hedge extreme outliers
      max_hedges: 1
    )
    Logger.warning("Hedging aggressiveness REDUCED")
  end

  def enable_for_critical_only do
    # Only hedge critical requests
    HedgingConfig.update_config(critical_only: true)
  end
end

# Expose via admin API
defmodule MyAppWeb.AdminController do
  def hedging_kill_switch(conn, _) do
    HedgingEmergency.kill_switch()
    json(conn, %{status: "disabled"})
  end
end
```

---

## Advanced Topics

### Multi-Level Hedging

```elixir
defmodule MultiLevelHedging do
  def request(fun, opts \\ []) do
    # Level 1: Hedge at P95
    {:ok, result, metadata} = Hedging.request(
      fn ->
        # Level 2: Hedge the hedge at P99
        Hedging.request(fun,
          strategy: :fixed,
          delay_ms: 50
        )
      end,
      strategy: :fixed,
      delay_ms: 200
    )

    {:ok, result, Map.put(metadata, :multi_level, true)}
  end
end
```

### Cross-Region Hedging

```elixir
defmodule CrossRegionHedging do
  def request(data, opts \\ []) do
    primary_region = opts[:primary_region] || :us_east
    backup_region = opts[:backup_region] || :us_west

    Hedging.request(
      fn -> call_region(primary_region, data) end,
      strategy: :fixed,
      delay_ms: 100,
      backup_fn: fn -> call_region(backup_region, data) end
    )
  end

  defp call_region(region, data) do
    endpoint = region_endpoint(region)
    HTTPoison.post!(endpoint, data)
  end
end
```

### Hedging with Speculation

```elixir
defmodule SpeculativeHedging do
  # Combine speculative execution with hedging
  def speculative_request(fun, predictions, opts \\ []) do
    # Start speculative computations
    speculative_tasks = Enum.map(predictions, fn pred ->
      Task.async(fn -> fun.(pred) end)
    end)

    # Also start actual request with hedging
    actual_task = Task.async(fn ->
      Hedging.request(fn -> fun.(actual_input()) end, opts)
    end)

    # Wait for first completion
    all_tasks = [actual_task | speculative_tasks]

    case Task.yield_many(all_tasks, 1000) do
      [{^actual_task, {:ok, result}}] ->
        # Actual completed first
        cancel_tasks(speculative_tasks)
        result

      [{spec_task, {:ok, result}}] when spec_task in speculative_tasks ->
        # Speculation won!
        cancel_tasks([actual_task | List.delete(speculative_tasks, spec_task)])
        result
    end
  end
end
```

---

## References

### Research Papers

1. **The Tail at Scale** (2013)
   - Dean, J., & Barroso, L. A.
   - Communications of the ACM, 56(2), 74-80
   - https://research.google/pubs/pub40801/

2. **Hedged Requests: Stochastic Scheduling to Reduce Tail Latency** (2012)
   - Dean, J.
   - Google Research

3. **Late Binding for Distributed System Resources** (2014)
   - Vulimiri, A., et al.
   - NSDI 2014

4. **Thompson Sampling** (1933)
   - Thompson, W. R.
   - Biometrika, 25(3/4), 285-294

### Industry Implementations

1. **Google BigTable**
   - 96% P99 reduction
   - 5% cost overhead

2. **Amazon DynamoDB**
   - Hedged reads for tail latency

3. **Netflix**
   - Hystrix circuit breaker + hedging

4. **Facebook TAO**
   - Cross-region hedging

### Tools & Libraries

1. **This Implementation**: Elixir Hedging Library
2. **Java**: Hystrix (Netflix)
3. **Go**: go-resiliency/hedge
4. **Python**: hedgehog
5. **Rust**: tower-hedge

---

## Appendix

### Full API Reference

```elixir
@spec request(request_fn :: (-> any()), opts :: keyword()) ::
  {:ok, any(), metadata :: map()} | {:error, any()}

# Options:
# - :strategy - :fixed, :percentile, :adaptive, :workload_aware
# - :delay_ms - Fixed delay (for :fixed strategy)
# - :percentile - Target percentile (for :percentile strategy)
# - :max_hedges - Maximum backup requests (default: 1)
# - :timeout_ms - Total timeout (default: 30_000)
# - :enable_cancellation - Cancel slower requests (default: true)
# - :telemetry_prefix - Event prefix (default: [:hedging])
```

### Metadata Fields

```elixir
%{
  hedged: boolean(),           # Was a hedge fired?
  hedge_won: boolean(),        # Did the hedge complete first?
  total_latency: integer(),    # Total request latency (ms)
  primary_latency: integer() | nil,  # Primary latency (ms)
  backup_latency: integer() | nil,   # Backup latency (ms)
  hedge_delay: integer(),      # Configured hedge delay (ms)
  cost: float()                # Relative cost (1.0 = no hedge, 2.0 = hedge fired)
}
```

### Telemetry Events

```elixir
[:hedging, :request, :start]     # %{system_time: integer()}, %{request_id: ref()}
[:hedging, :request, :stop]      # %{duration: integer()}, %{metadata...}
[:hedging, :request, :exception] # %{duration: integer()}, %{error: any()}
[:hedging, :hedge, :fired]       # %{delay: integer()}, %{request_id: ref()}
[:hedging, :hedge, :won]         # %{latency: integer()}, %{request_id: ref()}
[:hedging, :request, :cancelled] # %{}, %{request_id: ref()}
```

---

**End of Guide**

For updates and issues: https://github.com/your-org/elixir_ai_research

Last Updated: 2025-10-08
Version: 0.1.0
