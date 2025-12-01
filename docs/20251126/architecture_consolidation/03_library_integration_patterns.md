# Library Integration Patterns

**Date:** 2025-11-26
**Status:** Design Proposal

---

## Overview

This document defines standard patterns for how Crucible ecosystem libraries should integrate with the shared IR and with each other.

## Pattern Categories

1. **IR Consumer** - Libraries that use IR configs as input
2. **Stage Provider** - Libraries that provide pipeline stages
3. **Adapter-Based** - Libraries integrated via adapters
4. **Direct Integration** - Libraries called directly

---

## Pattern 1: IR Consumer

Libraries that accept IR config structs as their primary configuration.

### Example: crucible_bench

```elixir
# Before: Library has own config
defmodule CrucibleBench do
  def compare(group1, group2, opts \\ [])
end

# After: Library accepts IR config directly
defmodule CrucibleBench do
  alias CrucibleIR.Reliability.Stats

  @doc """
  Run comparison tests using Stats configuration.
  """
  def compare(group1, group2, %Stats{} = config) do
    tests = config.tests
    alpha = config.alpha
    # ... use config directly
  end

  # Backwards compatible: also accept keyword opts
  def compare(group1, group2, opts) when is_list(opts) do
    config = struct(Stats, opts)
    compare(group1, group2, config)
  end
end
```

### Example: crucible_ensemble

```elixir
defmodule CrucibleEnsemble do
  alias CrucibleIR.Reliability.Ensemble

  @doc """
  Execute ensemble with configuration.
  """
  def vote(responses, %Ensemble{} = config) do
    case config.strategy do
      :majority -> Voting.majority(responses, config.min_agreement)
      :weighted -> Voting.weighted(responses, config.weights)
      :best_confidence -> Voting.best_confidence(responses)
      :unanimous -> Voting.unanimous(responses)
    end
  end

  @doc """
  Create executor from config.
  """
  def executor(%Ensemble{} = config) do
    %Executor{
      models: config.models,
      mode: config.execution_mode,
      timeout: config.timeout_ms,
      weights: config.weights
    }
  end
end
```

### Example: crucible_hedging

```elixir
defmodule CrucibleHedging do
  alias CrucibleIR.Reliability.Hedging

  @doc """
  Create hedger from config.
  """
  def hedger(%Hedging{strategy: :off}), do: {:ok, :no_hedging}

  def hedger(%Hedging{} = config) do
    strategy_module = strategy_to_module(config.strategy)

    {:ok, %Hedger{
      strategy: strategy_module,
      delay_ms: config.delay_ms,
      percentile: config.percentile,
      max_hedges: config.max_hedges,
      budget_percent: config.budget_percent
    }}
  end

  defp strategy_to_module(:fixed), do: CrucibleHedging.Strategy.Fixed
  defp strategy_to_module(:percentile), do: CrucibleHedging.Strategy.Percentile
  defp strategy_to_module(:adaptive), do: CrucibleHedging.Strategy.Adaptive
  defp strategy_to_module(:workload_aware), do: CrucibleHedging.Strategy.WorkloadAware
end
```

---

## Pattern 2: Stage Provider

Libraries that provide implementations of the `Crucible.Stage` behaviour.

### Definition

```elixir
# In crucible_framework
defmodule Crucible.Stage do
  @moduledoc """
  Behaviour for pipeline stages.
  """

  alias Crucible.Context

  @callback run(context :: Context.t(), opts :: map()) ::
    {:ok, Context.t()} | {:error, term()}

  @callback describe(opts :: map()) :: map()

  @optional_callbacks describe: 1
end
```

### Example: crucible_bench providing a stage

```elixir
# In crucible_bench library
defmodule CrucibleBench.Stage do
  @moduledoc """
  Pipeline stage for statistical benchmarking.

  This module can be used directly as a stage in crucible_framework pipelines.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias CrucibleIR.Reliability.Stats

  @impl true
  def run(%Context{} = ctx, opts) do
    stats_config = ctx.experiment.reliability.stats
    merged_config = merge_opts(stats_config, opts)

    data_groups = extract_data(ctx, opts)

    case CrucibleBench.analyze(data_groups, merged_config) do
      {:ok, results} ->
        {:ok, Context.put_metric(ctx, :bench, results)}

      {:error, reason} ->
        {:error, {:bench_failed, reason}}
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: :bench,
      description: "Statistical benchmarking via crucible_bench",
      tests: Map.get(opts, :tests, [])
    }
  end

  defp merge_opts(%Stats{} = config, opts) do
    %Stats{config |
      tests: Map.get(opts, :tests, config.tests),
      alpha: Map.get(opts, :alpha, config.alpha)
    }
  end

  defp extract_data(ctx, opts) do
    # ... extract data groups from context
  end
end
```

### Registering Library Stages

```elixir
# In crucible_framework config
config :crucible_framework,
  stage_registry: %{
    # Built-in stages
    data_load: Crucible.Stage.DataLoad,
    data_checks: Crucible.Stage.DataChecks,
    validate: Crucible.Stage.Validate,
    backend_call: Crucible.Stage.BackendCall,
    report: Crucible.Stage.Report,

    # Library-provided stages (preferred)
    bench: CrucibleBench.Stage,
    ensemble: CrucibleEnsemble.Stage,
    hedging: CrucibleHedging.Stage,
    fairness: ExFairness.Stage,
    guardrails: LlmGuard.Stage,
    adversarial: CrucibleAdversary.Stage,
    xai: CrucibleXAI.Stage
  }
```

---

## Pattern 3: Adapter-Based Integration

For libraries that don't directly use IR or when extra transformation is needed.

### Adapter Behaviour

```elixir
# In crucible_framework
defmodule Crucible.Adapter do
  @moduledoc """
  Behaviour for adapters that bridge external libraries.
  """

  @callback configure(config :: struct()) :: {:ok, term()} | {:error, term()}
  @callback execute(state :: term(), input :: term()) :: {:ok, term()} | {:error, term()}
end
```

### Example: ExFairness Adapter

```elixir
defmodule Crucible.Fairness.ExFairnessAdapter do
  @moduledoc """
  Adapter bridging ExFairness library to Crucible.
  """

  @behaviour Crucible.Adapter

  alias CrucibleIR.Reliability.Fairness

  @impl true
  def configure(%Fairness{} = config) do
    {:ok, %{
      metrics: map_metrics(config.metrics),
      threshold: config.threshold,
      fail_on_violation: config.fail_on_violation
    }}
  end

  @impl true
  def execute(state, {predictions, labels, sensitive_attrs}) do
    results = Enum.reduce(state.metrics, %{}, fn metric, acc ->
      result = run_metric(metric, predictions, labels, sensitive_attrs, state)
      Map.put(acc, metric, result)
    end)

    overall_passes = Enum.all?(results, fn {_, r} -> r.passes end)

    {:ok, %{
      overall_passes: overall_passes,
      metrics: results,
      threshold: state.threshold
    }}
  end

  # Map IR metric names to ExFairness functions
  defp map_metrics(metrics) do
    Enum.map(metrics, fn
      :demographic_parity -> {:demographic_parity, &ExFairness.demographic_parity/3}
      :equalized_odds -> {:equalized_odds, &ExFairness.equalized_odds/3}
      :equal_opportunity -> {:equal_opportunity, &ExFairness.equal_opportunity/3}
      :predictive_parity -> {:predictive_parity, &ExFairness.predictive_parity/3}
    end)
  end

  defp run_metric({name, func}, predictions, labels, sensitive, state) do
    case func.(predictions, labels, sensitive) do
      {:ok, result} ->
        %{
          name: name,
          value: result.disparity,
          passes: result.disparity <= state.threshold,
          details: result
        }

      {:error, reason} ->
        %{name: name, error: reason, passes: false}
    end
  end
end
```

### Example: LlmGuard Adapter

```elixir
defmodule Crucible.Guardrails.LlmGuardAdapter do
  @moduledoc """
  Adapter bridging LlmGuard to Crucible guardrails.
  """

  @behaviour Crucible.Adapter

  alias CrucibleIR.Reliability.Guardrail

  @impl true
  def configure(%Guardrail{} = config) do
    llm_guard_config = LlmGuard.Config.new(
      prompt_injection_detection: config.prompt_injection_detection,
      jailbreak_detection: config.jailbreak_detection,
      data_leakage_prevention: config.pii_detection,
      confidence_threshold: Map.get(config.options, :confidence_threshold, 0.7)
    )

    {:ok, %{
      config: llm_guard_config,
      fail_on_detection: config.fail_on_detection,
      redact_pii: config.pii_redaction
    }}
  end

  @impl true
  def execute(state, input) when is_binary(input) do
    case LlmGuard.validate_input(input, state.config) do
      {:ok, safe_input} ->
        output = if state.redact_pii do
          LlmGuard.PIIRedactor.redact(safe_input, state.config)
        else
          safe_input
        end
        {:ok, %{status: :safe, output: output}}

      {:error, :detected, details} ->
        if state.fail_on_detection do
          {:error, {:guardrail_violation, details}}
        else
          {:ok, %{status: :blocked, reason: details.reason, output: nil}}
        end
    end
  end

  def execute(state, inputs) when is_list(inputs) do
    results = Enum.map(inputs, &execute(state, &1))
    # Aggregate results
    {:ok, %{results: results}}
  end
end
```

---

## Pattern 4: Direct Integration

For simple cases where a library function is called directly without adaptation.

### Example: crucible_trace

```elixir
# In crucible_framework Stage
defmodule Crucible.Stage.BackendCall do
  @behaviour Crucible.Stage

  alias Crucible.Context
  alias CrucibleTrace

  @impl true
  def run(%Context{} = ctx, opts) do
    # Direct use of crucible_trace
    trace_id = CrucibleTrace.start_trace(ctx.experiment_id, ctx.run_id)

    result = with {:ok, ctx} <- call_backend(ctx, opts) do
      CrucibleTrace.add_event(trace_id, :backend_complete, %{
        outputs: length(ctx.outputs)
      })
      {:ok, %{ctx | trace: trace_id}}
    end

    CrucibleTrace.end_trace(trace_id)
    result
  end
end
```

### Example: crucible_telemetry

```elixir
# Direct telemetry emission in any stage
defmodule Crucible.Stage.DataLoad do
  @behaviour Crucible.Stage

  alias CrucibleTelemetry

  @impl true
  def run(%Context{} = ctx, opts) do
    CrucibleTelemetry.span([:crucible, :stage, :data_load], %{
      experiment_id: ctx.experiment_id,
      run_id: ctx.run_id
    }, fn ->
      # Load data...
      {:ok, ctx}
    end)
  end
end
```

---

## Integration Matrix

| Library | Pattern | IR Struct Used | Notes |
|---------|---------|----------------|-------|
| crucible_bench | IR Consumer + Stage | `Stats` | Provides stage |
| crucible_ensemble | IR Consumer + Stage | `Ensemble` | Provides stage |
| crucible_hedging | IR Consumer + Stage | `Hedging` | Provides stage |
| ExFairness | Adapter | `Fairness` | Needs mapping |
| LlmGuard | Adapter | `Guardrail` | Needs config conversion |
| crucible_trace | Direct | None | Utility library |
| crucible_telemetry | Direct | None | Utility library |
| crucible_datasets | IR Consumer | `DatasetRef` | Data provider |
| crucible_adversary | IR Consumer + Stage | Custom | Attack configs |
| crucible_xai | IR Consumer + Stage | Custom | XAI configs |
| ExDataCheck | Adapter | Custom | Validation adapter |

---

## Best Practices

### 1. Prefer IR Configs Over Options

```elixir
# Good: Accept IR config
def analyze(data, %Stats{} = config)

# Also good: Accept opts, convert internally
def analyze(data, opts) when is_list(opts) do
  analyze(data, struct(Stats, opts))
end

# Avoid: Only accept options
def analyze(data, opts \\ [])
```

### 2. Provide Stage Implementations

```elixir
# Good: Library provides own stage
defmodule MyLib.Stage do
  @behaviour Crucible.Stage
  # ...
end

# Less ideal: Framework wraps library
defmodule Crucible.Stage.MyLib do
  def run(ctx, opts) do
    MyLib.do_thing(ctx.data, opts)
  end
end
```

### 3. Emit Telemetry Events

```elixir
defmodule MyLib do
  def process(data) do
    :telemetry.span([:my_lib, :process], %{}, fn ->
      result = do_processing(data)
      {result, %{count: length(result)}}
    end)
  end
end
```

### 4. Support Streaming Contexts

```elixir
def process_stream(ctx) do
  ctx.batches
  |> Stream.map(&process_batch/1)
  |> Stream.run()
end
```

### 5. Preserve Context Immutability

```elixir
# Good: Return new context
def run(ctx, _opts) do
  {:ok, Context.put_metric(ctx, :result, value)}
end

# Bad: Mutate context (not possible, but conceptually)
def run(ctx, _opts) do
  ctx.metrics[:result] = value  # Won't work
end
```
