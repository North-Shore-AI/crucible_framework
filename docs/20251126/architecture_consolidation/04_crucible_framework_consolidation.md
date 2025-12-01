# Crucible Framework Consolidation

**Date:** 2025-11-26
**Status:** Design Proposal

---

## Objective

Refactor `crucible_framework` to:
1. Depend on `crucible_ir` instead of defining IR locally
2. Delegate to specialized libraries instead of reimplementing functionality
3. Focus on orchestration, pipeline execution, and backend management

## Current Structure

```
crucible_framework/
├── lib/
│   ├── crucible_framework.ex
│   ├── crucible_framework/
│   │   ├── application.ex
│   │   ├── persistence.ex
│   │   ├── repo.ex
│   │   └── persistence/
│   │       ├── experiment_record.ex
│   │       ├── run_record.ex
│   │       └── artifact_record.ex
│   └── crucible/
│       ├── registry.ex
│       ├── trace_integration.ex
│       ├── backend.ex
│       ├── backend_manager.ex
│       ├── stage.ex
│       ├── context.ex
│       ├── data/
│       │   ├── provider.ex
│       │   └── in_memory.ex
│       ├── fairness/              # REMOVE - delegate to ExFairness
│       │   ├── adapter.ex
│       │   ├── ex_fairness_adapter.ex
│       │   └── noop.ex
│       ├── stage/
│       │   ├── data_load.ex
│       │   ├── bench.ex           # SIMPLIFY - delegate to crucible_bench
│       │   ├── validate.ex
│       │   ├── guardrails.ex      # SIMPLIFY - delegate to LlmGuard
│       │   ├── backend_call.ex
│       │   ├── data_checks.ex
│       │   ├── report.ex
│       │   ├── fairness.ex        # SIMPLIFY - delegate to ExFairness
│       │   ├── guardrails/
│       │   │   ├── adapter.ex
│       │   │   └── noop.ex
│       │   └── analysis/
│       │       ├── surrogate_validation.ex
│       │       ├── filter.ex
│       │       ├── metrics.ex
│       │       └── tda_validation.ex
│       ├── backend/
│       │   └── tinkex/
│       │       ├── tinkex.ex
│       │       ├── client.ex
│       │       └── live_client.ex
│       ├── pipeline/
│       │   └── runner.ex
│       ├── protocols/
│       │   └── jason_encoder.ex
│       ├── ir/                    # REMOVE - move to crucible_ir
│       │   ├── experiment.ex
│       │   ├── dataset_ref.ex
│       │   ├── backend_ref.ex
│       │   ├── stage_def.ex
│       │   ├── reliability_config.ex
│       │   ├── ensemble_config.ex
│       │   ├── hedging_config.ex
│       │   ├── guardrail_config.ex
│       │   ├── stats_config.ex
│       │   ├── fairness_config.ex
│       │   └── output_spec.ex
│       └── analysis/
│           └── surrogate_adapter.ex
```

## Proposed Structure

```
crucible_framework/
├── lib/
│   ├── crucible_framework.ex          # Main entry point
│   ├── crucible_framework/
│   │   ├── application.ex
│   │   ├── persistence.ex
│   │   ├── repo.ex
│   │   └── persistence/
│   │       ├── experiment_record.ex
│   │       ├── run_record.ex
│   │       └── artifact_record.ex
│   └── crucible/
│       ├── registry.ex                # Stage/backend registry
│       ├── context.ex                 # Runtime context (KEEP)
│       ├── stage.ex                   # Stage behaviour (KEEP)
│       ├── backend.ex                 # Backend behaviour (KEEP)
│       ├── backend_manager.ex         # Backend lifecycle (KEEP)
│       ├── ir.ex                      # Backwards-compat aliases (NEW)
│       ├── data/
│       │   ├── provider.ex
│       │   └── in_memory.ex
│       ├── stage/
│       │   ├── data_load.ex           # KEEP - core stage
│       │   ├── validate.ex            # KEEP - core stage
│       │   ├── data_checks.ex         # KEEP - core stage
│       │   ├── backend_call.ex        # KEEP - core stage
│       │   ├── report.ex              # KEEP - core stage
│       │   └── analysis/
│       │       ├── surrogate_validation.ex  # KEEP - pluggable
│       │       ├── filter.ex                # KEEP - pluggable
│       │       ├── metrics.ex               # KEEP - pluggable
│       │       └── tda_validation.ex        # KEEP - pluggable
│       ├── backend/
│       │   └── tinkex/
│       │       ├── tinkex.ex          # KEEP
│       │       ├── client.ex          # KEEP
│       │       └── live_client.ex     # KEEP
│       └── pipeline/
│           └── runner.ex              # KEEP - core orchestration
```

## Key Changes

### 1. Remove IR Directory

**Before:**
```elixir
# lib/crucible/ir/experiment.ex
defmodule Crucible.IR.Experiment do
  # ... local definition
end
```

**After:**
```elixir
# lib/crucible/ir.ex (backwards compatibility)
defmodule Crucible.IR do
  @moduledoc """
  Backwards-compatible aliases for IR structs.

  ## Migration Guide

  Replace:
      alias Crucible.IR.Experiment

  With:
      alias CrucibleIR.Experiment
  """

  @deprecated "Use CrucibleIR.Experiment instead"
  defdelegate experiment, to: CrucibleIR.Experiment

  # Module aliases for struct pattern matching
  defmodule Experiment do
    @moduledoc false
    defstruct CrucibleIR.Experiment.__struct__() |> Map.keys()

    def __struct__, do: CrucibleIR.Experiment.__struct__()
    def __struct__(kv), do: CrucibleIR.Experiment.__struct__(kv)
  end

  # Similar for other structs...
end
```

### 2. Simplify Stage.Bench

**Before (472 lines):**
```elixir
defmodule Crucible.Stage.Bench do
  @behaviour Crucible.Stage

  # ... 400+ lines of statistical logic
  defp run_single_test(:ttest, group1, group2, alpha, _opts) do
    # ... reimplementation
  end
end
```

**After (< 50 lines):**
```elixir
defmodule Crucible.Stage.Bench do
  @moduledoc """
  Statistical benchmarking stage.

  Delegates to crucible_bench library.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @impl true
  def run(%Context{} = ctx, opts) do
    # Use library-provided stage if available
    case Application.get_env(:crucible_framework, :bench_stage) do
      nil ->
        CrucibleBench.Stage.run(ctx, opts)

      module when is_atom(module) ->
        module.run(ctx, opts)
    end
  end

  @impl true
  def describe(opts), do: CrucibleBench.Stage.describe(opts)
end
```

### 3. Simplify Stage.Fairness

**Before (339 lines):**
```elixir
defmodule Crucible.Stage.Fairness do
  # ... complex adapter logic and data extraction
end
```

**After (< 30 lines):**
```elixir
defmodule Crucible.Stage.Fairness do
  @moduledoc """
  Fairness evaluation stage.

  Delegates to ExFairness library via its stage implementation.
  """

  @behaviour Crucible.Stage

  @impl true
  def run(%Crucible.Context{} = ctx, opts) do
    case Application.get_env(:crucible_framework, :fairness_stage) do
      nil -> ExFairness.Stage.run(ctx, opts)
      module -> module.run(ctx, opts)
    end
  end

  @impl true
  def describe(opts) do
    %{stage: :fairness, description: "Fairness evaluation via ExFairness"}
  end
end
```

### 4. Simplify Stage.Guardrails

**Before:**
```elixir
defmodule Crucible.Stage.Guardrails do
  # ... adapter pattern with noop fallback
end
```

**After:**
```elixir
defmodule Crucible.Stage.Guardrails do
  @moduledoc """
  Safety guardrails stage.

  Delegates to LlmGuard library.
  """

  @behaviour Crucible.Stage

  @impl true
  def run(%Crucible.Context{} = ctx, opts) do
    case Application.get_env(:crucible_framework, :guardrails_stage) do
      nil -> LlmGuard.Stage.run(ctx, opts)
      :noop -> {:ok, ctx}
      module -> module.run(ctx, opts)
    end
  end
end
```

### 5. Update Dependencies

**Before (mix.exs):**
```elixir
defp deps do
  [
    {:jason, "~> 1.4"},
    {:ecto_sql, "~> 3.10"},
    {:postgrex, "~> 0.17"},
    # Optional integrations
    {:crucible_bench, "~> 0.2.0", optional: true},
    {:crucible_ensemble, "~> 0.2.0", optional: true},
    # ...
  ]
end
```

**After:**
```elixir
defp deps do
  [
    # Core IR dependency
    {:crucible_ir, "~> 0.1.0"},

    # Infrastructure
    {:jason, "~> 1.4"},
    {:ecto_sql, "~> 3.10"},
    {:postgrex, "~> 0.17"},

    # Required reliability libraries
    {:crucible_bench, "~> 0.2.0"},
    {:crucible_telemetry, "~> 0.2.0"},
    {:crucible_trace, "~> 0.2.0"},

    # Optional libraries (provide stages)
    {:crucible_ensemble, "~> 0.2.0", optional: true},
    {:crucible_hedging, "~> 0.2.0", optional: true},
    {:crucible_adversary, "~> 0.3.0", optional: true},
    {:crucible_xai, "~> 0.3.0", optional: true},
    {:ex_fairness, "~> 0.3.0", optional: true},
    {:llm_guard, "~> 0.2.0", optional: true},
    {:ex_data_check, "~> 0.2.0", optional: true}
  ]
end
```

### 6. Update Context to Use crucible_ir

**Before:**
```elixir
defmodule Crucible.Context do
  alias Crucible.IR.Experiment

  @type t :: %__MODULE__{
    experiment: Experiment.t(),
    # ...
  }
end
```

**After:**
```elixir
defmodule Crucible.Context do
  alias CrucibleIR.Experiment

  @type t :: %__MODULE__{
    experiment: Experiment.t(),
    # ...
  }
end
```

## Configuration Changes

### Stage Registry

```elixir
# config/config.exs
config :crucible_framework,
  stage_registry: %{
    # Core stages (always available)
    data_load: Crucible.Stage.DataLoad,
    validate: Crucible.Stage.Validate,
    data_checks: Crucible.Stage.DataChecks,
    backend_call: Crucible.Stage.BackendCall,
    report: Crucible.Stage.Report,

    # Library-provided stages (delegates to libraries)
    bench: Crucible.Stage.Bench,
    fairness: Crucible.Stage.Fairness,
    guardrails: Crucible.Stage.Guardrails,
    ensemble: Crucible.Stage.Ensemble,
    hedging: Crucible.Stage.Hedging,
    adversarial: Crucible.Stage.Adversarial,
    xai: Crucible.Stage.XAI,

    # Analysis stages (pluggable via adapters)
    analysis_metrics: Crucible.Stage.Analysis.Metrics,
    analysis_surrogate_validation: Crucible.Stage.Analysis.SurrogateValidation,
    analysis_tda_validation: Crucible.Stage.Analysis.TDAValidation,
    analysis_filter: Crucible.Stage.Analysis.Filter
  },

  # Override default stage implementations
  bench_stage: CrucibleBench.Stage,
  fairness_stage: ExFairness.Stage,
  guardrails_stage: LlmGuard.Stage,
  ensemble_stage: CrucibleEnsemble.Stage,
  hedging_stage: CrucibleHedging.Stage
```

## API Changes

### Main Module

```elixir
defmodule CrucibleFramework do
  @moduledoc """
  Reliability-first experiment engine for LLM training and evaluation.

  ## Quick Start

      alias CrucibleIR.{Experiment, BackendRef, StageDef}
      alias CrucibleIR.Reliability

      experiment = %Experiment{
        id: "my_experiment",
        backend: %BackendRef{id: :tinkex},
        pipeline: [
          %StageDef{name: :data_load},
          %StageDef{name: :backend_call},
          %StageDef{name: :bench}
        ],
        reliability: %Reliability.Config{
          stats: %Reliability.Stats{tests: [:ttest], alpha: 0.05}
        }
      }

      {:ok, ctx} = CrucibleFramework.run(experiment)
  """

  alias Crucible.{Context, Pipeline.Runner}
  alias CrucibleIR.Experiment

  @doc """
  Runs an experiment and returns the final context.
  """
  @spec run(Experiment.t(), keyword()) :: {:ok, Context.t()} | {:error, term()}
  def run(%Experiment{} = experiment, opts \\ []) do
    Runner.run(experiment, opts)
  end

  @doc """
  Validates an experiment definition without executing.
  """
  @spec validate(Experiment.t()) :: :ok | {:error, [term()]}
  def validate(%Experiment{} = experiment) do
    CrucibleIR.validate(experiment)
  end
end
```

## Testing Changes

### Update Test Imports

```elixir
# Before
alias Crucible.IR.{Experiment, BackendRef, StageDef}

# After
alias CrucibleIR.{Experiment, BackendRef, StageDef}
```

### Add Integration Tests

```elixir
defmodule CrucibleFramework.IntegrationTest do
  use ExUnit.Case

  alias CrucibleIR.{Experiment, BackendRef, StageDef}
  alias CrucibleIR.Reliability

  describe "library integration" do
    test "bench stage delegates to crucible_bench" do
      experiment = %Experiment{
        id: "bench_test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :bench}],
        reliability: %Reliability.Config{
          stats: %Reliability.Stats{tests: [:ttest]}
        }
      }

      # Prepare context with test data
      ctx = %Context{
        experiment: experiment,
        outputs: generate_test_outputs()
      }

      {:ok, result} = Crucible.Stage.Bench.run(ctx, %{})
      assert result.metrics.bench != nil
    end

    test "fairness stage delegates to ex_fairness" do
      # Similar test for fairness
    end

    test "guardrails stage delegates to llm_guard" do
      # Similar test for guardrails
    end
  end
end
```

## Migration Checklist

### Phase 1: Prepare
- [ ] Ensure crucible_ir is published
- [ ] Add crucible_ir dependency
- [ ] Create Crucible.IR alias module

### Phase 2: Update Imports
- [ ] Update all `Crucible.IR.*` imports to `CrucibleIR.*`
- [ ] Update Context type specs
- [ ] Update Pipeline.Runner

### Phase 3: Simplify Stages
- [ ] Refactor Stage.Bench to delegate
- [ ] Refactor Stage.Fairness to delegate
- [ ] Refactor Stage.Guardrails to delegate
- [ ] Remove fairness/ directory
- [ ] Remove guardrails/ subdirectory

### Phase 4: Cleanup
- [ ] Remove lib/crucible/ir/ directory
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Update CHANGELOG

### Phase 5: Release
- [ ] Version bump to 0.5.0
- [ ] Publish to Hex
- [ ] Announce breaking changes
