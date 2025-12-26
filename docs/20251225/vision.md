# CrucibleFramework Vision: Thin Orchestration Layer

**Date**: 2025-12-25
**Target Version**: 1.0.0

---

## The Vision

CrucibleFramework should be a **thin orchestration layer** that:

1. Defines the `Stage` behaviour
2. Runs pipelines via `Pipeline.Runner`
3. Threads a generic `Context` through stages
4. Provides optional persistence
5. **Does NOT know about training, backends, datasets, or domain-specific analysis**

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                 │
│  tinkex_cookbook  │  cns_crucible  │  crucible_examples  │  your_app    │
│  (Recipes)        │  (Experiments) │  (Demos)            │              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CRUCIBLE TRAINING STACK                             │
│  crucible_train     - Training loops, renderers, types                   │
│  crucible_datasets  - Dataset loaders, evaluation                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CRUCIBLE ORCHESTRATION (THIS PACKAGE)                 │
│  crucible_framework - Pipeline.Runner, Stage behaviour, Context         │
│                       Registry, optional Persistence                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CRUCIBLE RELIABILITY LAYER                          │
│  crucible_ir        - Shared experiment IR types                         │
│  crucible_ensemble  - Multi-model voting                                 │
│  crucible_hedging   - Request hedging                                    │
│  crucible_bench     - Statistical testing                                │
│  crucible_trace     - Causal reasoning chains                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What STAYS in crucible_framework

### 1. Pipeline Runner (~100 lines)

```elixir
defmodule Crucible.Pipeline.Runner do
  @moduledoc """
  Executes experiment pipelines stage-by-stage.
  """

  def run(%CrucibleIR.Experiment{} = experiment, opts \\ []) do
    # Build initial context
    # Iterate stages
    # Handle errors
    # Optionally persist
  end
end
```

### 2. Stage Behaviour (~20 lines)

```elixir
defmodule Crucible.Stage do
  @moduledoc """
  Behaviour for pipeline stages.
  """

  @callback run(context :: Context.t(), opts :: map()) ::
              {:ok, Context.t()} | {:error, term()}

  @callback describe(opts :: map()) :: map()
  @optional_callbacks describe: 1
end
```

### 3. Simplified Context (~150 lines)

```elixir
defmodule Crucible.Context do
  @moduledoc """
  Runtime context threaded through pipeline stages.
  """

  @type t :: %__MODULE__{
    experiment_id: String.t(),
    run_id: String.t(),
    experiment: CrucibleIR.Experiment.t(),

    # Generic results
    outputs: list(),
    metrics: map(),
    artifacts: map(),

    # Observability
    trace: term() | nil,
    telemetry_context: map(),

    # Extension point - stages put domain-specific data here
    assigns: map()
  }

  # Helper functions for metrics, outputs, artifacts, assigns
  # Stage completion tracking
end
```

### 4. Registry (~50 lines)

```elixir
defmodule Crucible.Registry do
  @moduledoc """
  Resolves stage modules from configuration.
  """

  def stage_module(name), do: # lookup from config
end
```

### 5. Optional Persistence (~250 lines)

```elixir
defmodule CrucibleFramework.Persistence do
  # upsert_experiment/1
  # start_run/2
  # finish_run/3
  # record_artifact/2
end

# Ecto schemas for experiments, runs, artifacts
```

### 6. Trace Integration (~100 lines, simplified)

```elixir
defmodule Crucible.TraceIntegration do
  # init_trace/2
  # emit_event/4
  # emit_stage_start/3
  # emit_stage_complete/3
  # emit_stage_failed/3
end
```

---

## What MOVES OUT

### To crucible_train

| Current Module | New Location |
|----------------|--------------|
| `Crucible.Backend` | `CrucibleTrain.Backend` |
| `Crucible.BackendManager` | `CrucibleTrain.BackendManager` |
| `Crucible.Backend.Tinkex` | `tinkex_cookbook` (adapter) |
| `Crucible.Stage.BackendCall` | `CrucibleTrain.Stages.Train` |
| `Crucible.Data.Provider` | `CrucibleTrain.Dataset.Behaviour` |
| `Crucible.Stage.DataLoad` | `CrucibleTrain.Stages.DataLoad` |

### To cns_crucible or cns

| Current Module | New Location |
|----------------|--------------|
| `Crucible.Analysis.Adapter` | `Cns.Crucible.Adapters.Analysis` |
| `Crucible.Analysis.TDAAdapter` | `Cns.Crucible.Adapters.TDA` |
| `Crucible.Analysis.SurrogateAdapter` | `Cns.Crucible.Adapters.Surrogate` |
| `Crucible.Stage.Analysis.*` | `CnsCrucible.Stages.*` |

### To ExFairness

| Current Module | New Location |
|----------------|--------------|
| `Crucible.Fairness.Adapter` | `ExFairness.Crucible.Adapter` |
| `Crucible.Fairness.ExFairnessAdapter` | `ExFairness.Crucible.Stage` |
| `Crucible.Stage.Fairness` | `ExFairness.Crucible.Stage` |

### DELETED (not needed)

| Module | Reason |
|--------|--------|
| `Crucible.IR` | Deprecated aliases, use CrucibleIR directly |
| `Crucible.Data.InMemory` | Move to recipe layer or crucible_datasets |
| All `*Noop` adapters | Move with their parent adapters |

---

## Simplified Context

### Before (Current)

```elixir
%Crucible.Context{
  experiment_id: "...",
  run_id: "...",
  experiment: %{},

  # Training-specific (REMOVE)
  dataset: ...,
  batches: ...,
  examples: ...,
  backend_sessions: %{},
  backend_state: %{},

  # Generic (KEEP)
  outputs: [],
  metrics: %{},
  artifacts: %{},
  trace: nil,
  telemetry_context: %{},
  assigns: %{}
}
```

### After (Target)

```elixir
%Crucible.Context{
  experiment_id: "...",
  run_id: "...",
  experiment: %CrucibleIR.Experiment{},

  # Generic outputs
  outputs: [],
  metrics: %{},
  artifacts: %{},

  # Observability
  trace: nil,
  telemetry_context: %{},

  # All domain-specific data goes here
  assigns: %{
    # Training stages put their data here
    dataset: ...,
    batches: ...,
    backend_session: ...,

    # CNS stages put their data here
    snos: ...,
    surrogates: ...,

    # Any stage can add anything
    custom: ...
  }
}
```

---

## Simplified Dependencies

### Before (Current)

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.1"},
    {:crucible_ensemble, path: "../crucible_ensemble"},
    {:crucible_hedging, path: "../crucible_hedging"},
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_trace, path: "../crucible_trace"},
    {:ex_fairness, path: "../ExFairness", optional: true},
    {:tinkex, "~> 0.1.12"},              # REMOVE
    {:ecto_sql, "~> 3.11"},
    {:postgrex, ">= 0.0.0"},
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},
    {:nx, "~> 0.7"},                      # REMOVE
    {:mox, "~> 1.1", only: :test},
    # ...
  ]
end
```

### After (Target)

```elixir
defp deps do
  [
    # Core IR
    {:crucible_ir, "~> 0.2.0"},

    # Reliability libraries (for built-in stage wrappers)
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_trace, path: "../crucible_trace"},

    # Optional persistence
    {:ecto_sql, "~> 3.11", optional: true},
    {:postgrex, ">= 0.0.0", optional: true},

    # Core utilities
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},

    # Testing
    {:mox, "~> 1.1", only: :test}
  ]
end
```

**Key removals**:
- `tinkex` - Framework doesn't know about Tinkex
- `nx` - Only needed by fairness (which moves)
- `ex_fairness` - Adapter moves there
- `crucible_ensemble` - Stages that need it import it
- `crucible_hedging` - Stages that need it import it

---

## Built-in Stages

The slimmed framework provides only thin wrapper stages:

### 1. Stage.Validate (simplified)

```elixir
defmodule Crucible.Stage.Validate do
  @behaviour Crucible.Stage

  @doc """
  Validates that pipeline stages are resolvable.
  Domain-specific validation left to domain stages.
  """
  def run(ctx, opts) do
    # Check stages resolve
    # Check experiment has required fields
    # That's it - no backend validation, no ensemble validation
  end
end
```

### 2. Stage.Bench (wrapper)

```elixir
defmodule Crucible.Stage.Bench do
  @behaviour Crucible.Stage

  @doc """
  Thin wrapper around CrucibleBench.
  Expects numeric data in assigns[:bench_data].
  """
  def run(ctx, opts) do
    # Pull data from assigns
    # Call CrucibleBench
    # Put results in metrics
  end
end
```

### 3. Stage.Report (simplified)

```elixir
defmodule Crucible.Stage.Report do
  @behaviour Crucible.Stage

  @doc """
  Generates basic reports from metrics/outputs.
  """
  def run(ctx, opts) do
    # Generate markdown/json from ctx.metrics and ctx.outputs
    # Write to file or stdout
  end
end
```

---

## Example: How Training Works After Slimming

```elixir
# In tinkex_cookbook or crucible_train

# Define training stages
defmodule CrucibleTrain.Stages.DataLoad do
  @behaviour Crucible.Stage

  def run(ctx, opts) do
    # Load dataset
    # Put in assigns[:dataset], assigns[:batches]
    {:ok, Context.assign(ctx, dataset: dataset, batches: batches)}
  end
end

defmodule CrucibleTrain.Stages.Train do
  @behaviour Crucible.Stage

  def run(ctx, opts) do
    # Get batches from assigns
    # Run training loop
    # Put checkpoint in assigns[:checkpoint_id]
    {:ok, Context.assign(ctx, checkpoint_id: checkpoint)}
  end
end

# Use in pipeline
experiment = %CrucibleIR.Experiment{
  pipeline: [
    %StageDef{name: :data_load, module: CrucibleTrain.Stages.DataLoad},
    %StageDef{name: :train, module: CrucibleTrain.Stages.Train},
    %StageDef{name: :evaluate, module: CrucibleTrain.Stages.Evaluate},
    %StageDef{name: :report, module: Crucible.Stage.Report}
  ]
}

Crucible.Pipeline.Runner.run(experiment)
```

---

## File Structure After Slimming

```
lib/
├── crucible/
│   ├── context.ex           # Simplified context
│   ├── stage.ex             # Stage behaviour
│   ├── registry.ex          # Stage lookup
│   ├── pipeline/
│   │   └── runner.ex        # Pipeline execution
│   ├── trace_integration.ex # Trace helpers
│   └── stage/
│       ├── validate.ex      # Simplified validation
│       ├── bench.ex         # CrucibleBench wrapper
│       └── report.ex        # Report generation
├── crucible_framework/
│   ├── application.ex
│   ├── repo.ex
│   └── persistence/
│       ├── experiment_record.ex
│       ├── run_record.ex
│       └── artifact_record.ex
└── crucible_framework.ex    # Entry point

test/
├── crucible/
│   ├── context_test.exs
│   ├── registry_test.exs
│   ├── pipeline/
│   │   └── runner_test.exs
│   └── stage/
│       ├── validate_test.exs
│       ├── bench_test.exs
│       └── report_test.exs
└── crucible_framework/
    └── persistence_test.exs
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Lines of Code | ~4,750 | ~1,200 |
| Dependencies | 12 | 7 |
| Stage count | 15+ | 4 |
| Adapter behaviours | 6 | 0 |
| Backend coupling | Heavy | None |
| Tinkex references | Many | Zero |
| Domain-specific code | ~60% | ~0% |

---

## Benefits

1. **Clarity**: Framework does ONE thing - orchestration
2. **Flexibility**: Any domain can build on it
3. **Testability**: Simple, mockable stages
4. **Maintainability**: 75% less code
5. **Decoupling**: Training infrastructure lives in training packages
6. **Future-proof**: New backends don't require framework changes
