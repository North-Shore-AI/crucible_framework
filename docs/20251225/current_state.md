# CrucibleFramework Current State

**Date**: 2025-12-25
**Version**: 0.4.0

---

## Overview

CrucibleFramework is an ML experimentation engine that has evolved beyond its original scope. What was designed as a "thin orchestration layer" has accumulated significant domain-specific functionality that should be extracted or removed.

---

## Module Inventory

### Core Orchestration (KEEP)

| Module | File | Purpose | Lines |
|--------|------|---------|-------|
| `CrucibleFramework` | `lib/crucible_framework.ex` | Public entry point, delegates to Runner | ~14 |
| `Crucible.Pipeline.Runner` | `lib/crucible/pipeline/runner.ex` | Executes pipeline stages sequentially | ~124 |
| `Crucible.Context` | `lib/crucible/context.ex` | Runtime context threaded through stages | ~393 |
| `Crucible.Stage` | `lib/crucible/stage.ex` | Stage behaviour definition | ~19 |
| `Crucible.Registry` | `lib/crucible/registry.ex` | Resolves backends and stages from config | ~34 |

### Persistence (KEEP - Optional)

| Module | File | Purpose |
|--------|------|---------|
| `CrucibleFramework.Application` | `lib/crucible_framework/application.ex` | OTP application, starts Repo |
| `CrucibleFramework.Repo` | `lib/crucible_framework/repo.ex` | Ecto Repo |
| `CrucibleFramework.Persistence` | `lib/crucible_framework/persistence.ex` | Experiment/run persistence helpers |
| `CrucibleFramework.Persistence.ExperimentRecord` | `lib/crucible_framework/persistence/experiment_record.ex` | Ecto schema |
| `CrucibleFramework.Persistence.RunRecord` | `lib/crucible_framework/persistence/run_record.ex` | Ecto schema |
| `CrucibleFramework.Persistence.ArtifactRecord` | `lib/crucible_framework/persistence/artifact_record.ex` | Ecto schema |

### Backend Infrastructure (REMOVE - Move to crucible_train or dedicated packages)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Backend` | `lib/crucible/backend.ex` | Backend behaviour | Training-specific, move to crucible_train |
| `Crucible.BackendManager` | `lib/crucible/backend_manager.ex` | Backend state/session management | Training-specific |
| `Crucible.Backend.Tinkex` | `lib/crucible/backend/tinkex.ex` | Tinkex implementation (~263 lines) | Should be in tinkex_cookbook |
| `Crucible.Backend.Tinkex.Client` | `lib/crucible/backend/tinkex/client.ex` | Tinkex client behaviour | Should be in tinkex_cookbook |
| `Crucible.Backend.Tinkex.LiveClient` | `lib/crucible/backend/tinkex/live_client.ex` | Live Tinkex implementation | Should be in tinkex_cookbook |

### Data Loading (REMOVE - Move to crucible_datasets or recipe layer)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Data.Provider` | `lib/crucible/data/provider.ex` | Dataset provider behaviour | Domain-specific |
| `Crucible.Data.InMemory` | `lib/crucible/data/in_memory.ex` | In-memory dataset provider | Domain-specific |
| `Crucible.Stage.DataLoad` | `lib/crucible/stage/data_load.ex` | Data loading stage | Domain-specific |

### Analysis Stages (REMOVE - Move to CNS or dedicated packages)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Analysis.Adapter` | `lib/crucible/analysis/adapter.ex` | Analysis adapter behaviour | Domain-specific (CNS) |
| `Crucible.Analysis.Noop` | `lib/crucible/analysis/noop.ex` | No-op adapter | Domain-specific |
| `Crucible.Analysis.TDAAdapter` | `lib/crucible/analysis/tda_adapter.ex` | TDA adapter behaviour | Domain-specific (CNS) |
| `Crucible.Analysis.TDANoop` | `lib/crucible/analysis/tda_noop.ex` | No-op TDA | Domain-specific |
| `Crucible.Analysis.SurrogateAdapter` | `lib/crucible/analysis/surrogate_adapter.ex` | Surrogate adapter behaviour | Domain-specific (CNS) |
| `Crucible.Analysis.SurrogateNoop` | `lib/crucible/analysis/surrogate_noop.ex` | No-op surrogate | Domain-specific |
| `Crucible.Stage.Analysis.Metrics` | `lib/crucible/stage/analysis/metrics.ex` | Analysis metrics stage | Domain-specific |
| `Crucible.Stage.Analysis.TDAValidation` | `lib/crucible/stage/analysis/tda_validation.ex` | TDA validation stage | Domain-specific |
| `Crucible.Stage.Analysis.SurrogateValidation` | `lib/crucible/stage/analysis/surrogate_validation.ex` | Surrogate validation stage | Domain-specific |
| `Crucible.Stage.Analysis.Filter` | `lib/crucible/stage/analysis/filter.ex` | SNO filtering stage | Domain-specific (CNS) |

### Fairness (SIMPLIFY or REMOVE)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Fairness.Adapter` | `lib/crucible/fairness/adapter.ex` | Fairness adapter behaviour | Could be separate package |
| `Crucible.Fairness.Noop` | `lib/crucible/fairness/noop.ex` | No-op fairness | Supporting noop |
| `Crucible.Fairness.ExFairnessAdapter` | `lib/crucible/fairness/ex_fairness_adapter.ex` | ExFairness integration (~329 lines) | Should be in ExFairness |
| `Crucible.Stage.Fairness` | `lib/crucible/stage/fairness.ex` | Fairness evaluation stage (~339 lines) | Heavy, domain-specific |

### Guardrails (SIMPLIFY)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Stage.Guardrails.Adapter` | `lib/crucible/stage/guardrails/adapter.ex` | Guardrail adapter behaviour | Thin, acceptable |
| `Crucible.Stage.Guardrails.Noop` | `lib/crucible/stage/guardrails/noop.ex` | No-op guardrails | Thin, acceptable |
| `Crucible.Stage.Guardrails` | `lib/crucible/stage/guardrails.ex` | Guardrails stage | Thin, acceptable |

### Domain Stages (EVALUATE - Some may stay as examples)

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.Stage.BackendCall` | `lib/crucible/stage/backend_call.ex` | Backend training/sampling (~734 lines) | HEAVY - training logic |
| `Crucible.Stage.Validate` | `lib/crucible/stage/validate.ex` | Pre-flight validation (~527 lines) | Useful but couples to IR |
| `Crucible.Stage.Bench` | `lib/crucible/stage/bench.ex` | Statistical benchmarking (~471 lines) | Wraps crucible_bench |
| `Crucible.Stage.Report` | `lib/crucible/stage/report.ex` | Report generation | Acceptable |
| `Crucible.Stage.DataChecks` | `lib/crucible/stage/data_checks.ex` | Data validation | Acceptable, thin |

### Integration & Compatibility

| Module | File | Purpose | Issue |
|--------|------|---------|-------|
| `Crucible.IR` | `lib/crucible/ir.ex` | Backwards-compat aliases | Deprecated, remove in v1.0 |
| `Crucible.TraceIntegration` | `lib/crucible/trace_integration.ex` | crucible_trace integration (~336 lines) | Acceptable |
| `Crucible.Protocols.DeepJason` | `lib/crucible/protocols/jason_encoder.ex` | JSON encoding helpers | Acceptable |

---

## Dependencies Analysis

### Current mix.exs Dependencies

```elixir
# Shared IR
{:crucible_ir, "~> 0.1.1"},  # KEEP - Core

# Component Libraries
{:crucible_ensemble, path: "../crucible_ensemble"},  # KEEP
{:crucible_hedging, path: "../crucible_hedging"},    # KEEP
{:crucible_bench, path: "../crucible_bench"},        # KEEP
{:crucible_trace, path: "../crucible_trace"},        # KEEP

# Domain Libraries (optional - for full feature set)
{:ex_fairness, path: "../ExFairness", optional: true},  # REMOVE - move adapter there

# Backend Integration
{:tinkex, "~> 0.1.12"},  # REMOVE - should not depend on tinkex

# Core Dependencies
{:ecto_sql, "~> 3.11"},   # KEEP (optional for persistence)
{:postgrex, ">= 0.0.0"},  # KEEP (optional for persistence)
{:jason, "~> 1.4"},       # KEEP
{:telemetry, "~> 1.2"},   # KEEP
{:nx, "~> 0.7"},          # REMOVE - not needed for orchestration
```

### Dependency Problems

1. **tinkex dependency**: Framework should not know about Tinkex
2. **nx dependency**: Only needed by Fairness adapter (which should move)
3. **Optional ex_fairness**: Adapter code should live there, not here

---

## Current Context Struct

The `Crucible.Context` struct has training/backend-specific fields that should be generalized:

```elixir
@type t :: %__MODULE__{
  experiment_id: String.t(),
  run_id: String.t(),
  experiment: Experiment.t(),

  # Data fields (domain-specific)
  dataset: term() | nil,      # Training-specific
  batches: Enumerable.t() | nil,  # Training-specific
  examples: list() | nil,     # Training-specific

  # Backend fields (training-specific)
  backend_sessions: %{atom() => term()},  # Training-specific
  backend_state: map(),       # Training-specific

  # Results (generic)
  outputs: list(),            # KEEP - generic
  metrics: map(),             # KEEP - generic
  artifacts: map(),           # KEEP - generic

  # Observability (generic)
  trace: term() | nil,        # KEEP - generic
  telemetry_context: map(),   # KEEP - generic

  # Extension point
  assigns: map()              # KEEP - generic
}
```

---

## Lines of Code Summary

| Category | Approximate LOC |
|----------|-----------------|
| Core Orchestration | ~600 |
| Persistence | ~250 |
| Backend/Tinkex | ~400 |
| Data Loading | ~100 |
| Analysis (CNS) | ~500 |
| Fairness | ~700 |
| Domain Stages | ~1,800 |
| Integration | ~400 |
| **Total** | **~4,750** |

### Target After Slimming

| Category | Target LOC |
|----------|------------|
| Core Orchestration | ~600 (same) |
| Persistence (optional) | ~250 (same) |
| Domain Stages (examples) | ~200 |
| Integration | ~100 |
| **Total** | **~1,150** |

**Reduction: ~75%**

---

## Test Files

| Test | Tests |
|------|-------|
| `test/crucible_framework/persistence_test.exs` | Persistence layer |
| `test/crucible/fairness/noop_test.exs` | Fairness noop |
| `test/crucible/stage/data_checks_test.exs` | Data checks stage |
| `test/crucible/stage/guardrails_test.exs` | Guardrails stage |
| `test/crucible/stage/data_load_test.exs` | Data loading (REMOVE) |
| `test/crucible/backend/tinkex_test.exs` | Tinkex backend (REMOVE) |
| `test/crucible/trace_integration_test.exs` | Trace integration |
| `test/crucible/ir/experiment_test.exs` | IR (REMOVE - in crucible_ir) |
| `test/crucible/stage/validate_test.exs` | Validation stage |
| `test/crucible/stage/fairness_test.exs` | Fairness stage |
| `test/crucible/stage/bench_test.exs` | Bench stage |
| `test/crucible/stage/backend_call_test.exs` | Backend call (REMOVE) |
| `test/crucible/pipeline/runner_test.exs` | Pipeline runner |
| `test/crucible/context_test.exs` | Context |

---

## Configuration Surface

Currently configured in application config:

```elixir
config :crucible_framework,
  # Backend registry - REMOVE
  backends: %{tinkex: Crucible.Backend.Tinkex},

  # Stage registry - SIMPLIFY
  stage_registry: %{
    data_load: Crucible.Stage.DataLoad,
    data_checks: Crucible.Stage.DataChecks,
    guardrails: Crucible.Stage.Guardrails,
    backend_call: Crucible.Stage.BackendCall,  # REMOVE
    # ... many more
  },

  # Adapters - REMOVE most
  analysis_adapter: Crucible.Analysis.Noop,
  analysis_tda_adapter: Crucible.Analysis.TDANoop,
  analysis_surrogate_adapter: Crucible.Analysis.SurrogateNoop,
  fairness_adapter: Crucible.Fairness.Noop,
  guardrail_adapter: Crucible.Stage.Guardrails.Noop,

  # Tinkex client - REMOVE
  tinkex_client: Crucible.Backend.Tinkex.LiveClient,

  # Persistence - KEEP
  enable_repo: true
```

---

## Key Observations

1. **Backend abstraction is training-specific**: The entire `Crucible.Backend` behaviour with `train_step`, `save_checkpoint`, `create_sampler` assumes ML training. This should be in `crucible_train`.

2. **BackendCall stage is massive**: At 734 lines, it handles:
   - Ensemble voting
   - Request hedging
   - Training steps
   - Checkpoint management
   - Sampling
   - Trace integration

   This is doing too much and couples the framework to training concerns.

3. **Analysis adapters are CNS-specific**: TDA, surrogates, beta-1, fragility - all CNS terminology that doesn't belong in a generic orchestration framework.

4. **Fairness is self-contained**: The fairness adapter and stage could be extracted to ExFairness with minimal impact.

5. **Context has too many training-specific fields**: `backend_sessions`, `backend_state`, `batches`, etc.

6. **Tinkex is hardcoded**: Direct dependency on `tinkex` package, Tinkex-specific types in Backend.Tinkex implementation.
