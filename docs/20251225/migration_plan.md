# CrucibleFramework Migration Plan

**Date**: 2025-12-25
**From**: v0.4.0 (current)
**To**: v1.0.0 (slim orchestration)

---

## Overview

This plan describes the step-by-step process to slim down crucible_framework from a monolithic ML experiment engine to a thin orchestration layer.

---

## Phase 1: Preparation (No Breaking Changes)

### Step 1.1: Create Destination Packages

Before removing anything, ensure destination packages exist:

| Content | Destination Package | Status |
|---------|---------------------|--------|
| Backend behaviour, BackendManager | `crucible_train` | Create new |
| DataLoad, Data.Provider | `crucible_train` or `crucible_datasets` | Create new |
| Tinkex backend | `tinkex_cookbook` | Update existing |
| Analysis adapters | `cns_crucible` | Update existing |
| Fairness adapter | `ExFairness` | Update existing |

### Step 1.2: Mark Deprecations

Add deprecation warnings to modules slated for removal:

```elixir
defmodule Crucible.Backend do
  @moduledoc """
  DEPRECATED: This module will be removed in v1.0.0.
  Use CrucibleTrain.Backend instead.
  """
  # ...
end
```

### Step 1.3: Create Feature Flags

Add configuration to disable functionality:

```elixir
config :crucible_framework,
  # Disable built-in backends (use external)
  enable_builtin_backends: false,

  # Disable built-in data loading (use external stages)
  enable_builtin_data_loading: false,

  # Disable analysis adapters (use external)
  enable_builtin_analysis: false
```

---

## Phase 2: Extract Backend Infrastructure

### Step 2.1: Move Backend Behaviour to crucible_train

**Files to move**:
- `lib/crucible/backend.ex` -> `crucible_train/lib/crucible_train/backend.ex`
- `lib/crucible/backend_manager.ex` -> `crucible_train/lib/crucible_train/backend_manager.ex`

**Update crucible_train**:
```elixir
defmodule CrucibleTrain.Backend do
  @callback init(backend_id, backend_config) :: {:ok, backend_state} | {:error, term()}
  @callback start_session(backend_state, experiment) :: {:ok, session} | {:error, term()}
  @callback train_step(session, batch) :: {:ok, map()} | {:error, term()}
  @callback save_checkpoint(session, step) :: {:ok, checkpoint_ref} | {:error, term()}
  @callback create_sampler(session, checkpoint_ref) :: {:ok, sampler} | {:error, term()}
  @callback sample(sampler, prompt, opts) :: {:ok, [binary()]} | {:error, term()}
end
```

### Step 2.2: Move Tinkex Backend to tinkex_cookbook

**Files to move**:
- `lib/crucible/backend/tinkex.ex` -> `tinkex_cookbook/lib/tinkex_cookbook/backend.ex`
- `lib/crucible/backend/tinkex/client.ex` -> `tinkex_cookbook/lib/tinkex_cookbook/backend/client.ex`
- `lib/crucible/backend/tinkex/live_client.ex` -> `tinkex_cookbook/lib/tinkex_cookbook/backend/live_client.ex`

**Update tinkex_cookbook mix.exs**:
```elixir
defp deps do
  [
    {:crucible_train, "~> 1.0"},  # For Backend behaviour
    {:tinkex, "~> 0.2.0"},        # Tinkex SDK
    # ...
  ]
end
```

### Step 2.3: Move BackendCall Stage

**File to move**:
- `lib/crucible/stage/backend_call.ex` -> `crucible_train/lib/crucible_train/stages/train.ex`

**Refactor**:
Split the 734-line BackendCall into focused stages:
- `CrucibleTrain.Stages.Train` - Training loop
- `CrucibleTrain.Stages.Sample` - Sampling/inference
- `CrucibleTrain.Stages.Checkpoint` - Checkpoint management

### Step 2.4: Update Registry

Remove backend registry from crucible_framework:

```elixir
# BEFORE
def backend_module(id) do
  case Application.fetch_env(:crucible_framework, :backends) do
    {:ok, map} -> Map.fetch(map, id)
    :error -> {:error, :no_backends_configured}
  end
end

# AFTER
# Delete this function - backends are stage implementation details
```

---

## Phase 3: Extract Data Loading

### Step 3.1: Move Data Infrastructure

**Files to move**:
- `lib/crucible/data/provider.ex` -> `crucible_train/lib/crucible_train/dataset/behaviour.ex`
- `lib/crucible/data/in_memory.ex` -> `crucible_train/lib/crucible_train/dataset/in_memory.ex`
- `lib/crucible/stage/data_load.ex` -> `crucible_train/lib/crucible_train/stages/data_load.ex`

### Step 3.2: Update Context

Remove training-specific fields from Context:

```elixir
# BEFORE
@type t :: %__MODULE__{
  # ...
  dataset: term() | nil,
  batches: Enumerable.t() | nil,
  examples: list() | nil,
  backend_sessions: %{atom() => term()},
  backend_state: map(),
  # ...
}

# AFTER
@type t :: %__MODULE__{
  experiment_id: String.t(),
  run_id: String.t(),
  experiment: Experiment.t(),
  outputs: list(),
  metrics: map(),
  artifacts: map(),
  trace: term() | nil,
  telemetry_context: map(),
  assigns: map()
}
```

### Step 3.3: Update Helper Functions

Remove/update Context helpers that reference removed fields:

```elixir
# REMOVE
def has_data?(%__MODULE__{dataset: dataset, examples: examples}) do
  not is_nil(dataset) and not is_nil(examples) and examples != []
end

def has_backend_session?(ctx, backend_id) do
  # ...
end

def get_backend_session(ctx, backend_id) do
  # ...
end
```

---

## Phase 4: Extract Analysis Adapters

### Step 4.1: Move to cns_crucible

**Files to move**:
- `lib/crucible/analysis/adapter.ex` -> `cns_crucible/lib/cns_crucible/adapters/analysis.ex`
- `lib/crucible/analysis/noop.ex` -> `cns_crucible/lib/cns_crucible/adapters/analysis_noop.ex`
- `lib/crucible/analysis/tda_adapter.ex` -> `cns_crucible/lib/cns_crucible/adapters/tda.ex`
- `lib/crucible/analysis/tda_noop.ex` -> `cns_crucible/lib/cns_crucible/adapters/tda_noop.ex`
- `lib/crucible/analysis/surrogate_adapter.ex` -> `cns_crucible/lib/cns_crucible/adapters/surrogate.ex`
- `lib/crucible/analysis/surrogate_noop.ex` -> `cns_crucible/lib/cns_crucible/adapters/surrogate_noop.ex`

### Step 4.2: Move Analysis Stages

**Files to move**:
- `lib/crucible/stage/analysis/metrics.ex` -> `cns_crucible/lib/cns_crucible/stages/metrics.ex`
- `lib/crucible/stage/analysis/tda_validation.ex` -> `cns_crucible/lib/cns_crucible/stages/tda_validation.ex`
- `lib/crucible/stage/analysis/surrogate_validation.ex` -> `cns_crucible/lib/cns_crucible/stages/surrogate_validation.ex`
- `lib/crucible/stage/analysis/filter.ex` -> `cns_crucible/lib/cns_crucible/stages/filter.ex`

---

## Phase 5: Extract Fairness

### Step 5.1: Move to ExFairness

**Files to move**:
- `lib/crucible/fairness/adapter.ex` -> `ExFairness/lib/ex_fairness/crucible/adapter.ex`
- `lib/crucible/fairness/noop.ex` -> `ExFairness/lib/ex_fairness/crucible/noop.ex`
- `lib/crucible/fairness/ex_fairness_adapter.ex` -> `ExFairness/lib/ex_fairness/crucible/stage.ex`
- `lib/crucible/stage/fairness.ex` -> `ExFairness/lib/ex_fairness/crucible/stage.ex`

### Step 5.2: Update ExFairness mix.exs

```elixir
defp deps do
  [
    {:crucible_framework, "~> 1.0"},  # For Stage behaviour
    {:nx, "~> 0.7"},
    # ...
  ]
end
```

---

## Phase 6: Simplify Remaining Stages

### Step 6.1: Simplify Validate Stage

Reduce from 527 lines to ~100 lines:
- Remove backend validation
- Remove ensemble validation
- Remove dataset validation
- Keep only stage resolution validation

### Step 6.2: Simplify Bench Stage

The Bench stage is mostly a wrapper around crucible_bench. Keep it thin:
- Remove data extraction logic (stages should provide data in assigns)
- Keep statistical test invocation

### Step 6.3: Keep Report Stage

The Report stage is already thin and useful. Keep as-is.

### Step 6.4: Evaluate Guardrails

The Guardrails stage is thin. Consider:
- Keep adapter pattern
- Move LlmGuard integration to LlmGuard package

---

## Phase 7: Update Dependencies

### Step 7.1: Remove Unnecessary Dependencies

**mix.exs changes**:
```elixir
defp deps do
  [
    # KEEP - Core
    {:crucible_ir, "~> 0.2.0"},

    # KEEP - Used by remaining stages
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_trace, path: "../crucible_trace"},

    # KEEP - Core utilities
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},

    # KEEP - Optional persistence
    {:ecto_sql, "~> 3.11", optional: true},
    {:postgrex, ">= 0.0.0", optional: true},

    # REMOVE
    # {:tinkex, "~> 0.1.12"},
    # {:nx, "~> 0.7"},
    # {:crucible_ensemble, path: "../crucible_ensemble"},
    # {:crucible_hedging, path: "../crucible_hedging"},
    # {:ex_fairness, path: "../ExFairness", optional: true},

    # Testing
    {:mox, "~> 1.1", only: :test},
    {:ex_doc, "~> 0.38", only: :dev, runtime: false},
    {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
  ]
end
```

### Step 7.2: Update Application

```elixir
def application do
  [
    mod: {CrucibleFramework.Application, []},
    extra_applications: [:logger, :telemetry]
    # Remove :crypto, :runtime_tools if not needed
  ]
end
```

---

## Phase 8: Delete Removed Code

### Step 8.1: Delete Files

After moving code to destination packages, delete:

```
rm lib/crucible/backend.ex
rm lib/crucible/backend_manager.ex
rm -rf lib/crucible/backend/
rm -rf lib/crucible/data/
rm lib/crucible/stage/data_load.ex
rm lib/crucible/stage/backend_call.ex
rm -rf lib/crucible/analysis/
rm -rf lib/crucible/stage/analysis/
rm -rf lib/crucible/fairness/
rm lib/crucible/stage/fairness.ex
rm lib/crucible/ir.ex  # Deprecated aliases
```

### Step 8.2: Delete Tests

```
rm test/crucible/backend/tinkex_test.exs
rm test/crucible/stage/data_load_test.exs
rm test/crucible/stage/backend_call_test.exs
rm test/crucible/ir/experiment_test.exs
rm test/crucible/fairness/noop_test.exs
rm test/crucible/stage/fairness_test.exs
```

---

## Phase 9: Update Documentation

### Step 9.1: Update README

- Remove references to backends
- Remove training examples
- Focus on orchestration use case
- Update architecture diagram

### Step 9.2: Update Guides

- Remove ENSEMBLE_GUIDE.md (move to crucible_ensemble)
- Remove HEDGING_GUIDE.md (move to crucible_hedging)
- Remove tinkex_integration docs
- Keep ARCHITECTURE.md (updated)
- Keep INSTRUMENTATION.md

### Step 9.3: Update mix.exs Docs

```elixir
defp docs do
  [
    main: "readme",
    extras: [
      "README.md",
      "ARCHITECTURE.md",
      "INSTRUMENTATION.md",
      "CHANGELOG.md"
    ]
  ]
end
```

---

## Phase 10: Testing & Validation

### Step 10.1: Run Quality Checks

```bash
# No warnings
mix compile --warnings-as-errors

# Dialyzer clean
mix dialyzer

# Credo strict
mix credo --strict

# All tests pass
mix test
```

### Step 10.2: Integration Testing

Test that the slimmed framework works with:
- crucible_train stages
- cns_crucible experiments
- ExFairness evaluation

### Step 10.3: Version Bump

```elixir
@version "1.0.0"
```

---

## Migration Checklist

- [ ] Phase 1: Preparation
  - [ ] Create crucible_train package
  - [ ] Add deprecation warnings
  - [ ] Add feature flags
- [ ] Phase 2: Extract Backend Infrastructure
  - [ ] Move Backend behaviour
  - [ ] Move Tinkex backend
  - [ ] Move BackendCall stage
  - [ ] Update Registry
- [ ] Phase 3: Extract Data Loading
  - [ ] Move Data.Provider
  - [ ] Move DataLoad stage
  - [ ] Simplify Context struct
- [ ] Phase 4: Extract Analysis Adapters
  - [ ] Move to cns_crucible
- [ ] Phase 5: Extract Fairness
  - [ ] Move to ExFairness
- [ ] Phase 6: Simplify Remaining Stages
  - [ ] Simplify Validate
  - [ ] Simplify Bench
- [ ] Phase 7: Update Dependencies
  - [ ] Remove tinkex dependency
  - [ ] Remove nx dependency
  - [ ] Make ensemble/hedging optional
- [ ] Phase 8: Delete Removed Code
  - [ ] Delete source files
  - [ ] Delete test files
- [ ] Phase 9: Update Documentation
  - [ ] Update README
  - [ ] Remove obsolete guides
- [ ] Phase 10: Testing & Validation
  - [ ] No warnings
  - [ ] Dialyzer clean
  - [ ] Credo strict
  - [ ] All tests pass
  - [ ] Integration tests pass

---

## Risk Mitigation

### Breaking Changes

This is a major version bump (0.4.0 -> 1.0.0) with breaking changes:
1. Removed modules require import from new packages
2. Context struct has fewer fields
3. Configuration keys changed

### Migration Support

1. Provide clear CHANGELOG with migration guide
2. Publish crucible_train v1.0.0 before crucible_framework v1.0.0
3. Ensure all destination packages are ready
4. Provide adapter examples

### Rollback Plan

If issues arise:
1. Keep v0.4.x branch available
2. Dependent packages can pin to `{:crucible_framework, "~> 0.4.0"}`
3. Document workarounds
