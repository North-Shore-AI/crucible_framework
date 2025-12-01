# Migration Strategy

**Date:** 2025-11-26
**Status:** Design Proposal

---

## Overview

This document outlines the step-by-step migration plan for consolidating the Crucible architecture. The migration is designed to be:

- **Incremental** - Can be done in phases
- **Reversible** - Easy to roll back if issues arise
- **Non-breaking** - Maintains backwards compatibility

## Migration Phases

```
Phase 1: Create crucible_ir (Day 1-2)
    │
    ▼
Phase 2: Update Libraries to Use IR (Day 3-5)
    │
    ▼
Phase 3: Refactor crucible_framework (Day 6-8)
    │
    ▼
Phase 4: Update Dependent Projects (Day 9-10)
    │
    ▼
Phase 5: Cleanup and Release (Day 11-12)
```

---

## Phase 1: Create crucible_ir

**Duration:** 1-2 days
**Risk:** Low
**Dependencies:** None

### Tasks

#### 1.1 Create Repository

```bash
# Create new repo in North-Shore-AI organization
gh repo create North-Shore-AI/crucible_ir --public --description "Intermediate Representation for Crucible ML ecosystem"

# Clone and initialize
git clone git@github.com:North-Shore-AI/crucible_ir.git
cd crucible_ir
mix new crucible_ir
```

#### 1.2 Copy and Adapt IR Structs

```bash
# Copy from crucible_framework
cp -r ../crucible_framework/lib/crucible/ir/* lib/crucible_ir/

# Rename modules (Crucible.IR.* -> CrucibleIR.*)
# Update all internal references
```

#### 1.3 Add Helper Functions

```elixir
# lib/crucible_ir.ex
defmodule CrucibleIR do
  def validate(struct), do: CrucibleIR.Validation.validate(struct)
  def to_json(struct), do: Jason.encode(struct)
  def from_json(json, type), do: CrucibleIR.Serialization.decode(json, type)
end
```

#### 1.4 Write Tests

```elixir
# test/crucible_ir/experiment_test.exs
defmodule CrucibleIR.ExperimentTest do
  use ExUnit.Case

  alias CrucibleIR.{Experiment, BackendRef, StageDef}

  test "creates experiment with required fields" do
    exp = %Experiment{
      id: "test",
      backend: %BackendRef{id: :mock},
      pipeline: [%StageDef{name: :test}]
    }
    assert exp.id == "test"
  end

  test "serializes to JSON" do
    exp = %Experiment{
      id: "test",
      backend: %BackendRef{id: :mock},
      pipeline: []
    }
    assert {:ok, json} = Jason.encode(exp)
    assert String.contains?(json, "test")
  end
end
```

#### 1.5 Publish to Hex

```bash
# Update mix.exs with package info
mix hex.publish
```

### Verification

- [ ] All structs compile without warnings
- [ ] All tests pass
- [ ] Documentation generates correctly
- [ ] Package published to Hex

---

## Phase 2: Update Libraries to Use IR

**Duration:** 2-3 days
**Risk:** Medium
**Dependencies:** Phase 1 complete

### Order of Updates

1. **crucible_bench** (most used)
2. **crucible_ensemble**
3. **crucible_hedging**
4. **crucible_telemetry**
5. **crucible_trace**
6. **crucible_datasets**
7. **crucible_adversary**
8. **crucible_xai**
9. **ExFairness**
10. **LlmGuard**
11. **ExDataCheck**

### For Each Library

#### 2.1 Add Dependency

```elixir
# mix.exs
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    # ... existing deps
  ]
end
```

#### 2.2 Create Stage Implementation (if applicable)

```elixir
# lib/crucible_bench/stage.ex
defmodule CrucibleBench.Stage do
  @behaviour Crucible.Stage

  alias Crucible.Context
  alias CrucibleIR.Reliability.Stats

  @impl true
  def run(%Context{} = ctx, opts) do
    config = ctx.experiment.reliability.stats
    # ... implementation using config
  end

  @impl true
  def describe(_opts) do
    %{stage: :bench, description: "Statistical benchmarking"}
  end
end
```

#### 2.3 Add Config Consumers

```elixir
# lib/crucible_bench.ex
defmodule CrucibleBench do
  alias CrucibleIR.Reliability.Stats

  @doc """
  Run analysis with Stats configuration.
  """
  def analyze(data, %Stats{} = config) do
    # Use config.tests, config.alpha, etc.
  end

  # Backwards compatibility
  def analyze(data, opts) when is_list(opts) do
    analyze(data, struct(Stats, opts))
  end
end
```

#### 2.4 Update Tests

```elixir
defmodule CrucibleBenchTest do
  alias CrucibleIR.Reliability.Stats

  test "accepts Stats config" do
    config = %Stats{tests: [:ttest], alpha: 0.05}
    assert {:ok, _} = CrucibleBench.analyze([1,2,3], config)
  end
end
```

#### 2.5 Release Minor Version

```bash
# Bump version in mix.exs (e.g., 0.2.0 -> 0.2.1)
mix hex.publish
```

### Library-Specific Notes

#### crucible_ensemble

```elixir
# Uses EnsembleConfig for voting strategies
alias CrucibleIR.Reliability.Ensemble

def vote(responses, %Ensemble{strategy: :majority} = config) do
  Voting.majority(responses, config.min_agreement)
end
```

#### crucible_hedging

```elixir
# Uses HedgingConfig for strategy selection
alias CrucibleIR.Reliability.Hedging

def create_hedger(%Hedging{strategy: :fixed} = config) do
  %Hedger{delay: config.delay_ms, max: config.max_hedges}
end
```

#### ExFairness

```elixir
# Uses FairnessConfig for metric selection
alias CrucibleIR.Reliability.Fairness

def evaluate(data, %Fairness{} = config) do
  Enum.map(config.metrics, &run_metric(&1, data, config.threshold))
end
```

#### LlmGuard

```elixir
# Uses GuardrailConfig for detection settings
alias CrucibleIR.Reliability.Guardrail

def configure(%Guardrail{} = config) do
  LlmGuard.Config.new(
    prompt_injection_detection: config.prompt_injection_detection,
    jailbreak_detection: config.jailbreak_detection
  )
end
```

### Verification

- [ ] Each library compiles with crucible_ir dependency
- [ ] All existing tests still pass
- [ ] New IR-based APIs work correctly
- [ ] Backwards compatibility maintained

---

## Phase 3: Refactor crucible_framework

**Duration:** 2-3 days
**Risk:** Medium-High
**Dependencies:** Phase 2 complete

### Tasks

#### 3.1 Add crucible_ir Dependency

```elixir
# mix.exs
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    # Remove local IR files
  ]
end
```

#### 3.2 Create Backwards-Compatible Aliases

```elixir
# lib/crucible/ir.ex
defmodule Crucible.IR do
  @moduledoc """
  Backwards-compatible module for IR access.

  DEPRECATED: Use CrucibleIR directly.
  """

  defmodule Experiment do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Experiment
    defdelegate __struct__(kv), to: CrucibleIR.Experiment
  end

  defmodule ReliabilityConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Config
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Config
  end

  # ... similar for other modules
end
```

#### 3.3 Update Internal Imports

```bash
# Find and replace
grep -r "Crucible.IR." lib/ --include="*.ex" -l | \
  xargs sed -i 's/Crucible\.IR\./CrucibleIR./g'

# Manual review for edge cases
```

#### 3.4 Simplify Stage Implementations

```elixir
# lib/crucible/stage/bench.ex
defmodule Crucible.Stage.Bench do
  @behaviour Crucible.Stage

  @impl true
  def run(ctx, opts) do
    stage_module = Application.get_env(
      :crucible_framework,
      :bench_stage,
      CrucibleBench.Stage
    )
    stage_module.run(ctx, opts)
  end

  @impl true
  def describe(opts) do
    %{stage: :bench, description: "Statistical benchmarking"}
  end
end
```

#### 3.5 Remove Old IR Directory

```bash
rm -rf lib/crucible/ir/
git add -A
git commit -m "Remove local IR, use crucible_ir dependency"
```

#### 3.6 Update Configuration

```elixir
# config/config.exs
config :crucible_framework,
  # Stage implementations from libraries
  bench_stage: CrucibleBench.Stage,
  fairness_stage: ExFairness.Stage,
  guardrails_stage: LlmGuard.Stage,
  ensemble_stage: CrucibleEnsemble.Stage,
  hedging_stage: CrucibleHedging.Stage
```

### Verification

- [ ] All tests pass
- [ ] No compilation warnings
- [ ] Examples still work
- [ ] Integration tests pass
- [ ] Backwards compatibility verified

---

## Phase 4: Update Dependent Projects

**Duration:** 1-2 days
**Risk:** Low
**Dependencies:** Phase 3 complete

### Projects to Update

1. **crucible_harness**
2. **crucible_examples**
3. **crucible_ui**
4. **cns_crucible** (if still relevant)

### For Each Project

#### 4.1 Update Dependencies

```elixir
# mix.exs
defp deps do
  [
    {:crucible_framework, "~> 0.5.0"},
    {:crucible_ir, "~> 0.1.0"},  # May be transitive
    # ...
  ]
end
```

#### 4.2 Update Imports

```elixir
# Before
alias Crucible.IR.{Experiment, BackendRef}

# After
alias CrucibleIR.{Experiment, BackendRef}
```

#### 4.3 Test and Release

```bash
mix deps.get
mix test
mix hex.publish  # if applicable
```

### Verification

- [ ] crucible_harness works with new framework
- [ ] crucible_examples run successfully
- [ ] crucible_ui compiles and starts
- [ ] End-to-end experiments execute correctly

---

## Phase 5: Cleanup and Release

**Duration:** 1-2 days
**Risk:** Low
**Dependencies:** Phase 4 complete

### Tasks

#### 5.1 Update Documentation

- [ ] Update README files
- [ ] Update API documentation
- [ ] Add migration guide
- [ ] Update architecture diagrams

#### 5.2 Deprecation Warnings

```elixir
# Add to Crucible.IR module
@deprecated "Use CrucibleIR.Experiment instead"
def experiment, do: CrucibleIR.Experiment
```

#### 5.3 Version Bumps

| Package | Old | New |
|---------|-----|-----|
| crucible_ir | - | 0.1.0 |
| crucible_bench | 0.2.1 | 0.3.0 |
| crucible_ensemble | 0.2.0 | 0.3.0 |
| crucible_hedging | 0.2.0 | 0.3.0 |
| crucible_framework | 0.4.0 | 0.5.0 |
| crucible_harness | 0.2.0 | 0.3.0 |

#### 5.4 Release Announcements

- [ ] CHANGELOG updates
- [ ] GitHub releases
- [ ] Documentation updates

---

## Rollback Plan

### If Issues in Phase 1-2

Simply don't release updated libraries. No impact on existing code.

### If Issues in Phase 3

```bash
# Revert framework changes
git revert HEAD~N  # N = number of commits

# Re-release previous version
mix hex.publish --revert 0.5.0
mix hex.publish  # 0.4.1 with fixes
```

### If Issues in Phase 4-5

```bash
# Pin dependencies to previous versions
{:crucible_framework, "~> 0.4.0"},  # Not 0.5.0

# Wait for fixes before upgrading
```

---

## Success Criteria

### Technical

- [ ] All tests pass in all repositories
- [ ] No compilation warnings
- [ ] Zero breaking changes for external users
- [ ] Documentation is complete and accurate

### Performance

- [ ] No performance regression (benchmark suite)
- [ ] Memory usage unchanged or improved
- [ ] Startup time unchanged

### Usability

- [ ] Migration guide is clear
- [ ] Deprecation warnings are helpful
- [ ] Error messages are informative

---

## Timeline Summary

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| 1. Create crucible_ir | 2 days | Day 1 | Day 2 |
| 2. Update libraries | 3 days | Day 3 | Day 5 |
| 3. Refactor framework | 3 days | Day 6 | Day 8 |
| 4. Update dependents | 2 days | Day 9 | Day 10 |
| 5. Cleanup & release | 2 days | Day 11 | Day 12 |

**Total: ~12 working days (2-3 weeks)**
