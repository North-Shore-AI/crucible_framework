# Implementation Prompt: Slim crucible_framework

**Date**: 2025-12-25
**Task**: Slim crucible_framework from monolithic ML engine to thin orchestration layer

---

## Required Reading

Before implementing, read these files IN ORDER:

### 1. Vision Documents

```
/home/home/p/g/North-Shore-AI/crucible_framework/docs/20251225/current_state.md
/home/home/p/g/North-Shore-AI/crucible_framework/docs/20251225/vision.md
/home/home/p/g/North-Shore-AI/crucible_framework/docs/20251225/migration_plan.md
```

### 2. Ecosystem Context

```
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251225/ecosystem_design/00_EXECUTIVE_SUMMARY.md
/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251225/ecosystem_design/05_INTEGRATION_PATTERNS.md
```

### 3. Current Source (To Understand What Exists)

**Core to KEEP**:
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/pipeline/runner.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/context.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/registry.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/trace_integration.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework/persistence.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework/persistence/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework/application.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework/repo.ex
```

**To REMOVE** (understand before deleting):
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/backend.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/backend_manager.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/backend/tinkex.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/backend/tinkex/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/data/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/data_load.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/backend_call.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/analysis/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/analysis/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/fairness/*.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/fairness.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/ir.ex
```

**Stages to SIMPLIFY**:
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/validate.ex
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/bench.ex
```

### 4. Test Files

```
/home/home/p/g/North-Shore-AI/crucible_framework/test/test_helper.exs
/home/home/p/g/North-Shore-AI/crucible_framework/test/crucible/pipeline/runner_test.exs
/home/home/p/g/North-Shore-AI/crucible_framework/test/crucible/context_test.exs
```

### 5. Configuration

```
/home/home/p/g/North-Shore-AI/crucible_framework/mix.exs
```

---

## What to REMOVE

### 1. Backend Infrastructure (DELETE ENTIRE)

```elixir
# DELETE these files:
lib/crucible/backend.ex                  # Backend behaviour
lib/crucible/backend_manager.ex          # Backend state management
lib/crucible/backend/tinkex.ex           # Tinkex implementation
lib/crucible/backend/tinkex/client.ex    # Tinkex client behaviour
lib/crucible/backend/tinkex/live_client.ex # Live client
```

**Why**: Backend/training is domain-specific. Belongs in crucible_train.

### 2. Data Loading (DELETE ENTIRE)

```elixir
# DELETE these files:
lib/crucible/data/provider.ex            # Provider behaviour
lib/crucible/data/in_memory.ex           # InMemory provider
lib/crucible/stage/data_load.ex          # DataLoad stage
```

**Why**: Data loading is training-specific. Belongs in crucible_train.

### 3. Analysis Adapters (DELETE ENTIRE)

```elixir
# DELETE these files:
lib/crucible/analysis/adapter.ex         # Analysis adapter behaviour
lib/crucible/analysis/noop.ex            # Noop adapter
lib/crucible/analysis/tda_adapter.ex     # TDA adapter behaviour
lib/crucible/analysis/tda_noop.ex        # TDA noop
lib/crucible/analysis/surrogate_adapter.ex # Surrogate adapter behaviour
lib/crucible/analysis/surrogate_noop.ex  # Surrogate noop
lib/crucible/stage/analysis/metrics.ex   # Metrics stage
lib/crucible/stage/analysis/tda_validation.ex # TDA stage
lib/crucible/stage/analysis/surrogate_validation.ex # Surrogate stage
lib/crucible/stage/analysis/filter.ex    # Filter stage
```

**Why**: CNS-specific. Belongs in cns_crucible.

### 4. Fairness (DELETE ENTIRE)

```elixir
# DELETE these files:
lib/crucible/fairness/adapter.ex         # Fairness adapter behaviour
lib/crucible/fairness/noop.ex            # Noop adapter
lib/crucible/fairness/ex_fairness_adapter.ex # ExFairness integration
lib/crucible/stage/fairness.ex           # Fairness stage
```

**Why**: Fairness-specific. Belongs in ExFairness.

### 5. Deprecated IR Aliases (DELETE)

```elixir
# DELETE this file:
lib/crucible/ir.ex                       # Deprecated aliases
```

**Why**: Users should use CrucibleIR directly.

### 6. BackendCall Stage (DELETE)

```elixir
# DELETE this file:
lib/crucible/stage/backend_call.ex       # 734 lines of training logic
```

**Why**: Training-specific. Belongs in crucible_train.

### 7. Test Files for Removed Code

```bash
# DELETE these test files:
test/crucible/backend/tinkex_test.exs
test/crucible/stage/data_load_test.exs
test/crucible/stage/backend_call_test.exs
test/crucible/ir/experiment_test.exs
test/crucible/fairness/noop_test.exs
test/crucible/stage/fairness_test.exs
```

---

## What to KEEP

### 1. Core Orchestration

```elixir
# KEEP these files (may need updates):
lib/crucible_framework.ex                # Entry point
lib/crucible/pipeline/runner.ex          # Pipeline execution
lib/crucible/context.ex                  # Runtime context (SIMPLIFY)
lib/crucible/stage.ex                    # Stage behaviour
lib/crucible/registry.ex                 # Stage lookup (SIMPLIFY)
```

### 2. Persistence (Keep as Optional)

```elixir
# KEEP these files:
lib/crucible_framework/application.ex
lib/crucible_framework/repo.ex
lib/crucible_framework/persistence.ex
lib/crucible_framework/persistence/experiment_record.ex
lib/crucible_framework/persistence/run_record.ex
lib/crucible_framework/persistence/artifact_record.ex
```

### 3. Trace Integration

```elixir
# KEEP this file:
lib/crucible/trace_integration.ex        # Trace helpers
```

### 4. Simple Stages

```elixir
# KEEP these files (may need simplification):
lib/crucible/stage/validate.ex           # SIMPLIFY
lib/crucible/stage/bench.ex              # Keep as wrapper
lib/crucible/stage/report.ex             # Keep as-is
lib/crucible/stage/data_checks.ex        # Keep as-is
lib/crucible/stage/guardrails.ex         # Keep as-is (thin)
lib/crucible/stage/guardrails/adapter.ex # Keep behaviour
lib/crucible/stage/guardrails/noop.ex    # Keep noop
```

### 5. Protocols

```elixir
# KEEP this file:
lib/crucible/protocols/jason_encoder.ex  # JSON helpers
```

---

## How to Simplify Context

### Current Context (BEFORE)

```elixir
defmodule Crucible.Context do
  @type t :: %__MODULE__{
    experiment_id: String.t(),
    run_id: String.t(),
    experiment: Experiment.t(),

    # REMOVE - Training specific
    dataset: term() | nil,
    batches: Enumerable.t() | nil,
    examples: list() | nil,
    backend_sessions: %{atom() => term()},
    backend_state: map(),

    # KEEP
    outputs: list(),
    metrics: map(),
    artifacts: map(),
    trace: term() | nil,
    telemetry_context: map(),
    assigns: map()
  }
end
```

### Target Context (AFTER)

```elixir
defmodule Crucible.Context do
  @moduledoc """
  Runtime context threaded through pipeline stages.

  Domain-specific data should be stored in `assigns`.
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

    # Extension point for domain-specific data
    assigns: map()
  }

  @enforce_keys [:experiment_id, :run_id, :experiment]
  defstruct [
    :experiment_id,
    :run_id,
    :experiment,
    outputs: [],
    metrics: %{},
    artifacts: %{},
    trace: nil,
    telemetry_context: %{},
    assigns: %{}
  ]

  # KEEP these helper functions:
  # - put_metric/3
  # - get_metric/3
  # - update_metric/3
  # - merge_metrics/2
  # - has_metric?/2
  # - add_output/2
  # - add_outputs/2
  # - put_artifact/3
  # - get_artifact/3
  # - has_artifact?/2
  # - assign/2
  # - assign/3
  # - mark_stage_complete/2
  # - stage_completed?/2
  # - completed_stages/1

  # REMOVE these helper functions:
  # - has_data?/1            (references removed fields)
  # - has_backend_session?/2 (references removed fields)
  # - get_backend_session/2  (references removed fields)
end
```

---

## How to Simplify Registry

### Current Registry (BEFORE)

```elixir
defmodule Crucible.Registry do
  # REMOVE - backend_module/1
  def backend_module(id) when is_atom(id) do
    case Application.fetch_env(:crucible_framework, :backends) do
      {:ok, map} -> Map.fetch(map, id)
      :error -> {:error, :no_backends_configured}
    end
  end

  # KEEP - stage_module/1
  def stage_module(name) when is_atom(name) do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, map} -> Map.fetch(map, name)
      :error -> {:error, :no_stage_registry}
    end
  end
end
```

### Target Registry (AFTER)

```elixir
defmodule Crucible.Registry do
  @moduledoc """
  Resolves stage modules from configuration.
  """

  @spec stage_module(atom()) :: {:ok, module()} | {:error, term()}
  def stage_module(name) when is_atom(name) do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, map} ->
        case Map.fetch(map, name) do
          {:ok, mod} -> {:ok, mod}
          :error -> {:error, {:unknown_stage, name}}
        end

      :error ->
        {:error, :no_stage_registry}
    end
  end
end
```

---

## How to Simplify Validate Stage

### Current (527 lines) -> Target (~100 lines)

```elixir
defmodule Crucible.Stage.Validate do
  @moduledoc """
  Pre-flight validation of experiment configuration.

  Validates:
  - All pipeline stages can be resolved
  - Stage modules implement Crucible.Stage behaviour

  Domain-specific validation (backends, datasets, etc.)
  should be done by domain-specific stages.
  """

  @behaviour Crucible.Stage

  require Logger
  alias Crucible.{Context, Registry}

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    strict = Map.get(opts, :strict, false)

    # Only validate stages
    case validate_stages(experiment.pipeline) do
      {:ok, _} ->
        Logger.info("Experiment validation passed")
        ctx = Context.put_metric(ctx, :validation, %{status: :passed})
        {:ok, ctx}

      {:error, errors} when strict ->
        Logger.error("Experiment validation failed: #{inspect(errors)}")
        {:error, {:validation_failed, errors}}

      {:error, warnings} ->
        Logger.warning("Experiment validation warnings: #{inspect(warnings)}")
        ctx = Context.put_metric(ctx, :validation, %{status: :warnings, warnings: warnings})
        {:ok, ctx}
    end
  end

  @impl true
  def describe(_opts) do
    %{stage: :validate, description: "Pre-flight validation of pipeline stages"}
  end

  defp validate_stages(pipeline) when is_list(pipeline) do
    errors =
      pipeline
      |> Enum.map(&validate_single_stage/1)
      |> Enum.reject(&is_nil/1)

    if errors == [], do: {:ok, []}, else: {:error, errors}
  end

  defp validate_single_stage(%{module: mod}) when not is_nil(mod) do
    cond do
      not Code.ensure_loaded?(mod) ->
        "Stage module #{inspect(mod)} cannot be loaded"

      not function_exported?(mod, :run, 2) ->
        "Stage module #{inspect(mod)} does not implement run/2"

      true ->
        nil
    end
  end

  defp validate_single_stage(%{name: name}) do
    case Registry.stage_module(name) do
      {:ok, mod} ->
        if Code.ensure_loaded?(mod), do: nil, else: "Stage :#{name} module cannot be loaded"

      {:error, _} ->
        "Stage :#{name} is not registered"
    end
  end
end
```

---

## How to Update mix.exs

### Current (BEFORE)

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.1"},
    {:crucible_ensemble, path: "../crucible_ensemble"},
    {:crucible_hedging, path: "../crucible_hedging"},
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_trace, path: "../crucible_trace"},
    {:ex_fairness, path: "../ExFairness", optional: true},
    {:tinkex, "~> 0.1.12"},
    {:ecto_sql, "~> 3.11"},
    {:postgrex, ">= 0.0.0"},
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},
    {:nx, "~> 0.7"},
    {:mox, "~> 1.1", only: :test},
    {:stream_data, "~> 1.0", only: [:dev, :test]},
    {:ex_doc, "~> 0.38", only: :dev, runtime: false},
    {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
  ]
end
```

### Target (AFTER)

```elixir
defp deps do
  [
    # Core IR
    {:crucible_ir, "~> 0.2.0"},

    # Reliability (for built-in stages)
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_trace, path: "../crucible_trace"},

    # Optional persistence
    {:ecto_sql, "~> 3.11", optional: true},
    {:postgrex, ">= 0.0.0", optional: true},

    # Core utilities
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},

    # Testing
    {:mox, "~> 1.1", only: :test},
    {:ex_doc, "~> 0.38", only: :dev, runtime: false},
    {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
  ]
end
```

---

## TDD Approach with Mox

### Step 1: Define Mock Behaviours

For any behaviour that remains (e.g., Stage), ensure Mox is set up:

```elixir
# test/support/mocks.ex
Mox.defmock(Crucible.StageMock, for: Crucible.Stage)
```

### Step 2: Write Tests FIRST for Simplified Context

```elixir
# test/crucible/context_test.exs
defmodule Crucible.ContextTest do
  use ExUnit.Case

  alias Crucible.Context

  describe "struct" do
    test "creates with required fields" do
      ctx = %Context{
        experiment_id: "exp1",
        run_id: "run1",
        experiment: %CrucibleIR.Experiment{id: "exp1"}
      }

      assert ctx.experiment_id == "exp1"
      assert ctx.outputs == []
      assert ctx.metrics == %{}
      assert ctx.assigns == %{}
    end

    test "does not have dataset, batches, or backend fields" do
      fields = Context.__struct__() |> Map.keys()

      refute :dataset in fields
      refute :batches in fields
      refute :examples in fields
      refute :backend_sessions in fields
      refute :backend_state in fields
    end
  end

  describe "assigns" do
    test "stages can store domain data in assigns" do
      ctx = %Context{experiment_id: "exp1", run_id: "run1", experiment: %{id: "exp1"}}

      # Training stage stores its data
      ctx = Context.assign(ctx, :dataset, [%{input: "a", output: "b"}])
      ctx = Context.assign(ctx, :batches, [[%{input: "a"}]])

      assert ctx.assigns[:dataset] == [%{input: "a", output: "b"}]
      assert ctx.assigns[:batches] == [[%{input: "a"}]]
    end
  end
end
```

### Step 3: Write Tests for Pipeline Runner

```elixir
# test/crucible/pipeline/runner_test.exs
defmodule Crucible.Pipeline.RunnerTest do
  use ExUnit.Case

  import Mox

  alias Crucible.Pipeline.Runner
  alias Crucible.Context

  setup :verify_on_exit!

  describe "run/2" do
    test "runs stages in sequence" do
      experiment = %CrucibleIR.Experiment{
        id: "test",
        pipeline: [
          %CrucibleIR.StageDef{name: :step1, module: Crucible.StageMock},
          %CrucibleIR.StageDef{name: :step2, module: Crucible.StageMock}
        ]
      }

      Crucible.StageMock
      |> expect(:run, fn ctx, _opts ->
        {:ok, Context.put_metric(ctx, :step1, true)}
      end)
      |> expect(:run, fn ctx, _opts ->
        {:ok, Context.put_metric(ctx, :step2, true)}
      end)

      assert {:ok, ctx} = Runner.run(experiment, persist: false)
      assert ctx.metrics[:step1] == true
      assert ctx.metrics[:step2] == true
    end

    test "stops on error" do
      experiment = %CrucibleIR.Experiment{
        id: "test",
        pipeline: [
          %CrucibleIR.StageDef{name: :fail, module: Crucible.StageMock}
        ]
      }

      Crucible.StageMock
      |> expect(:run, fn _ctx, _opts ->
        {:error, :something_went_wrong}
      end)

      assert {:error, {:fail, :something_went_wrong, _}} = Runner.run(experiment, persist: false)
    end
  end
end
```

---

## Quality Requirements

### 1. No Compiler Warnings

```bash
mix compile --warnings-as-errors
```

All code must compile without warnings.

### 2. Dialyzer Clean

```bash
mix dialyzer
```

No dialyzer errors or warnings.

### 3. Credo Strict

```bash
mix credo --strict
```

No credo issues at strict level.

### 4. All Tests Passing

```bash
mix test
```

100% of tests must pass.

### 5. Documentation Complete

All public modules and functions must have `@moduledoc` and `@doc`.

---

## Update README.md

After completing the slimming, update the README to reflect:

1. **New Purpose**: Thin orchestration layer, not ML engine
2. **What's Included**: Pipeline.Runner, Stage behaviour, Context, optional Persistence
3. **What's NOT Included**: Backends, data loading, training stages
4. **How to Use**: Show simple pipeline example
5. **Related Packages**: Point to crucible_train for training, cns_crucible for CNS, etc.

### Example Updated README Sections

```markdown
# CrucibleFramework

A thin orchestration layer for running experiment pipelines.

## What This Package Provides

- `Crucible.Pipeline.Runner` - Executes stage pipelines
- `Crucible.Stage` - Behaviour for pipeline stages
- `Crucible.Context` - Runtime context with helpers
- `CrucibleFramework.Persistence` - Optional experiment/run persistence

## What This Package Does NOT Provide

- Training infrastructure (see crucible_train)
- Dataset loading (see crucible_datasets)
- Backend implementations (see tinkex_cookbook)
- Domain-specific stages (see cns_crucible, ExFairness)

## Quick Example

```elixir
# Define a simple stage
defmodule MyStage do
  @behaviour Crucible.Stage

  def run(ctx, opts) do
    # Do work, update context
    {:ok, Crucible.Context.put_metric(ctx, :my_metric, 42)}
  end
end

# Create experiment
experiment = %CrucibleIR.Experiment{
  id: "my-experiment",
  pipeline: [
    %CrucibleIR.StageDef{name: :my_stage, module: MyStage}
  ]
}

# Run
{:ok, ctx} = CrucibleFramework.run(experiment)
```
```

---

## Implementation Sequence

1. **First**: Read all required files
2. **Second**: Write tests for simplified Context
3. **Third**: Simplify Context struct (remove fields)
4. **Fourth**: Update Context helper functions
5. **Fifth**: Delete backend infrastructure files
6. **Sixth**: Delete data loading files
7. **Seventh**: Delete analysis adapter files
8. **Eighth**: Delete fairness files
9. **Ninth**: Delete IR aliases
10. **Tenth**: Simplify Registry (remove backend_module)
11. **Eleventh**: Simplify Validate stage
12. **Twelfth**: Update mix.exs dependencies
13. **Thirteenth**: Delete obsolete test files
14. **Fourteenth**: Run quality checks
15. **Fifteenth**: Update README.md
16. **Sixteenth**: Final test run

---

## Verification Commands

After each major change, run:

```bash
# Compile check
mix compile --warnings-as-errors

# Test check
mix test

# After all changes complete:
mix dialyzer
mix credo --strict
```

---

## Success Criteria

- [ ] No references to `Crucible.Backend` in remaining code
- [ ] No references to `Tinkex` in remaining code
- [ ] No references to `Nx` in remaining code
- [ ] Context struct has only 8 fields (was 12)
- [ ] Total LOC < 1,500 (was ~4,750)
- [ ] mix compile --warnings-as-errors passes
- [ ] mix dialyzer passes
- [ ] mix credo --strict passes
- [ ] mix test passes
- [ ] README reflects new purpose
