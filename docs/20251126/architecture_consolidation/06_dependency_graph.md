# Dependency Graph

**Date:** 2025-11-26
**Status:** Design Proposal

---

## Current Dependency Structure

```
                                    ┌─────────────────────┐
                                    │   Applications      │
                                    │                     │
                                    │  • cns_crucible     │
                                    │  • crucible_ui      │
                                    │  • crucible_examples│
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  crucible_harness   │
                                    │  (Experiment DSL)   │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
         ┌─────────────────┐       ┌─────────────────────┐    ┌─────────────────┐
         │crucible_framework│◄──────│    Contains IR      │    │ Other Crucible  │
         │                 │       │  (Experiment,       │    │   Libraries     │
         │ • Context       │       │   ReliabilityConfig,│    │                 │
         │ • Pipeline      │       │   etc.)             │    │ NO DEPENDENCY   │
         │ • Stages        │       └─────────────────────┘    │ ON FRAMEWORK    │
         │ • Backends      │                                  │                 │
         └────────┬────────┘                                  └─────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┬─────────────┐
    │             │             │             │             │
    ▼             ▼             ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│crucible│  │crucible│  │crucible│  │crucible│  │crucible│
│_bench  │  │_ensemble│ │_hedging│  │_trace  │  │_telemetry│
└────────┘  └────────┘  └────────┘  └────────┘  └────────┘
    │             │             │
    │   (Adapters/Bridges)      │
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐
│ExFairness│ │LlmGuard│  │ExDataCheck│
└────────┘  └────────┘  └────────┘
```

### Problems with Current Structure

1. **IR locked in framework** - Libraries can't use IR types without depending on entire framework
2. **No direct library dependencies** - Libraries don't know about each other
3. **Adapter complexity** - Framework stages wrap library calls
4. **Circular risk** - If libraries want IR, they'd depend on framework which may depend on them

---

## Proposed Dependency Structure

```
                                    ┌─────────────────────┐
                                    │   Applications      │
                                    │                     │
                                    │  • crucible_ui      │
                                    │  • crucible_examples│
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  crucible_harness   │
                                    │  (Experiment DSL)   │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │ crucible_framework  │
                                    │                     │
                                    │  • Context          │
                                    │  • Pipeline.Runner  │
                                    │  • Stage behaviour  │
                                    │  • Backend behaviour│
                                    │  • Persistence      │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
         ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
         │ Reliability     │       │ Analysis        │       │ Security        │
         │ Libraries       │       │ Libraries       │       │ Libraries       │
         │                 │       │                 │       │                 │
         │ • crucible_bench│       │ • crucible_xai  │       │ • LlmGuard      │
         │ • crucible_     │       │ • crucible_     │       │ • crucible_     │
         │   ensemble      │       │   adversary     │       │   adversary     │
         │ • crucible_     │       │ • ExDataCheck   │       │                 │
         │   hedging       │       │ • ExFairness    │       │                 │
         └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
                  │                         │                         │
                  └─────────────────────────┼─────────────────────────┘
                                            │
                                            ▼
                                 ┌─────────────────────┐
                                 │    crucible_ir      │
                                 │                     │
                                 │  • Experiment       │
                                 │  • ReliabilityConfig│
                                 │  • EnsembleConfig   │
                                 │  • HedgingConfig    │
                                 │  • StatsConfig      │
                                 │  • FairnessConfig   │
                                 │  • GuardrailConfig  │
                                 │  • DatasetRef       │
                                 │  • BackendRef       │
                                 │  • StageDef         │
                                 │  • OutputSpec       │
                                 └─────────────────────┘
                                            │
                                            ▼
                                 ┌─────────────────────┐
                                 │   Core Libraries    │
                                 │                     │
                                 │  • jason            │
                                 │  • (no other deps)  │
                                 └─────────────────────┘
```

---

## Detailed Dependency Matrix

### crucible_ir (Base Layer)

```elixir
# mix.exs
defp deps do
  [
    {:jason, "~> 1.4"}  # Only dependency
  ]
end
```

**Depends on:** jason
**Depended on by:** All Crucible libraries

### Reliability Libraries

#### crucible_bench

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:nx, "~> 0.7", optional: true},
    {:statistex, "~> 1.0"}
  ]
end
```

#### crucible_ensemble

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:telemetry, "~> 1.2"}
  ]
end
```

#### crucible_hedging

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:telemetry, "~> 1.2"}
  ]
end
```

### Analysis Libraries

#### crucible_xai

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:nx, "~> 0.7"},
    {:scholar, "~> 0.3"}
  ]
end
```

#### crucible_adversary

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:telemetry, "~> 1.2"}
  ]
end
```

#### ExFairness

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:nx, "~> 0.7"},
    {:telemetry, "~> 1.2"}
  ]
end
```

#### ExDataCheck

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},  # Optional, for integration
    {:nx, "~> 0.7", optional: true},
    {:telemetry, "~> 1.2"}
  ]
end
```

### Security Libraries

#### LlmGuard

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},  # Optional, for GuardrailConfig
    {:telemetry, "~> 1.2"}
  ]
end
```

### Instrumentation Libraries

#### crucible_telemetry

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:telemetry, "~> 1.2"},
    {:telemetry_metrics, "~> 0.6"},
    {:telemetry_poller, "~> 1.0"}
  ]
end
```

#### crucible_trace

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:jason, "~> 1.4"}
  ]
end
```

### Data Libraries

#### crucible_datasets

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:req, "~> 0.4"},
    {:jason, "~> 1.4"}
  ]
end
```

### Orchestration Layer

#### crucible_framework

```elixir
defp deps do
  [
    # Core
    {:crucible_ir, "~> 0.1.0"},
    {:jason, "~> 1.4"},

    # Persistence
    {:ecto_sql, "~> 3.10"},
    {:postgrex, "~> 0.17"},

    # Instrumentation (required)
    {:crucible_telemetry, "~> 0.2.0"},
    {:crucible_trace, "~> 0.2.0"},

    # Reliability (required)
    {:crucible_bench, "~> 0.3.0"},

    # Optional integrations
    {:crucible_ensemble, "~> 0.3.0", optional: true},
    {:crucible_hedging, "~> 0.3.0", optional: true},
    {:crucible_adversary, "~> 0.3.0", optional: true},
    {:crucible_xai, "~> 0.3.0", optional: true},
    {:crucible_datasets, "~> 0.2.0", optional: true},
    {:ex_fairness, "~> 0.3.0", optional: true},
    {:llm_guard, "~> 0.2.0", optional: true},
    {:ex_data_check, "~> 0.2.0", optional: true}
  ]
end
```

#### crucible_harness

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:crucible_framework, "~> 0.5.0"},
    {:flow, "~> 1.2"}
  ]
end
```

### Application Layer

#### crucible_ui

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:crucible_framework, "~> 0.5.0"},
    {:crucible_telemetry, "~> 0.2.0"},
    {:phoenix, "~> 1.7"},
    {:phoenix_live_view, "~> 0.20"}
  ]
end
```

#### crucible_examples

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.0"},
    {:crucible_framework, "~> 0.5.0"},
    {:crucible_bench, "~> 0.3.0"},
    {:crucible_ensemble, "~> 0.3.0"},
    {:crucible_hedging, "~> 0.3.0"},
    {:phoenix, "~> 1.7"},
    {:phoenix_live_view, "~> 0.20"}
  ]
end
```

---

## Dependency Graph (ASCII)

```
                           crucible_ui
                               │
                               ▼
                        crucible_harness
                               │
                               ▼
┌──────────────────────crucible_framework──────────────────────┐
│                              │                               │
│  ┌───────────┬───────────┬───┴───┬───────────┬───────────┐  │
│  │           │           │       │           │           │  │
│  ▼           ▼           ▼       ▼           ▼           ▼  │
│crucible   crucible   crucible crucible   crucible    crucible│
│ _bench    _ensemble  _hedging _trace    _telemetry  _datasets│
│  │           │           │       │           │           │  │
│  └───────────┴───────────┴───┬───┴───────────┴───────────┘  │
│                              │                               │
│              ┌───────────────┼───────────────┐              │
│              │               │               │              │
│              ▼               ▼               ▼              │
│         ExFairness       LlmGuard      ExDataCheck         │
│              │               │               │              │
│              └───────────────┼───────────────┘              │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
                         crucible_ir
                               │
                               ▼
                            jason
```

---

## Package Sizes (Estimated)

| Package | Files | LOC | Dependencies |
|---------|-------|-----|--------------|
| crucible_ir | ~15 | ~800 | 1 (jason) |
| crucible_bench | ~20 | ~2,000 | 3 |
| crucible_ensemble | ~15 | ~1,500 | 2 |
| crucible_hedging | ~15 | ~1,200 | 2 |
| crucible_trace | ~10 | ~800 | 2 |
| crucible_telemetry | ~15 | ~1,000 | 4 |
| crucible_datasets | ~10 | ~600 | 3 |
| crucible_adversary | ~25 | ~3,000 | 2 |
| crucible_xai | ~20 | ~2,500 | 3 |
| ExFairness | ~15 | ~1,800 | 3 |
| LlmGuard | ~20 | ~2,200 | 2 |
| ExDataCheck | ~25 | ~3,500 | 3 |
| crucible_framework | ~40 | ~4,000 | 15 |
| crucible_harness | ~15 | ~1,500 | 3 |

---

## Benefits of New Structure

### 1. Minimal Base Dependency

```
crucible_ir (1 dep) vs crucible_framework (15+ deps)
```

Libraries only need to depend on `crucible_ir` to use shared types.

### 2. Clear Layering

```
Applications → Harness → Framework → Libraries → IR → Core
```

Each layer only depends on layers below it.

### 3. Optional Features

Framework can work with minimal libraries, adding features as needed:

```elixir
# Minimal setup
{:crucible_framework, "~> 0.5.0"}

# With ensemble support
{:crucible_framework, "~> 0.5.0"},
{:crucible_ensemble, "~> 0.3.0"}

# Full reliability suite
{:crucible_framework, "~> 0.5.0"},
{:crucible_bench, "~> 0.3.0"},
{:crucible_ensemble, "~> 0.3.0"},
{:crucible_hedging, "~> 0.3.0"},
{:ex_fairness, "~> 0.3.0"},
{:llm_guard, "~> 0.2.0"}
```

### 4. Independent Development

Libraries can be developed, tested, and released independently without touching framework code.

### 5. Reduced Coupling

Changes to one library don't cascade through the system (except IR changes, which should be rare and additive).
