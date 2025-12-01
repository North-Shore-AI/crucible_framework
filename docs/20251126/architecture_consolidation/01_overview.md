# Crucible Architecture Consolidation - Overview

**Date:** 2025-11-26
**Status:** Design Proposal
**Author:** Architecture Review

---

## Executive Summary

This document outlines a comprehensive plan to consolidate the Crucible ecosystem architecture by:

1. **Extracting the IR** to a new `crucible_ir` library
2. **Refactoring crucible_framework** to delegate to specialized libraries
3. **Standardizing integration patterns** across all libraries
4. **Reducing duplication** while maintaining clean separation of concerns

## Current State Analysis

### Library Inventory (14 Active Projects)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| **crucible_framework** | 0.4.0 | Hub/orchestration | Source of IR |
| **crucible_ensemble** | 0.2.0 | Multi-model voting | Standalone |
| **crucible_hedging** | 0.2.0 | Request hedging | Standalone |
| **crucible_bench** | 0.2.1 | Statistical testing | Standalone |
| **crucible_trace** | 0.2.0 | Causal tracing | Standalone |
| **crucible_telemetry** | 0.2.0 | Instrumentation | Standalone |
| **crucible_datasets** | 0.2.0 | Dataset management | Standalone |
| **crucible_adversary** | 0.3.0 | Adversarial testing | Standalone |
| **crucible_xai** | 0.3.0 | Explainability | Standalone |
| **crucible_harness** | 0.2.0 | Experiment DSL | Higher-level |
| **crucible_examples** | - | Demonstrations | Phoenix app |
| **crucible_ui** | - | Dashboard | Phoenix app |
| **ExDataCheck** | 0.2.1 | Data validation | Standalone |
| **ExFairness** | 0.3.0 | Fairness metrics | Standalone |
| **LlmGuard** | 0.2.1 | Security/guardrails | Standalone |

### Current IR Location

The IR lives exclusively in `crucible_framework/lib/crucible/ir/`:

```
crucible/ir/
├── experiment.ex           # Top-level experiment definition
├── dataset_ref.ex          # Dataset reference
├── backend_ref.ex          # Backend reference
├── stage_def.ex            # Pipeline stage definition
├── reliability_config.ex   # Container for reliability configs
├── ensemble_config.ex      # Ensemble configuration
├── hedging_config.ex       # Hedging configuration
├── guardrail_config.ex     # Guardrail configuration
├── stats_config.ex         # Stats configuration
├── fairness_config.ex      # Fairness configuration
└── output_spec.ex          # Output specification
```

### Key Problems

1. **IR Coupling**: Libraries wanting to use IR must depend on entire crucible_framework
2. **Duplication**: Some libraries redefine similar config structures
3. **Adapter Overhead**: crucible_framework stages wrap library calls with extra indirection
4. **Inconsistent Integration**: Different libraries have different integration patterns

## Proposed Architecture

```
                    ┌───────────────────────────┐
                    │      crucible_ir          │
                    │  (Pure data structures)   │
                    │                           │
                    │  - Experiment             │
                    │  - ReliabilityConfig      │
                    │  - EnsembleConfig         │
                    │  - HedgingConfig          │
                    │  - StatsConfig            │
                    │  - FairnessConfig         │
                    │  - GuardrailConfig        │
                    │  - DatasetRef             │
                    │  - BackendRef             │
                    │  - StageDef               │
                    │  - OutputSpec             │
                    └───────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│crucible_bench │       │crucible_ensemble│     │crucible_hedging│
│               │       │               │       │               │
│Uses: StatsConfig│     │Uses: EnsembleConfig│  │Uses: HedgingConfig│
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │   crucible_framework      │
                    │                           │
                    │  - Pipeline.Runner        │
                    │  - Context                │
                    │  - Stage behaviour        │
                    │  - Backend behaviour      │
                    │  - Built-in stages        │
                    │  - Persistence            │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │    crucible_harness       │
                    │  (High-level DSL)         │
                    └───────────────────────────┘
```

## Design Documents

| Document | Purpose |
|----------|---------|
| [02_crucible_ir_extraction.md](02_crucible_ir_extraction.md) | Detailed IR extraction plan |
| [03_library_integration_patterns.md](03_library_integration_patterns.md) | Standard integration patterns |
| [04_crucible_framework_consolidation.md](04_crucible_framework_consolidation.md) | Framework refactoring plan |
| [05_migration_strategy.md](05_migration_strategy.md) | Step-by-step migration |
| [06_dependency_graph.md](06_dependency_graph.md) | Visual dependency relationships |

## Key Benefits

### 1. Cleaner Dependencies
- Libraries only depend on what they need
- No circular dependency risk
- Smaller dependency trees

### 2. Better Modularity
- Each library owns its domain logic
- IR is shared vocabulary, not implementation
- Clear boundaries between concerns

### 3. Easier Maintenance
- Changes to IR are isolated
- Library upgrades don't cascade
- Testing is more focused

### 4. Improved Extensibility
- New libraries can use IR immediately
- Custom stages can use proper types
- Third-party integrations are cleaner

## Decision Summary

| Decision | Rationale |
|----------|-----------|
| Extract IR to `crucible_ir` | Eliminates framework dependency for basic types |
| Keep framework as orchestration hub | Maintains clear pipeline execution ownership |
| Libraries consume IR configs directly | Reduces adapter complexity |
| Framework delegates to libraries | Avoids reimplementation in stages |
| Harness uses framework | Maintains high-level abstraction |

## Timeline Estimate

| Phase | Scope | Effort |
|-------|-------|--------|
| 1. Create crucible_ir | Extract structs, no behavior changes | 1-2 days |
| 2. Update libraries | Add crucible_ir dependency | 2-3 days |
| 3. Refactor framework | Remove IR, add lib dependencies | 2-3 days |
| 4. Update harness | Use new integration patterns | 1 day |
| 5. Test & validate | End-to-end verification | 1-2 days |

**Total: ~1-2 weeks**

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing integrations | Maintain backwards-compatible aliases |
| Version coordination | Semantic versioning, changelog communication |
| Missing edge cases | Comprehensive integration tests |
| Performance regression | Benchmark critical paths |

## Next Steps

1. Review this proposal with stakeholders
2. Finalize IR struct list for extraction
3. Create crucible_ir repository
4. Begin phased migration
