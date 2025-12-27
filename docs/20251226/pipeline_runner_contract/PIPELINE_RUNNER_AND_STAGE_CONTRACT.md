# Pipeline Runner and Stage Contract

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Clarify where the pipeline runner lives and define a consistent contract for stage implementations across the ecosystem.

## 2) Runner Location (Authoritative)

- The pipeline runner lives in `crucible_framework`:
  - `Crucible.Pipeline.Runner`
  - `CrucibleFramework.run/2`
- `crucible_ir` does not execute anything; it only defines specs.

## 3) Stage Contract

### Behaviour
- `Crucible.Stage` defines the runtime contract.
- Required callback: `run(context, opts)`.
- Optional callback: `describe(opts)`.

### Required Semantics
- `run/2` must return `{:ok, %Crucible.Context{}}` or `{:error, reason}`.
- Stages must not mutate global state or bypass persistence helpers.
- Stages must be network-mockable and testable in isolation.

### Options Handling
- `CrucibleIR.StageDef.options` is an opaque map.
- Each stage owns its own options schema and validation.
- Stages may accept typed configs (e.g., `%CrucibleIR.Training.Config{}`) but must normalize internally.

### Describe Contract (Required by Policy)
Every stage module should implement `describe/1` to provide a discoverable schema:

```
%{
  name: :supervised_train,
  required: [:training_config, :ports],
  optional: [:log_path, :seed, :checkpoint_every],
  types: %{
    training_config: {:struct, CrucibleIR.Training.Config},
    ports: {:struct, CrucibleTrain.Ports}
  }
}
```

This is a policy requirement even though the callback is optional at the behaviour level.

## 4) Scope: Which Stages Must Implement describe/1

All modules that implement `Crucible.Stage` across the ecosystem, including:

- `crucible_train` (`lib/crucible_train/stages/*.ex`)
- `crucible_model_registry` (`lib/crucible_model_registry/stages/*.ex`)
- `crucible_deployment` (`lib/crucible_deployment/stages/*.ex`)
- `crucible_feedback` (`lib/crucible_feedback/stages/*.ex`)
- `crucible_bench`, `crucible_ensemble`, `crucible_hedging` (any Stage modules)
- Product stages such as `ExFairness` or `ex_topology` where they implement `Crucible.Stage`

If a repo has no stage modules, no action is required.

## 5) Acceptance Criteria

- Runner location is documented and unambiguous.
- Stage contracts are discoverable via `describe/1` in all stage modules.
- IR remains free of execution logic; framework owns execution.

