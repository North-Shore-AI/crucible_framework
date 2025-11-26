# CNS ↔ Crucible ↔ Tinkex Integration Surfaces

**Date:** 2025-11-24  
**Status:** Draft (actionable)  
**Audience:** CNS owners, Crucible core, cns_crucible maintainers

## What plugs into what

- **CNS (library)**
  - Provides domain logic (SNOs, topology, chirality, critics).
  - Exposes behaviours only; concrete adapters live outside the core library.
  - Should stay backend-agnostic; avoid hard deps on Crucible/Tinkex in `mix.exs`.
- **cns_crucible (app)**
  - Owns concrete Experiment IR definitions and data loaders.
  - Should host training orchestration (the future `TrainingV2`) to avoid pulling Crucible/Tinkex into the core CNS lib.
  - Responsible for wiring CNS adapter, stages, and backend config.
- **crucible_framework (engine)**
  - Owns IR structs (`Crucible.IR.*`), `Crucible.Context`, pipeline runner, stage registry, backend behaviour.
- Ships stages (`BackendCall`, `CNSMetrics`, `CNSSurrogateValidation`, `CNSTDAValidation`, `Bench`, `Report`, etc.).
- **tinkex (backend)**
  - Backend implementation behind `Crucible.Backend.Tinkex`; selected via `Experiment.backend`.

## Recommended placement of the bridge

- Keep **CNS core** clean: ship theory/algorithms only; adapters belong in the integration app.
- Move/keep **training + experiment IR** in **cns_crucible** (or another app), not in `cns`. That avoids forcing Crucible/Tinkex deps into the library and keeps the core publishable to Hex without heavy deps.
- If you need shared helpers, create a thin `CNS.TrainingV2` in `cns_crucible` (or a new `cns_training` app) that wraps Crucible IR. Leave the disabled file in `cns` as a reference only.

## Integration points (today)

- Metrics: `Crucible.Stage.Analysis.Metrics` → configured `Crucible.Analysis.Adapter.evaluate/3`.
- Surrogates: `Crucible.Stage.Analysis.SurrogateValidation` (+ optional `Analysis.Filter`).
- Backend: `Crucible.Stage.BackendCall` → `Crucible.Backend.Tinkex` (or others).
- Reliability: ensemble + hedging inside `BackendCall`; stats via `Stage.Bench`; trace via `TraceIntegration`.

## Minimal wiring checklist (per experiment)

- Add to config (consumer app):
  - `config :crucible_framework,
    analysis_adapter: YourMetricsAdapter,
    analysis_surrogate_adapter: YourSurrogateAdapter,
    analysis_tda_adapter: YourTDAAdapter,
    enable_repo: false`
- Build `Crucible.IR.Experiment` with stages:
  - `:data_load -> :data_checks -> :guardrails -> :backend_call -> :analysis_surrogate_validation -> :analysis_tda_validation -> :analysis_metrics -> :bench -> :report`
- Backend ref: `%BackendRef{id: :tinkex, profile: :lora_finetune, options: %{...}}`
- Dataset ref: local JSONL with `input_key: :prompt`, `output_key: :completion`.
