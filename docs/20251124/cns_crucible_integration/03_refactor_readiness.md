# Refactor Readiness for TrainingV2

**Date:** 2025-11-24  
**Status:** Checklist before enabling TrainingV2

## What must be in place

1) **Placement decision**
   - Keep TrainingV2 in `cns_crucible` (preferred) or a new `cns_training` app.
   - Leave `cns` core with metrics/adapter only to avoid pulling in Crucible/Tinkex.

2) **Dependencies**
   - Add to the host app (not `cns`):
     - `{:crucible_framework, path: "../crucible_framework"}`
     - `{:tinkex, path: "../tinkex"}`
   - Ensure `crucible_framework` pulls `crucible_ensemble`, `crucible_hedging`, `crucible_bench`, `crucible_trace` (already in its `mix.exs`).

3) **Config**
   - Disable repo if no DB: `config :crucible_framework, :enable_repo, false`
   - Set analysis adapters in the host app: `config :crucible_framework, :analysis_adapter, YourMetricsAdapter; :analysis_surrogate_adapter, YourSurrogateAdapter; :analysis_tda_adapter, YourTDAAdapter`
   - Provide Tinkex creds: `TINKER_API_KEY`, `TINKER_BASE_URL`

4) **Data + pipeline**
   - Dataset: JSONL with `input_key: :prompt`, `output_key: :completion`, or a loader stage.
   - Pipeline: `[:data_load, :data_checks, :guardrails, :backend_call, :analysis_surrogate_validation, :analysis_tda_validation, :analysis_metrics, :bench, :report]`
   - Backend: `%BackendRef{id: :tinkex, profile: :lora_finetune, options: %{base_model: "...", lora_rank: ..., learning_rate: ...}}`

5) **Tests**
   - Integration test that builds an Experiment IR, runs `CrucibleFramework.run/1` on a tiny dataset, and asserts CNS metrics + bench present.
   - Mock backend option for CI if Tinkex is unavailable.

## Migration steps (if/when enabling TrainingV2)

1. Move/rename `cns/lib/cns/training_v2.ex.disabled` into the host app (e.g., `cns_crucible/lib/.../training_v2.ex`).
2. Wire dataset prep to real files and ensure `Stage.DataLoad` options match.
3. Ensure CNS adapters are configured in the consumer app (see `cns_crucible` adapters).
4. Decide on ensemble/hedging defaults in the Experiment IR (keep off for first run).
5. Add docs and examples in `cns_crucible` (CLI script or Mix task).

## Open questions

- Do we want `cns` Hex package to stay light? If yes, keep all Crucible/Tinkex deps out of it.
- Do we need a mock backend for offline CI? If so, add a `Crucible.Backend.Mock` in tests.
- When to enable DB-backed runs? Only if persistence/reporting is required; otherwise keep `enable_repo: false`.
