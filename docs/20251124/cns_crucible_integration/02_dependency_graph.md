# Dependency Graph (CNS ↔ Crucible ↔ Tinkex)

**Date:** 2025-11-24  
**Status:** Source-of-truth snapshot

## Graph (modules/apps)

```
            +----------------+
            |   cns (lib)    |
            |----------------|
            | SNO, Metrics   |
            | Surrogates     |
            | Adapter behaviours |
            +--------+-------+
                     |
                     |  (via config :cns_adapter)
                     v
            +-----------------------+
            |  crucible_framework   |
            |-----------------------|
            | IR structs            |
            | Context/Runner        |
            | Stages:               |
            |  - BackendCall        |
            |  - CNSMetrics         |
            |  - CNSSurrogateVal    |
            |  - CNSTDAValidation   |
            |  - Bench, Report      |
            | TraceIntegration      |
            +----+----+----+-------+
                 |    |    |
                 |    |    +-------------------+
                 |    |                        |
                 v    v                        v
        +-------------+          +--------------------+
        | crucible_*  |          |     tinkex         |
        | ensemble    |          | (backend impl)     |
        | hedging     |          |  - client SDK      |
        | bench       |          |  - service calls   |
        | trace       |          +--------------------+
        +-------------+

            ^  
            |  (Experiment IR + data)  
    +----------------------+
    |    cns_crucible      |
    |----------------------|
    | Experiment builders  |
    | Dataset loaders      |
    | Optional TrainingV2  |
    +----------------------+
```

## Dependency table

- **cns**
  - Runtime: `nx`, `libgraph`, `jason`, etc.
  - Optional: `crucible_framework` **not** declared (by design). Adapter assumes presence when used.
- **crucible_framework**
  - Direct: `crucible_ensemble`, `crucible_hedging`, `crucible_bench`, `crucible_trace`, `tinkex`, `ecto_sql` (can be disabled), `postgrex`.
  - Uses CNS via configured adapter only (no hard dep).
- **crucible_ensemble / crucible_hedging / crucible_bench / crucible_trace**
  - Standalone libs; pulled in by `crucible_framework`.
- **tinkex**
  - Standalone SDK; pulled in by `crucible_framework` or consumer app.
- **cns_crucible**
  - Direct: `cns`, `crucible_framework`, `tinkex`, ML stack (nx/bumblebee/axon), data utils.
  - Should host TrainingV2 / Experiment IR to avoid polluting `cns` deps.

## Notes

- Turning on `CNS.TrainingV2` in `cns` would force `crucible_framework` + `tinkex` into the core lib. Prefer keeping training orchestration in `cns_crucible`.
- If you need `TrainingV2` shared, consider a small adapter app (e.g., `cns_training`) that depends on both `cns` and `crucible_framework`.
