# Prompt: Enforce Pipeline Runner and Stage Contract

Date: 2025-12-26

## Goal

Implement the contract defined in:
- `/home/home/p/g/North-Shore-AI/crucible_framework/docs/20251226/pipeline_runner_contract/PIPELINE_RUNNER_AND_STAGE_CONTRACT.md`

Ensure the runner is clearly documented in `crucible_framework` and that all stage modules across the ecosystem implement `describe/1` with a discoverable schema.

## Required Reading (Full Paths)

### Repo Guidance
- `/home/home/p/g/North-Shore-AI/crucible_framework/AGENTS.md`
- `/home/home/p/g/North-Shore-AI/crucible_framework/README.md`
- `/home/home/p/g/North-Shore-AI/crucible_framework/CHANGELOG.md`
- `/home/home/p/g/North-Shore-AI/crucible_framework/mix.exs`

### Runner and Stage Contracts
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible_framework.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/pipeline/runner.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage.ex`
- `/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/context.ex`

### Stage Modules to Update (If Present)
- `/home/home/p/g/North-Shore-AI/crucible_train/lib/crucible_train/stages/`
- `/home/home/p/g/North-Shore-AI/crucible_model_registry/lib/crucible_model_registry/stages/`
- `/home/home/p/g/North-Shore-AI/crucible_deployment/lib/crucible_deployment/stages/`
- `/home/home/p/g/North-Shore-AI/crucible_feedback/lib/crucible_feedback/stages/`
- `/home/home/p/g/North-Shore-AI/crucible_bench/lib/`
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/`
- `/home/home/p/g/North-Shore-AI/crucible_hedging/lib/`
- `/home/home/p/g/North-Shore-AI/ExFairness/lib/`
- `/home/home/p/g/North-Shore-AI/ex_topology/lib/`

### Design Doc
- `/home/home/p/g/North-Shore-AI/crucible_framework/docs/20251226/pipeline_runner_contract/PIPELINE_RUNNER_AND_STAGE_CONTRACT.md`

## Context Summary

The runner lives only in `crucible_framework`. Stages across the ecosystem must implement `describe/1` to document required/optional options and expected types. Stage options remain opaque in IR.

## Implementation Requirements

1) Document runner location in `crucible_framework` docs and README if needed.
2) Add or update `describe/1` implementations in all stage modules that implement `Crucible.Stage`.
3) If any stage module lacks `describe/1`, add a schema map describing required/optional keys and types.
4) Keep stage option validation inside stages (not in IR).

## TDD and Quality Gates

- Write tests first where behavior is introduced.
- `mix test` must pass in all modified repos.
- `mix compile --warnings-as-errors` must be clean.
- `mix format` must be clean.
- `mix credo --strict` must be clean.
- `mix dialyzer` must be clean.

## Version Bump (Required)

- Bump version `0.x.y` in `/home/home/p/g/North-Shore-AI/crucible_framework/mix.exs`.
- Update `/home/home/p/g/North-Shore-AI/crucible_framework/README.md` to reflect the new version.
- Add a 2025-12-26 entry to `/home/home/p/g/North-Shore-AI/crucible_framework/CHANGELOG.md`.

If other repos are modified for stage `describe/1` additions, bump their versions and update their README and CHANGELOG as well.

