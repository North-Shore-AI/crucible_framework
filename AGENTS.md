# AGENTS

Operational guidance for humans and automation working inside `crucible_framework`.

## Roles
- **Operator (you):** orchestrates runs, approves live credentials, and owns production data.
- **Builder:** writes stages/backends and updates IR contracts.
- **Reviewer:** performs code review with a reliability/safety mindset.
- **Automation:** CI, linters, formatters, and scripted migrations.

## Interaction Model
- Prefer **stage implementations** over ad-hoc scripts; every new behavior should be a `Crucible.Stage`.
- Treat the **IR structs** as the single source of truth; any surface (CLI, tests, examples) should produce/consume them.
- Backends must implement **`Crucible.Backend`** and be registered in `config/config.exs`.
- For safety layers, use adapter modules: **guardrails** and **CNS** are swappable and should not be hard-coded inside stages.
- Persistence is opt-in only through the **`CrucibleFramework.Persistence`** helpers; do not bypass the repo directly in stages.

## Local Development
- Use `./scripts/setup_db.sh` to provision the Postgres role/databases and run migrations.
- `mix test` for unit coverage; `MIX_ENV=test mix test --include integration` to exercise persistence.
- For live Tinkex pipelines, export `TINKER_API_KEY` and run `mix run examples/tinkex_live.exs`.

## Design Principles
- **Declarative first:** experiments are data (IR) not code.
- **Composable stages:** small, replaceable steps over a shared `Crucible.Context`.
- **Mock-friendly backends:** everything that talks to the network must be mockable.
- **Verbose reports:** every run should emit Markdown/JSON artifacts with metrics and traceable outputs.

## When Adding New Features
1. Start with IR additions if the behavior needs new fields.
2. Add or extend a `Stage` module; wire it in `config/config.exs` if generic.
3. Update tests with Mox for external boundaries.
4. Document runnable examples (prefer `examples/*.exs`) to keep workflows reproducible.
