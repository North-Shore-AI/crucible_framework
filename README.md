<div align="center"><img src="assets/crucible_framework.svg" width="400" alt="Crucible Framework Logo" /></div>

# CrucibleFramework

**A reliability-first experiment engine for LLM training and evaluation**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-blue.svg)](https://www.erlang.org)
[![Hex.pm](https://img.shields.io/hexpm/v/crucible_framework.svg)](https://hex.pm/packages/crucible_framework)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-purple.svg)](https://hexdocs.pm/crucible_framework)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/North-Shore-AI/crucible_framework/blob/main/LICENSE)

---

## What’s New (v0.3.0 · 2025-11-23)

- Declarative Experiment IR (`Crucible.IR.*`) that fully describes datasets, stages, backends, and outputs.
- Stage-based pipeline runner (`Crucible.Pipeline.Runner`) with built-in stages: data_load → data_checks → guardrails → backend_call → cns_surrogate_validation → cns_tda_validation → cns_metrics → bench → report.
- Backend behaviour plus a mockable Tinkex implementation for LoRA training and sampling.
- Persistence layer (Ecto/Postgres) for experiments, runs, and artifacts; one-step bootstrap script (`scripts/setup_db.sh`).
- Live Tinkex demo pipeline (`examples/tinkex_live.exs`) wired to the new IR and stages.

---

## Quick Start

### Prerequisites
- Elixir ≥ 1.14 / OTP ≥ 25
- Local PostgreSQL (listening on `localhost:5432`)
- (Optional) `TINKER_API_KEY` for live Tinkex runs

### Install from Hex
```elixir
def deps do
  [
    {:crucible_framework, "~> 0.3.0"}
  ]
end
```

### 1) Bootstrap the database (dev + test)
```bash
./scripts/setup_db.sh
```
Creates the `crucible_dev` role, dev/test databases, runs migrations, and aligns with the baked-in configs (`config/dev.exs`, `config/test.exs`). No env vars required for DB access.

### 2) Run the suite
```bash
mix test                  # unit suite
MIX_ENV=test mix test --include integration  # includes persistence tests
```

### 3) Run the live Tinkex demo
```bash
export TINKER_API_KEY=your_key
mix run examples/tinkex_live.exs
```
This executes a tiny SciFact-style pipeline through the stage engine, trains via Tinkex, samples a prompt, and emits a report to stdout + `reports/`.

---

## Core Concepts

### Experiment IR
Experiments are pure structs—serializable, inspectable, and backend-agnostic:
```elixir
alias Crucible.IR.{Experiment, DatasetRef, BackendRef, StageDef, ReliabilityConfig,
                   EnsembleConfig, HedgingConfig, GuardrailConfig, StatsConfig, FairnessConfig, OutputSpec}

experiment = %Experiment{
  id: "tinkex_scifact_demo",
  description: "Minimal Tinkex training pipeline",
  dataset: %DatasetRef{name: "scifact_claims", options: %{path: "priv/data/scifact_claim_extractor_clean.jsonl", limit: 4, batch_size: 2}},
  pipeline: [
    %StageDef{name: :data_load, options: %{input_key: :prompt, output_key: :completion}},
    %StageDef{name: :data_checks, options: %{required_fields: [:input, :output]}},
    %StageDef{name: :guardrails},
    %StageDef{name: :backend_call, options: %{mode: :train, sample_prompts: ["Write a counterclaim."], create_sampler?: true}},
    %StageDef{name: :cns_metrics},
    %StageDef{name: :bench},
    %StageDef{name: :report, options: %{sink: :stdout, formats: [:markdown]}}
  ],
  backend: %BackendRef{id: :tinkex, profile: :lora_finetune, options: %{base_model: "meta-llama/Llama-3.2-1B"}},
  reliability: %ReliabilityConfig{
    ensemble: %EnsembleConfig{strategy: :none},
    hedging: %HedgingConfig{strategy: :off},
    guardrails: %GuardrailConfig{profiles: [:default]},
    stats: %StatsConfig{tests: [:bootstrap]},
    fairness: %FairnessConfig{enabled: false}
  },
  outputs: [
    %OutputSpec{name: :report, formats: [:markdown, :json], sink: :file, options: %{path: "reports/demo.md"}}
  ]
}
```

### Pipeline Engine
`Crucible.Pipeline.Runner` walks the `pipeline` list and calls each stage module. Built-in stages:
- `DataLoad`: streams and batches data (in-memory/JSONL helpers included)
- `DataChecks`: basic schema checks or pluggable validators
- `Guardrails`: adapter-based safety scanning (default no-op)
- `BackendCall`: training/sampling against a configured backend
- `CNSMetrics`: optional CNS adapter hook
- `Bench`: placeholder for statistical testing hooks (crucible_bench integration point)
- `Report`: renders Markdown/JSON, writes artifacts, and attaches to the run record

### Backends
- Behaviour: `Crucible.Backend` defines `init/start_session/train_step/save_checkpoint/create_sampler/sample`.
- Implementation: `Crucible.Backend.Tinkex` delegates to the `tinkex` SDK via a mockable client (`LiveClient` for production, `ClientMock` for tests).

### Persistence
- Repo: `CrucibleFramework.Repo` (Postgres)
- Schemas: experiments, runs, artifacts (`lib/crucible_framework/persistence/*.ex`)
- Helpers: `CrucibleFramework.Persistence.start_run/finish_run/record_artifact`
- Toggle via `:enable_repo` (enabled by default)

### Safety & Evaluation Adapters
- Guardrails: plug your adapter via `config :crucible_framework, :guardrail_adapter, YourModule` (default no-op).
- CNS: plug adapters via `config :crucible_framework, :cns_adapter / :cns_surrogate_adapter / :cns_tda_adapter, YourModule` (defaults are no-ops).

---

## Running Your Own Pipeline
```elixir
{:ok, ctx} = CrucibleFramework.run(experiment, persist: true)
IO.inspect(ctx.metrics, label: "metrics")
```
Persistence will store the experiment, the run record, and any file artifacts emitted by `Report`.

---

## Development Notes
- DB credentials are baked into `config/dev.exs` and `config/test.exs` for frictionless local work (`crucible_dev` / `crucible_dev_pw`).
- `scripts/setup_db.sh` is idempotent—safe to rerun anytime.
- For live Tinkex calls, set `TINKER_API_KEY` (see `config/runtime.exs`).

---

## License
MIT. See [LICENSE](LICENSE).
