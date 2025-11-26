<div align="center"><img src="assets/crucible_framework.svg" width="400" alt="Crucible Framework Logo" /></div>

# CrucibleFramework

**A reliability-first experiment engine for LLM training and evaluation**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-blue.svg)](https://www.erlang.org)
[![Hex.pm](https://img.shields.io/hexpm/v/crucible_framework.svg)](https://hex.pm/packages/crucible_framework)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-purple.svg)](https://hexdocs.pm/crucible_framework)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/North-Shore-AI/crucible_framework/blob/main/LICENSE)

---

## What's New (v0.3.0 - 2025-11-23)

- Declarative Experiment IR (`Crucible.IR.*`) that fully describes datasets, stages, backends, and outputs.
- Stage-based pipeline runner (`Crucible.Pipeline.Runner`) with built-in stages for the full ML lifecycle.
- Backend behaviour plus a mockable Tinkex implementation for LoRA training and sampling.
- Persistence layer (Ecto/Postgres) for experiments, runs, and artifacts.
- **CNS Integration**: Full support for CNS (Chiral Narrative Synthesis) experiments via the `cns_crucible` companion app.

---

## Quick Start

### Prerequisites
- Elixir >= 1.14 / OTP >= 25
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

---

## Architecture Overview

CrucibleFramework is the **reliability engine** at the center of the North-Shore-AI experiment infrastructure. It provides a backend-agnostic IR (Intermediate Representation) that describes experiments declaratively, allowing them to be executed through configurable pipeline stages.

```
          ┌────────────────────────────┐
          │  SURFACES / CLIENT APPS    │
          │                            │
          │  • cns_crucible            │
          │  • LiveView / Phoenix UI   │
          │  • Python SDK / notebooks  │
          │  • CLI (mix tasks)         │
          └──────────────┬─────────────┘
                         │  (Experiment IR)
          ┌──────────────▼─────────────┐
          │  CRUCIBLE RELIABILITY CORE │
          │                            │
          │  • crucible_framework      │
          │  • crucible_ensemble       │
          │  • crucible_hedging        │
          │  • crucible_bench          │
          │  • crucible_trace          │
          └──────────────┬─────────────┘
                         │  (Backend behaviour)
      ┌──────────────────┼────────────────────────────┐
      │                  │                            │
┌─────▼─────┐      ┌─────▼─────┐              ┌───────▼──────┐
│ Tinkex    │      │ NxLocal   │              │ External LLM │
│ backend   │      │ backend   │              │ backends     │
│ (LoRA)    │      │ (Axon/Nx) │              │ (OpenAI, etc)│
└───────────┘      └───────────┘              └──────────────┘
```

**Key insight**: Tinkex is "just" the first implementation of `Crucible.Backend`. CNS experiments are "just" the first client of the IR. The architecture scales to any backend and any experiment type.

---

## The Experiment IR

The Experiment IR is the **canonical contract** between all surfaces (CLI, Python SDK, LiveView), the engine, backends, and domain libraries. It's designed to be:

- **Backend-agnostic**: No assumptions about infrastructure
- **Serializable**: Can be stored, transmitted, and inspected
- **Composable**: Stages are plugins, not hardcoded

### Core IR Structs

```
lib/crucible/ir/
├── experiment.ex        # Top-level experiment definition
├── dataset_ref.ex       # Logical reference to a dataset
├── backend_ref.ex       # Logical reference to a backend
├── stage_def.ex         # Pipeline stage definition
├── reliability_config.ex # Ensemble, hedging, guardrails, stats, fairness
├── ensemble_config.ex   # Multi-model voting configuration
├── hedging_config.ex    # Request hedging configuration
├── guardrail_config.ex  # Safety/guardrail configuration
├── stats_config.ex      # Statistical testing configuration
├── fairness_config.ex   # Fairness evaluation configuration
└── output_spec.ex       # Output artifact specification
```

### IR Design Principles

From the design spec (`001_crucible_long_term_plan.md`):

1. **One experiment engine** - Everything plugs into Crucible via clean extension points
2. **Stages are behaviours** - Each stage implements `Crucible.Stage.run/2`
3. **Backends are location-independent** - Can be in-process, another BEAM node, or a remote API
4. **No single-node assumptions** - IR never talks about nodes/hosts

### Complete IR Example

```elixir
alias Crucible.IR.{
  Experiment, DatasetRef, BackendRef, StageDef, ReliabilityConfig,
  EnsembleConfig, HedgingConfig, GuardrailConfig, StatsConfig,
  FairnessConfig, OutputSpec
}

experiment = %Experiment{
  id: "cns_scifact_tinkex_v1",
  description: "CNS claim extraction on SciFact via Tinkex LoRA backend",
  tags: ["cns", "scifact", "tinkex"],

  dataset: %DatasetRef{
    provider: :crucible_datasets,
    name: "scifact_claims",
    split: :train,
    options: %{limit: 1000, batch_size: 4}
  },

  pipeline: [
    %StageDef{name: :data_load},
    %StageDef{name: :data_checks, options: %{required_fields: [:input, :output]}},
    %StageDef{name: :guardrails},
    %StageDef{name: :backend_call, options: %{mode: :train}},
    %StageDef{name: :analysis_surrogate_validation},
    %StageDef{name: :analysis_tda_validation},
    %StageDef{name: :analysis_metrics},
    %StageDef{name: :bench},
    %StageDef{name: :report}
  ],

  backend: %BackendRef{
    id: :tinkex,
    profile: :lora_finetune,
    options: %{base_model: "meta-llama/Llama-3.2-1B", lora_rank: 8}
  },

  reliability: %ReliabilityConfig{
    ensemble: %EnsembleConfig{strategy: :none},
    hedging: %HedgingConfig{strategy: :off},
    guardrails: %GuardrailConfig{profiles: [:default]},
    stats: %StatsConfig{tests: [:bootstrap, :mann_whitney], alpha: 0.05},
    fairness: %FairnessConfig{enabled: false}
  },

  outputs: [
    %OutputSpec{name: :metrics_report, formats: [:markdown, :json], sink: :file}
  ]
}
```

---

## Pipeline Stages

The pipeline is executed by `Crucible.Pipeline.Runner`, which iterates through stages and threads a `Crucible.Context` through each.

### Built-in Stages

| Stage | Module | Purpose |
|-------|--------|---------|
| `:data_load` | `Crucible.Stage.DataLoad` | Stream and batch dataset |
| `:data_checks` | `Crucible.Stage.DataChecks` | Schema validation |
| `:guardrails` | `Crucible.Stage.Guardrails` | Safety scanning (LlmGuard integration) |
| `:backend_call` | `Crucible.Stage.BackendCall` | Training/sampling via backend |
| `:analysis_surrogate_validation` | `Crucible.Stage.Analysis.SurrogateValidation` | Surrogate topology checks (wired by integration app) |
| `:analysis_tda_validation` | `Crucible.Stage.Analysis.TDAValidation` | Full TDA analysis (wired by integration app) |
| `:analysis_metrics` | `Crucible.Stage.Analysis.Metrics` | Quality metrics (wired by integration app) |
| `:analysis_filter` | `Crucible.Stage.Analysis.Filter` | Filter outputs by surrogate scores |
| `:bench` | `Crucible.Stage.Bench` | Statistical testing |
| `:report` | `Crucible.Stage.Report` | Generate output artifacts |

### Stage Behaviour

Every stage implements `Crucible.Stage`:

```elixir
defmodule Crucible.Stage do
  alias Crucible.Context

  @callback run(context :: Context.t(), opts :: map()) ::
              {:ok, Context.t()} | {:error, term()}

  @callback describe(opts :: map()) :: map()
  @optional_callbacks describe: 1
end
```

### Custom Stages

Create domain-specific stages by implementing the behaviour:

```elixir
defmodule MyApp.Stage.CustomValidation do
  @behaviour Crucible.Stage

  @impl true
  def run(%Crucible.Context{} = ctx, opts) do
    # Transform context, add metrics, etc.
    {:ok, %{ctx | metrics: Map.put(ctx.metrics, :custom, my_metrics)}}
  end
end
```

---

## Backend Behaviour

Backends abstract training and inference. The `Crucible.Backend` behaviour defines:

```elixir
defmodule Crucible.Backend do
  @callback init(backend_id, backend_config) :: {:ok, backend_state} | {:error, term()}
  @callback start_session(backend_state, Experiment.t()) :: {:ok, session} | {:error, term()}
  @callback train_step(session, batch) :: {:ok, %{loss: float(), ...}} | {:error, term()}
  @callback save_checkpoint(session, step) :: {:ok, checkpoint_ref} | {:error, term()}
  @callback create_sampler(session, checkpoint_ref) :: {:ok, sampler} | {:error, term()}
  @callback sample(sampler, prompt, opts) :: {:ok, [binary()]} | {:error, term()}
end
```

### Tinkex Backend

`Crucible.Backend.Tinkex` implements this behaviour, delegating to the Tinkex SDK for LoRA fine-tuning and sampling.

---

## Runtime Context

The `Crucible.Context` struct is threaded through all pipeline stages:

```elixir
%Crucible.Context{
  experiment_id: "cns_scifact_v1",
  run_id: "abc-123",
  experiment: %Experiment{...},

  # Data
  dataset: loaded_data,
  batches: stream_of_batches,
  examples: list_of_examples,

  # Backend state
  backend_sessions: %{tinkex: session_pid},
  backend_state: %{},

  # Results
  outputs: [generated_outputs],
  metrics: %{cns: %{...}, bench: %{...}},
  artifacts: %{report: "path/to/report.md"},

  # Observability
  trace: trace_chain,
  telemetry_context: %{},

  # Extension point
  assigns: %{custom_data: ...}
}
```

---

## Crucible Component Libraries

CrucibleFramework integrates with specialized reliability libraries:

### crucible_ensemble
Multi-model voting for increased reliability:
- **Strategies**: Majority vote, weighted vote, best confidence, unanimous
- **Expected improvement**: 96-99% accuracy vs 89-92% single model

### crucible_hedging
Tail latency reduction via request hedging:
- **Strategies**: Fixed delay, percentile-based, adaptive
- **Expected improvement**: 50-75% P99 latency reduction

### crucible_bench
Statistical testing for publication-quality results:
- **Tests**: t-tests, ANOVA, Mann-Whitney, Wilcoxon, bootstrap
- **Effect sizes**: Cohen's d, eta-squared
- **Power analysis**: Built-in

### crucible_trace
Causal transparency and decision provenance:
- **Visualization**: Interactive HTML trace viewer
- **Integration**: Automatic stage event emission

---

## CNS Integration: The "Hello World" Experiment

The CNS (Chiral Narrative Synthesis) SciFact experiment demonstrates the full integration. This is the canonical example of using CrucibleFramework.

### Data Flow

```
┌─────────────────┐
│  SciFact JSONL  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataLoad      │  Load and batch dataset
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataChecks    │  Validate required fields
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Guardrails    │  LlmGuard security checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BackendCall   │  Tinkex LoRA training
│   (Tinkex)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CNSSurrogate    │  β₁ and fragility surrogates
│ Validation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CNSTdaValidation│  Full TDA (if enabled)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CNSMetrics    │  Schema, citation, topology, chirality
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Bench         │  Statistical analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Report        │  Generate outputs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Markdown/JSON  │
└─────────────────┘
```

### IR Flow Through the System

1. **Surface layer** (`cns_crucible`): Builds `%Experiment{}` IR with CNS-specific configuration
2. **Engine** (`crucible_framework`): Receives IR, resolves stages, executes pipeline
3. **Stages**: Transform `%Context{}`, call adapters, accumulate metrics
4. **Backend** (`Crucible.Backend.Tinkex`): Handles actual Tinkex API calls
5. **Adapters** (`CnsExperiments.Adapters.*`): Bridge CNS metrics into Crucible stages

### Running the Experiment

```elixir
# Via cns_crucible
CnsExperiments.Experiments.ScifactClaimExtraction.run(
  batch_size: 4,
  limit: 100,
  base_model: "meta-llama/Llama-3.2-1B"
)

# Or directly with IR
{:ok, ctx} = CrucibleFramework.run(experiment)
IO.inspect(ctx.metrics.cns, label: "CNS Metrics")
```

---

## Adapter Architecture

CrucibleFramework defines adapter behaviours for pluggable evaluation:

### Analysis Adapters

```elixir
# lib/crucible/analysis/adapter.ex
defmodule Crucible.Analysis.Adapter do
  @callback evaluate(examples, outputs, opts) :: {:ok, map()} | {:error, term()}
end

# lib/crucible/analysis/surrogate_adapter.ex
defmodule Crucible.Analysis.SurrogateAdapter do
  @callback compute_surrogates(examples, outputs, opts) :: {:ok, map()} | {:error, term()}
end

# lib/crucible/analysis/tda_adapter.ex
defmodule Crucible.Analysis.TdaAdapter do
  @callback compute_tda(snos, opts) :: {:ok, map()} | {:error, term()}
end
```

Configure adapters in `config/config.exs`:

```elixir
config :crucible_framework,
  analysis_adapter: CnsExperiments.Adapters.Metrics,
  analysis_surrogate_adapter: CnsExperiments.Adapters.Surrogates,
  analysis_tda_adapter: CnsExperiments.Adapters.TDA
```

---

## Persistence

CrucibleFramework includes optional Ecto/Postgres persistence:

```elixir
# Schemas
CrucibleFramework.Persistence.ExperimentRecord
CrucibleFramework.Persistence.RunRecord
CrucibleFramework.Persistence.ArtifactRecord

# Usage
{:ok, ctx} = CrucibleFramework.run(experiment, persist: true)
```

---

## Configuration Reference

### Full Config Example

```elixir
# config/config.exs
config :crucible_framework,
  # Stage registry
  stage_registry: %{
    data_load: Crucible.Stage.DataLoad,
    data_checks: Crucible.Stage.DataChecks,
    guardrails: Crucible.Stage.Guardrails,
    backend_call: Crucible.Stage.BackendCall,
    analysis_metrics: Crucible.Stage.Analysis.Metrics,
    analysis_surrogate_validation: Crucible.Stage.Analysis.SurrogateValidation,
    analysis_tda_validation: Crucible.Stage.Analysis.TDAValidation,
    analysis_filter: Crucible.Stage.Analysis.Filter,
    bench: Crucible.Stage.Bench,
    report: Crucible.Stage.Report
  },

  # Backend registry
  backend_registry: %{
    tinkex: Crucible.Backend.Tinkex
  },

  # Adapters
  analysis_adapter: CnsExperiments.Adapters.Metrics,
  analysis_surrogate_adapter: CnsExperiments.Adapters.Surrogates,
  analysis_tda_adapter: CnsExperiments.Adapters.TDA,
  guardrail_adapter: Crucible.Stage.Guardrails.Noop,

  # Persistence
  ecto_repos: [CrucibleFramework.Repo],
  enable_repo: true

config :crucible_framework, CrucibleFramework.Repo,
  database: "crucible_dev",
  username: "crucible_dev",
  password: "crucible_dev_pw",
  hostname: "localhost"
```

---

## Development

### Prerequisites
- Elixir 1.14+
- OTP 25+
- PostgreSQL

### Setup
```bash
git clone https://github.com/North-Shore-AI/crucible_framework.git
cd crucible_framework
mix deps.get
./scripts/setup_db.sh
mix test
```

### Testing
```bash
mix test                              # Unit tests
mix test --include integration        # Integration tests
mix test --cover                      # Coverage
mix dialyzer                          # Static analysis
```

---

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [cns](https://github.com/North-Shore-AI/cns) | Core CNS dialectical reasoning library |
| [cns_crucible](https://github.com/North-Shore-AI/cns_crucible) | CNS + Crucible integration harness |
| [tinkex](https://github.com/North-Shore-AI/tinkex) | Tinker SDK for LoRA training |
| [crucible_ensemble](https://github.com/North-Shore-AI/crucible_ensemble) | Multi-model voting |
| [crucible_hedging](https://github.com/North-Shore-AI/crucible_hedging) | Request hedging |
| [crucible_bench](https://github.com/North-Shore-AI/crucible_bench) | Statistical testing |
| [crucible_trace](https://github.com/North-Shore-AI/crucible_trace) | Causal transparency |

---

## License

MIT. See [LICENSE](LICENSE).
