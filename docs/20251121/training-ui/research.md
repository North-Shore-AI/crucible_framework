# North-Shore-AI Training UI Ecosystem Research

**Date:** 2025-11-21
**Purpose:** Integration analysis for crucible_framework
**Repos Investigated:** tinkex, cns_ui, crucible_ui

---

## Executive Summary

The North-Shore-AI organization has three interconnected projects that form a comprehensive ML training and UI ecosystem:

1. **Tinkex** - Elixir SDK for the Tinker ML Training and Inference API
2. **Crucible UI** - Phoenix LiveView Dashboard for ML Reliability Research
3. **CNS UI** - Phoenix LiveView interface for dialectical reasoning experiments

These projects form a layered architecture where Tinkex provides the training API client, Crucible UI serves as the foundational dashboard, and CNS UI extends it with domain-specific visualizations for CNS (Critic-Network Synthesis) research.

---

## 1. Tinkex - Training API SDK

### Overview

Tinkex is a complete Elixir port of the Tinker Python SDK, providing a functional, concurrent interface to distributed ML training using LoRA (Low-Rank Adaptation) fine-tuning.

**Version:** 0.1.1
**License:** Apache 2.0
**Status:** Active development, production-ready core

### Key Modules and APIs

#### Client Architecture

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `Tinkex.ServiceClient` | Entry point, session management | `start_link/1`, `create_lora_training_client/2`, `create_sampling_client/2`, `create_rest_client/1` |
| `Tinkex.TrainingClient` | LoRA fine-tuning operations | `forward_backward/4`, `optim_step/3`, `save_weights_for_sampler/2` |
| `Tinkex.SamplingClient` | Text generation/inference | `sample/4` |
| `Tinkex.RestClient` | Session/checkpoint management | `list_sessions/2`, `list_user_checkpoints/2` |
| `Tinkex.CheckpointDownload` | Model artifact downloads | `download/3` |

#### API Layer (`lib/tinkex/api/`)

- **`Tinkex.API.Training`** - Forward/backward pass, optimizer steps
  - `forward_backward/2`, `forward_backward_future/2`
  - `optim_step/2`, `optim_step_future/2`
  - `forward/2`

- **`Tinkex.API.Sampling`** - Async text generation
  - `sample_async/2` - High-concurrency sampling (100 connection pool)

- **`Tinkex.API.Service`** - Model/session creation
  - `create_model/2`
  - `create_sampling_session/2`

- **`Tinkex.API.Weights`** - Weight management
  - `save_weights_for_sampler/2`

#### Type System (`lib/tinkex/types/`)

Comprehensive type definitions for:
- `Datum`, `ModelInput` - Training data structures
- `SamplingParams`, `SampleResponse` - Inference parameters
- `LoraConfig`, `AdamParams` - Model configuration
- `ForwardBackwardOutput`, `OptimStepResponse` - Training results

### Architecture Highlights

1. **GenServer-based Clients** - Each client type is a supervised GenServer for fault tolerance
2. **Task-based Futures** - Async operations return Elixir Tasks for composability
3. **HTTP/2 Connection Pools** - Finch-based pools per operation type:
   - `:training` pool (5 connections) - Sequential, long-running
   - `:sampling` pool (100 connections) - High concurrency
   - `:session` pool - Model/session creation
4. **ETS-based Registry** - Lock-free reads for sampling client lookup
5. **Rate Limiting** - Per-tenant rate limiters with exponential backoff
6. **Telemetry Integration** - Standard `:telemetry` events for metrics

### Dependencies

```elixir
{:finch, "~> 0.18"},      # HTTP/2 client
{:jason, "~> 1.4"},        # JSON
{:nx, "~> 0.7"},           # Tensor operations
{:tokenizers, "~> 0.5"},   # HuggingFace tokenizers
{:telemetry, "~> 1.2"}     # Observability
```

### CLI Capabilities

Built-in CLI (`tinkex`) for:
- `checkpoint` - Create LoRA checkpoints
- `run` - Generate text with sampling clients
- `version` - Build metadata

---

## 2. Crucible UI - ML Reliability Dashboard

### Overview

Phoenix LiveView dashboard for monitoring and managing ML experiments within the Crucible reliability stack.

**Version:** 0.1.0
**License:** MIT
**Status:** Active development

### Key Modules and APIs

#### Core Contexts (`lib/crucible_ui/`)

| Context | Purpose | Key Functions |
|---------|---------|---------------|
| `CrucibleUI.Experiments` | Experiment CRUD | `list_experiments/0`, `create_experiment/1`, `start_experiment/1`, `complete_experiment/1` |
| `CrucibleUI.Runs` | Run lifecycle | `list_runs/0`, `create_run/1`, `start_run/1`, `complete_run/2`, `fail_run/1` |
| `CrucibleUI.Telemetry` | Event management | Event storage and retrieval |
| `CrucibleUI.Models` | Model metadata | Model CRUD operations |
| `CrucibleUI.Statistics` | Statistical results | Result storage and analysis |

#### Schemas

- `CrucibleUI.Experiments.Experiment` - Experiment metadata, status, timestamps
- `CrucibleUI.Runs.Run` - Individual run data, metrics, checkpoints
- `CrucibleUI.Telemetry.Event` - Telemetry event records
- `CrucibleUI.Models.Model` - Model configurations
- `CrucibleUI.Statistics.Result` - Statistical test results

#### LiveViews (`lib/crucible_ui_web/live/`)

- `DashboardLive` - Main overview
- `ExperimentLive.Index/Show` - Experiment management
- `RunLive.Show` - Run details with telemetry
- `StatisticsLive` - Statistical test visualization
- `EnsembleLive` - Ensemble voting dashboards
- `HedgingLive` - Request hedging metrics

#### API Controllers (`lib/crucible_ui_web/controllers/api/`)

- `ExperimentController` - REST API for experiments
- `TelemetryController` - Telemetry event endpoints
- `ModelController` - Model metadata endpoints
- **`TinkexJobController`** - Tinkex job orchestration via `Crucible.Tinkex.API.Router`

### Tinkex Integration

The `TinkexJobController` provides a Phoenix wrapper over Crucible Framework's Tinkex routing:

```elixir
# POST /api/tinkex/jobs - Submit training job
def create(conn, params) do
  TinkexRouter.submit(%{params: params, headers: conn.req_headers})
end

# GET /api/tinkex/jobs/:id/stream - SSE telemetry stream
def stream(conn, %{"id" => job_id}) do
  TinkexStream.to_enum(stream.subscribe, timeout: timeout_ms)
end
```

### PubSub Architecture

Real-time updates via Phoenix PubSub:
- `experiments:list` - Experiment list changes
- `experiment:{id}` - Individual experiment updates
- `runs:list` - Run list changes
- `run:{id}` - Individual run updates
- `experiment:{id}:runs` - Runs for specific experiment

### Dependencies

```elixir
{:phoenix, "~> 1.7.10"},
{:phoenix_live_view, "~> 0.20.1"},
{:ecto_sql, "~> 3.10"},
{:postgrex, ">= 0.0.0"},
{:crucible_framework, path: "../crucible_framework"}  # Direct integration
```

---

## 3. CNS UI - Dialectical Reasoning Interface

### Overview

Domain-specific Phoenix LiveView interface for CNS (Critic-Network Synthesis) dialectical reasoning experiments.

**Version:** 0.1.0
**License:** Apache 2.0
**Status:** Active development

### Key Modules and APIs

#### Core Contexts (`lib/cns_ui/`)

| Context | Purpose | Key Functions |
|---------|---------|---------------|
| `CnsUi.SNOs` | Structured Narrative Objects | SNO CRUD, exploration |
| `CnsUi.Experiments` | Experiment management | Experiment CRUD |
| `CnsUi.Training` | Training runs | `list_training_runs/0`, `create_training_run/1`, `add_checkpoint/2` |
| `CnsUi.Citations` | Citation validation | Citation CRUD, validation |
| `CnsUi.Challenges` | Adversarial challenges | Challenge management |
| `CnsUi.Metrics` | Quality metrics | Metrics snapshots |
| `CnsUi.CrucibleClient` | Crucible API integration | `create_job/1`, `get_job/1`, `subscribe_job/1` |

#### Crucible Integration

`CnsUi.CrucibleClient` provides HTTP client for Crucible Framework:

```elixir
# Submit training job to Crucible
@spec create_job(map()) :: {:ok, job_response()} | {:error, term()}
def create_job(payload) do
  with {:ok, url} <- build_url("/api/jobs"),
       {:ok, response} <- request(:post, url, body) do
    {:ok, decoded}
  end
end

# Subscribe to job updates via PubSub
@spec subscribe_job(String.t()) :: :ok | {:error, term()}
def subscribe_job(job_id) do
  Phoenix.PubSub.subscribe(pubsub, "training:#{job_id}")
end
```

#### LiveViews (`lib/cns_ui_web/live/`)

- `DashboardLive` - Main CNS dashboard
- `SNOLive.Index/Show` - SNO explorer
- `GraphLive` - Dialectical graph visualization
- `ProposerLive` - Thesis generation
- `AntagonistLive` - Antithesis/challenge views
- `SynthesizerLive` - Synthesis emergence
- `TrainingLive` - 6-step training wizard
- `MetricsLive` - Quality metrics
- `ExperimentLive.Index/Show` - Experiment management
- `RunLive` - Run monitoring with Crucible integration
- `OverlayLive` - CNS overlay on Crucible runs

#### CNS-Specific Components (`lib/cns_ui_web/components/`)

- `ChiralityGauge` - Chirality score visualization
- `EntailmentMeter` - Entailment verification scores
- `TopologyGraph` - Betti number/topology visualization
- `EvidenceTree` - Evidence chain display
- `CrucibleComponents` - Shared Crucible UI components (vendored)

### Training Wizard (TrainingLive)

6-step wizard for configuring CNS training runs:

1. **Dataset** - Path/format configuration
2. **Model Selection** - Base model (LLaMA, Mistral)
3. **LoRA Configuration** - Rank, alpha, dropout
4. **Weight Settings** - Citation validity weight
5. **Hyperparameters** - Learning rate, epochs
6. **Review & Submit** - Final review, Crucible job submission

Job submission flow:
```elixir
def handle_event("start_training", _params, socket) do
  payload = build_job_payload(socket.assigns.config)

  case CrucibleClient.create_job(payload) do
    {:ok, %{"id" => job_id} = job} ->
      CrucibleClient.subscribe_job(job_id)
      push_navigate(to: ~p"/runs/#{job_id}")
  end
end
```

### Shared Component Pattern

CNS UI vendors Crucible UI components for consistent styling:

```elixir
defmodule CrucibleUIWeb.Components do
  # Shared stat cards, progress bars, etc.
  def stat_card(assigns), do: ~H"..."
  def progress_bar(assigns), do: ~H"..."
end
```

### Dependencies

```elixir
{:phoenix, "~> 1.7.10"},
{:phoenix_live_view, "~> 0.20.1"},
{:crucible_ui, path: "../crucible_ui"},  # Sibling dependency
{:cns, path: "../cns"}                   # CNS core library
```

---

## Integration Architecture

### Layering Model

```
                    ┌─────────────────┐
                    │     CNS UI      │  Domain-specific views
                    │ (Phoenix App)   │  SNO explorer, chirality
                    └────────┬────────┘
                             │ imports
                    ┌────────▼────────┐
                    │   Crucible UI   │  Generic ML dashboard
                    │ (Phoenix App)   │  Experiments, runs, stats
                    └────────┬────────┘
                             │ path dep
                    ┌────────▼────────┐
                    │Crucible Framework│ Core research infra
                    │(Elixir Library)  │ Ensemble, hedging, bench
                    └────────┬────────┘
                             │ integrates
                    ┌────────▼────────┐
                    │     Tinkex      │  ML training SDK
                    │(Elixir Library)  │  LoRA, sampling, futures
                    └─────────────────┘
```

### Data Flow

1. **User configures training** in CNS UI's TrainingLive
2. **CNS UI calls** `CnsUi.CrucibleClient.create_job/1`
3. **Crucible Framework** receives job, uses Tinkex for actual ML ops
4. **Crucible UI** provides monitoring via `TinkexJobController`
5. **Updates flow** via PubSub to all subscribed LiveViews

### Current Integration Points

| Source | Target | Mechanism |
|--------|--------|-----------|
| CNS UI | Crucible UI | `{:crucible_ui, path: "../crucible_ui"}` |
| CNS UI | Crucible Framework | HTTP API via `CrucibleClient` |
| Crucible UI | Crucible Framework | `{:crucible_framework, path: "../crucible_framework"}` |
| Crucible Framework | Tinkex | Likely direct dependency or API calls |

### Environment Configuration

**CNS UI:**
```elixir
config :cns_ui,
  crucible_api: [
    url: System.get_env("CRUCIBLE_API_URL"),
    token: System.get_env("CRUCIBLE_API_TOKEN"),
    pubsub: CrucibleUI.PubSub
  ]
```

**Crucible UI:**
```elixir
config :crucible_ui,
  telemetry_source: :crucible_telemetry,
  refresh_interval: 1000
```

---

## Integration Opportunities with crucible_framework

### 1. Direct Tinkex Integration

**Current State:** Tinkex is a standalone SDK. CNS UI and Crucible UI access training via HTTP APIs.

**Opportunity:** Add `tinkex` as a direct dependency to `crucible_framework`:

```elixir
# crucible_framework/mix.exs
{:tinkex, path: "../tinkex"}  # or from hex.pm
```

**Benefits:**
- Direct GenServer-based training clients
- Task-based async workflows integrate with Crucible's OTP patterns
- ETS-based registries align with Crucible telemetry patterns
- Telemetry events can feed crucible_telemetry directly

### 2. Unified Experiment Orchestration

Create a `Crucible.Training` context that wraps Tinkex:

```elixir
defmodule Crucible.Training do
  @moduledoc """
  High-level training orchestration using Tinkex.
  """

  alias Tinkex.{ServiceClient, TrainingClient}

  def start_lora_run(experiment_id, config) do
    {:ok, service} = ServiceClient.start_link(config: tinkex_config())
    {:ok, trainer} = ServiceClient.create_lora_training_client(service,
      base_model: config.model,
      lora_config: %Tinkex.Types.LoraConfig{
        rank: config.lora_rank,
        alpha: config.lora_alpha
      }
    )
    # Register with crucible_telemetry
    # Return handle for run tracking
  end
end
```

### 3. Telemetry Bridge

Bridge Tinkex telemetry to crucible_telemetry:

```elixir
defmodule Crucible.Telemetry.TinkexBridge do
  def attach do
    :telemetry.attach_many(
      "crucible-tinkex-bridge",
      [
        [:tinkex, :http, :request, :stop],
        [:tinkex, :sampling, :complete],
        [:tinkex, :training, :forward_backward, :complete]
      ],
      &handle_event/4,
      nil
    )
  end

  def handle_event(event, measurements, metadata, _config) do
    CrucibleTelemetry.emit(
      experiment_id: metadata[:experiment_id],
      event: event,
      measurements: measurements
    )
  end
end
```

### 4. Shared Component Library

Extract shared UI components into a package:

```elixir
# crucible_ui_components/mix.exs
defmodule CrucibleUIComponents.MixProject do
  # Stat cards, progress bars, charts, etc.
end
```

**Benefits:**
- CNS UI stops vendoring components
- Consistent styling across all dashboards
- Easier maintenance

### 5. Training Pipeline DSL

Extend crucible_harness DSL for training experiments:

```elixir
defmodule MyTrainingExperiment do
  use Crucible.Harness

  experiment do
    name "LoRA Fine-tuning Study"

    training do
      backend :tinkex
      base_model "meta-llama/Llama-3.1-8B"

      lora rank: 32, alpha: 64, dropout: 0.1

      optimizer :adam, learning_rate: 1.0e-4

      dataset "path/to/data.jsonl"
      epochs 3
    end

    hypothesis "H1: LoRA improves task performance" do
      compare :lora_model, :baseline
      metric :accuracy
      test :paired_t_test
    end
  end
end
```

### 6. Checkpoint Integration with crucible_datasets

Use `crucible_datasets` to manage training data and checkpoints:

```elixir
defmodule Crucible.Datasets.TinkexAdapter do
  @behaviour Crucible.Datasets.Adapter

  def load_checkpoint(path) do
    Tinkex.CheckpointDownload.download(rest_client, path)
  end

  def save_checkpoint(run_id, weights_path) do
    # Store in crucible_datasets with versioning
  end
end
```

---

## Recommendations

### Short-term (1-2 weeks)

1. **Add Tinkex as crucible_framework dependency**
   - Enables direct training client usage
   - Aligns telemetry patterns

2. **Create telemetry bridge**
   - Tinkex events flow into crucible_telemetry
   - Unified experiment instrumentation

3. **Document integration patterns**
   - How to use Tinkex within Crucible experiments
   - Configuration examples

### Medium-term (1-2 months)

4. **Build `Crucible.Training` context**
   - High-level training orchestration
   - Integrates with ResearchHarness

5. **Extract shared UI components**
   - Publish `crucible_ui_components` to Hex
   - Update CNS UI to use package

6. **Extend harness DSL**
   - Training experiment definitions
   - Automatic Tinkex configuration

### Long-term (3+ months)

7. **Unified dashboard framework**
   - Pluggable domain panels (CNS, custom)
   - Shared experiment/run views
   - Domain-specific overlays

8. **Training workflow automation**
   - Dataset versioning via crucible_datasets
   - Checkpoint management
   - Experiment reproducibility

9. **Multi-model training support**
   - Ensemble training coordination
   - Hedged training for reliability
   - Cost optimization

---

## Architecture Insights

### Strengths

1. **Clear separation of concerns** - Each repo has distinct responsibility
2. **OTP patterns** - GenServers, supervisors, Tasks throughout
3. **Real-time updates** - PubSub for live dashboard updates
4. **Type safety** - Comprehensive typespecs in Tinkex
5. **Extensibility** - Overlay pattern for domain-specific UIs

### Considerations

1. **Path dependencies** - Creates tight coupling between repos
2. **Component duplication** - CNS UI vendors Crucible components
3. **Configuration complexity** - Multiple env vars across apps
4. **Testing integration** - Need for integration test infrastructure

### Security Notes

- API tokens for Crucible/Tinkex should use env vars
- PII detection in training data (consider LlmGuard integration)
- Checkpoint storage security

---

## Conclusion

The tinkex, crucible_ui, and cns_ui projects form a well-designed ecosystem for ML training and research visualization. Integration with crucible_framework should focus on:

1. **Tinkex as training backend** - Direct SDK integration
2. **Telemetry unification** - Bridge to crucible_telemetry
3. **Component standardization** - Shared UI library
4. **DSL extension** - Training experiment definitions

This will create a cohesive research platform where Crucible Framework orchestrates experiments, Tinkex handles ML training, and the UI projects provide visualization - all with unified telemetry and reproducibility.

---

## References

### Source Files Analyzed

**Tinkex:**
- `lib/tinkex.ex` - Main module
- `lib/tinkex/api/training.ex` - Training API
- `lib/tinkex/api/sampling.ex` - Sampling API
- `lib/tinkex/api/service.ex` - Service API
- `lib/tinkex/training_client.ex` - Training GenServer
- `lib/tinkex/service_client.ex` - Service GenServer

**Crucible UI:**
- `lib/crucible_ui/experiments.ex` - Experiments context
- `lib/crucible_ui/runs.ex` - Runs context
- `lib/crucible_ui_web/controllers/api/tinkex_job_controller.ex` - Tinkex API

**CNS UI:**
- `lib/cns_ui/crucible_client.ex` - Crucible HTTP client
- `lib/cns_ui/training.ex` - Training context
- `lib/cns_ui_web/live/training_live.ex` - Training wizard
- `lib/cns_ui_web/components/crucible_components.ex` - Shared components
- `docs/20251121/cns_overlay/ui_extension_plan.md` - Integration plan

### Documentation

- `tinkex/README.md` - SDK overview
- `crucible_ui/README.md` - Dashboard overview
- `cns_ui/README.md` - CNS UI overview
- `crucible_ui/docs/20251121/architecture.md` - Architecture docs
