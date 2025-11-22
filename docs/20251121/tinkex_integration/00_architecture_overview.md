# Crucible Framework Architecture Overview

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Executive Summary

This document describes the high-level architecture for integrating Tinkex (ML training/inference API) into the Crucible Framework, creating a unified "big fat API" for LLM reliability research.

## Dependency Graph

```
tinkex (ML backend)
    ^
    |
crucible_framework (unified API)
    ^
    |
    +-- cns (Chiral Narrative Synthesis)
    |
    +-- crucible_ui (web interface)
         ^
         |
         cns_ui (CNS web interface)
```

## Architectural Layers

### Layer 1: Runtime Foundation (Elixir/OTP)
- Process supervision and fault tolerance
- Concurrent execution with process isolation
- Hot code reloading for long experiments

### Layer 2: Data Management
- **crucible_datasets**: Unified interface to MMLU, HumanEval, GSM8K, SciFact, FEVER
- Automatic caching and version tracking
- Streaming support for large datasets

### Layer 3: Transparency
- **crucible_trace**: Causal decision provenance
- Event capture with timestamps
- Interactive visualization generation

### Layer 4: Reliability Strategies
- **crucible_ensemble**: Multi-model voting (4 strategies)
- **crucible_hedging**: Tail latency reduction (50-75% P99 improvement)
- **crucible_adversary**: Robustness testing (21 attack types)

### Layer 5: Analysis & Reporting
- **crucible_bench**: 15+ statistical tests, effect sizes, power analysis
- **crucible_telemetry**: Research-grade metrics collection
- Multi-format reporting (Markdown, LaTeX, HTML, Jupyter)

### Layer 6: Orchestration
- **crucible_harness**: Experiment DSL and automation
- **Crucible.Tinkex**: ML training/inference adapter

## Tinkex Integration Architecture

```
+------------------------------------------------------------------+
|                    Crucible Framework API                         |
+------------------------------------------------------------------+
|                                                                    |
|  Crucible.Lora                     Crucible.Ensemble              |
|  +----------------+                +------------------+            |
|  | create_exp()   |                | vote()           |            |
|  | train()        |                | configure()      |            |
|  | evaluate()     |                +------------------+            |
|  +-------+--------+                        |                       |
|          |                                 |                       |
|          v                                 v                       |
|  +----------------+                +------------------+            |
|  | Crucible.Tinkex|                | Voting Strategies|            |
|  | (Adapter)      |                +------------------+            |
|  +-------+--------+                                                |
|          |                                                         |
+----------|--------------------------------------------------------+
           |
           v
+------------------------------------------------------------------+
|                      Tinkex SDK                                   |
+------------------------------------------------------------------+
|  Tinkex.TrainingClient    |  Tinkex.SamplingClient               |
|  - forward_backward()     |  - generate()                        |
|  - optim_step()          |  - stream()                          |
|  - save_weights()        |                                       |
+------------------------------------------------------------------+
```

## Unified API Surface

### Training & Fine-tuning

```elixir
# High-level experiment workflow
{:ok, experiment} = Crucible.Lora.create_experiment(
  name: "SciFact Fine-tuning",
  config: %{
    base_model: "llama-3-8b",
    lora_rank: 16,
    learning_rate: 1.0e-4
  }
)

# Training with automatic checkpointing
{:ok, metrics} = Crucible.Lora.train(experiment, dataset,
  epochs: 5,
  batch_size: 8,
  checkpoint_every: 100,
  quality_targets: %{
    schema_compliance: 0.95,
    citation_accuracy: 0.95
  }
)
```

### Ensemble Inference

```elixir
# Create ensemble from multiple adapters
{:ok, ensemble} = Crucible.Ensemble.create(
  adapters: [
    %{name: "scifact-v1", weight: 0.4},
    %{name: "scifact-v2", weight: 0.3},
    %{name: "scifact-v3", weight: 0.3}
  ],
  strategy: :weighted_majority
)

# Run ensemble inference
{:ok, result} = Crucible.Ensemble.infer(ensemble, prompt,
  hedging: :percentile_75,
  timeout: 5000
)
```

### Statistical Analysis

```elixir
# Compare model variants
{:ok, analysis} = Crucible.Bench.compare(
  baseline: results_v1,
  treatment: results_v2,
  metrics: [:accuracy, :latency, :cost],
  tests: [:t_test, :mann_whitney],
  alpha: 0.05
)
```

## Component Integration Map

| Component | Provides | Consumes | Integration Point |
|-----------|----------|----------|-------------------|
| crucible_framework | Unified API | All components | Top-level module |
| tinkex | ML backend | HTTP, Config | Crucible.Tinkex adapter |
| crucible_ensemble | Voting | Inference results | Ensemble strategies |
| crucible_hedging | Latency opt | Request timing | Request dispatcher |
| crucible_bench | Statistics | Experiment data | Analysis pipeline |
| crucible_telemetry | Metrics | Events | Event bus |
| crucible_trace | Provenance | Decisions | Trace middleware |
| crucible_datasets | Data | None | Dataset loaders |
| crucible_harness | DSL | All components | Experiment runner |
| crucible_adversary | Robustness | Model access | Attack generators |

## Data Flow

### Training Flow

```
Dataset -> crucible_datasets.load()
       -> Crucible.Lora.batch_dataset()
       -> Crucible.Tinkex.format_training_data()
       -> Tinkex.TrainingClient.forward_backward()
       -> crucible_telemetry.emit()
       -> crucible_bench.analyze()
       -> Reporter.generate()
```

### Inference Flow

```
Prompt -> Crucible.Ensemble.infer()
      -> crucible_hedging.dispatch()
      -> Tinkex.SamplingClient.generate() [N adapters]
      -> crucible_trace.record()
      -> Crucible.Ensemble.vote()
      -> crucible_telemetry.emit()
```

## Configuration Architecture

```elixir
# Application config
config :crucible_framework,
  lora_adapter: Crucible.Tinkex,
  telemetry_backend: :ets,  # or :postgres
  default_hedging: :percentile_75

# Tinkex-specific config
config :crucible_framework, Crucible.Tinkex,
  api_key: {:system, "TINKEX_API_KEY"},
  base_url: "https://api.tinker.example.com",
  timeout: 60_000,
  pool_size: 10

# Per-experiment config
%Crucible.Tinkex.Config{
  base_model: "llama-3-8b",
  lora_config: %{rank: 16, alpha: 32},
  quality_targets: %{...}
}
```

## Error Handling Strategy

All components use tagged tuples with structured errors:

```elixir
@type result(t) :: {:ok, t} | {:error, Crucible.Error.t()}

defmodule Crucible.Error do
  defstruct [:type, :message, :data, :stacktrace]

  @type error_type :: :validation | :network | :timeout | :rate_limit | :internal
end
```

## Telemetry Events

Standard telemetry events across all components:

- `[:crucible, :training, :start | :stop | :exception]`
- `[:crucible, :inference, :start | :stop | :exception]`
- `[:crucible, :ensemble, :vote]`
- `[:crucible, :hedging, :dispatch]`
- `[:crucible, :checkpoint, :save | :load]`

## Next Steps

1. **01_tinkex_adapter.md**: Detailed Tinkex adapter implementation
2. **02_lora_training_interface.md**: LoRA fine-tuning abstractions
3. **03_ensemble_ml_integration.md**: ML-aware ensemble strategies
