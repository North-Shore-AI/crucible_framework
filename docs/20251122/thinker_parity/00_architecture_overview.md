# Thinker Parity Architecture Overview

## Purpose

Achieve parity with tinkerer/thinker experiments in pure Elixir using the crucible ecosystem.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Crucible Framework                    │
├─────────────────────────────────────────────────────────┤
│  crucible_harness (Experiment DSL & Orchestration)      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Training   │  │ Validation  │  │   Antagonist    │  │
│  │             │  │             │  │                 │  │
│  │ Lora.Config │  │ Schema      │  │ CNS.Antagonist  │  │
│  │ Lora.Loop   │  │ Citation    │  │ (Quality flags) │  │
│  │ Tinkex.API  │  │ Entailment  │  │                 │  │
│  │             │  │ Similarity  │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  crucible_telemetry    │  crucible_bench               │
│  (Instrumentation)      │  (Statistical Analysis)       │
├─────────────────────────────────────────────────────────┤
│  crucible_datasets  │  ExDataCheck  │  ExFairness      │
│  (SciFact loader)   │  (Validation) │  (Fairness)      │
└─────────────────────────────────────────────────────────┘
```

## Crucible Library Integration

| Library | Role in Thinker Parity |
|---------|------------------------|
| crucible_telemetry | All training/validation events, metrics storage |
| crucible_datasets | SciFact loader (requires custom adapter) |
| crucible_bench | Statistical analysis of eval results |
| crucible_harness | Experiment definition DSL |
| ExDataCheck | Dataset validation, quality gates |
| ExFairness | Fairness metrics on model outputs |

## Module Structure

```elixir
Crucible.Thinker
├── Datasets.SciFact          # SciFact loader (via crucible_datasets)
├── Lora
│   ├── Config                # LoRA hyperparameters
│   └── TrainingLoop          # Training orchestration
├── Validation
│   ├── Schema                # CLAIM[c*] structure
│   ├── Citation              # Corpus lookup
│   ├── Entailment            # NLI via Tinkex (→ Bumblebee)
│   └── Similarity            # Embeddings via Tinkex (→ Bumblebee)
└── CNS.Antagonist            # Quality issue flagging
```

## Data Flow

```
SciFact Dataset
    │
    ▼
ExDataCheck.validate()  ─── Validation Gate
    │
    ▼
Lora.TrainingLoop.run()
    │
    ├─── Tinkex.train() calls
    │
    ├─── crucible_telemetry events
    │
    ▼
Validation Pipeline
    │
    ├─── Schema.check()
    ├─── Citation.verify()
    ├─── Entailment.score() ─── Tinkex API (→ Bumblebee)
    └─── Similarity.score() ─── Tinkex API (→ Bumblebee)
    │
    ▼
CNS.Antagonist.analyze()
    │
    ▼
crucible_bench.analyze()
    │
    ▼
Report (Markdown/JSON)
```

## Initial vs Future

### Phase 1: Tinkex-Based (Current)
- NLI via `POST /v1/predict/nli`
- Embeddings via `POST /v1/predict/embed`
- Training via `POST /v1/train`

### Phase 2: Bumblebee-Based (Future)
- NLI: `Bumblebee.load_model({:hf, "microsoft/deberta-v3-large-mnli"})`
- Embeddings: `Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})`
- Training: `Axon.Training` with custom loss functions

## Success Criteria

- 95% schema compliance (CLAIM[c*] format)
- 95% citation accuracy (corpus lookup)
- 50% mean entailment score
- Reproducible experiments via crucible_harness
- Full telemetry capture via crucible_telemetry
