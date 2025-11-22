# Crucible Ecosystem Integration Synthesis

**Date:** November 21, 2025
**Author:** Collector Agent
**Purpose:** Cross-domain integration brainstorms for design agents

---

## Executive Summary

This synthesis document consolidates research from five domain-specific reports analyzing 19+ North-Shore-AI repositories. The Crucible ecosystem represents a comprehensive AI/ML research infrastructure with significant integration opportunities across training, validation, security, fairness, and explainability domains.

**Key Finding:** The ecosystem is well-architected but fragmented. Unifying telemetry, configuration, and APIs would transform these standalone libraries into a cohesive research platform rivaling commercial offerings.

---

## 1. Cross-Cutting Themes

### 1.1 Common Patterns Across All Repos

| Pattern | Occurrences | Implementation |
|---------|-------------|----------------|
| **Telemetry Integration** | 14/19 repos | `:telemetry.execute/3` with varied namespaces |
| **Result Tuples** | All | `{:ok, result}` / `{:error, reason}` |
| **Keyword Options** | All | `opts \\ []` with defaults |
| **Type Specifications** | 16/19 | `@spec` on all public functions |
| **GenServer Clients** | 8/19 | Stateful clients (Tinkex, hedging, ensemble) |
| **JSON Serialization** | 15/19 | Jason for config/export |
| **Nx Tensors** | 4/19 | ExFairness, crucible_xai, crucible_bench, Tinkex |

### 1.2 Shared Dependencies

**Universal:**
- `jason ~> 1.4` - JSON encoding/decoding
- `ex_doc ~> 0.31` - Documentation generation

**Computation Layer:**
- `nx ~> 0.7` - ExFairness, crucible_xai, crucible_bench, Tinkex
- `statistex ~> 1.0` - crucible_bench, crucible_harness

**Observability:**
- `telemetry ~> 1.2-1.3` - Most runtime libraries

**Execution:**
- `gen_stage ~> 1.2` - crucible_harness
- `flow ~> 1.2` - crucible_harness
- `finch ~> 0.18` - Tinkex

### 1.3 Duplicate Functionality to Consolidate

#### Statistical Functions (CRITICAL)
Found in THREE places:
1. `CrucibleTelemetry.Analysis` - mean, median, percentiles, std_dev
2. `CrucibleHarness.Collector.StatisticalAnalyzer` - same functions
3. `CrucibleBench.Stats` - comprehensive statistical library

**Recommendation:** Extract to `CrucibleStats` shared module, all others delegate.

#### Prompt Injection Detection (HIGH)
Found in TWO places:
1. `LlmGuard.Detectors.PromptInjection` - 34 defensive patterns
2. `CrucibleAdversary.Attacks.Injection` - 4 attack generators

**Recommendation:** Shared pattern library, bidirectional red-team/blue-team testing.

#### Export Utilities (MEDIUM)
Found in FOUR places:
1. `CrucibleTelemetry.Export` - CSV, JSONL
2. `CrucibleTrace.Storage` - JSON, Markdown, CSV
3. `CrucibleBench.Export` - Markdown, LaTeX, HTML
4. `CrucibleHarness.Reporter` - All formats

**Recommendation:** Unified `CrucibleExport` module with consistent formatting.

#### Validation/Error Handling (MEDIUM)
- `LlmGuard.Config` - NimbleOptions validation
- `CrucibleHedging.Config` - NimbleOptions validation
- `ExFairness.Validation` - Custom validation
- `ExDataCheck` - Expectations-based validation

**Recommendation:** Shared validation behaviour with domain-specific implementations.

---

## 2. Integration Opportunities Matrix

### 2.1 Natural Connections

```
                    ┌─────────────────────────────────────┐
                    │         ORCHESTRATION               │
                    │    crucible_harness (experiments)   │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        │              │           │           │              │
        ▼              ▼           ▼           ▼              ▼
  ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐  ┌──────────┐
  │ TRAINING │  │   DATA   │ │   ML     │ │ ANALYSIS │  │   UI     │
  │  Tinkex  │  │ Datasets │ │ Ensemble │ │  Bench   │  │ Crucible │
  │  (LoRA)  │  │  (load)  │ │  (vote)  │ │  (stats) │  │   UI     │
  └────┬─────┘  └────┬─────┘ └────┬─────┘ └────┬─────┘  └────┬─────┘
       │             │            │            │              │
       └─────────────┴────────────┼────────────┴──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │        TELEMETRY           │
                    │   crucible_telemetry       │
                    │   (unified observation)    │
                    └───────────────────────────┘
```

### 2.2 Data Flow Paths

| Source | Target | Data Type | Integration Method |
|--------|--------|-----------|-------------------|
| crucible_datasets | crucible_harness | Dataset items | Direct function call |
| crucible_harness | crucible_ensemble | Query batches | Condition functions |
| crucible_ensemble | crucible_hedging | Prediction requests | Strategy delegation |
| crucible_harness | crucible_bench | Result metrics | Analysis pipeline |
| crucible_bench | crucible_harness.Reporter | Statistical results | Export formatting |
| Tinkex | crucible_telemetry | Training events | Telemetry bridge |
| LlmGuard | crucible_harness | Validated inputs | Pre-processing hook |
| crucible_adversary | LlmGuard | Attack payloads | Red-team testing |
| ExDataCheck | crucible_datasets | Validation results | Quality gates |
| ExFairness | crucible_bench | Fairness metrics | Metric category |
| crucible_xai | crucible_trace | Explanations | Decision provenance |

### 2.3 API Compatibility Analysis

#### Compatible (Direct Integration)
- `CrucibleBench.compare/2` ↔ `CrucibleHarness.StatisticalAnalyzer` (both use statistex)
- `ExDataCheck.validate/2` ↔ `CrucibleDatasets.load/2` (validator pattern)
- `Tinkex.TrainingClient` ↔ `Crucible.Lora` (both GenServer-based)

#### Requires Adaptation
- `LlmGuard.validate_input/2` returns `{:ok, input}` / `{:error, :detected, details}`
  - Non-standard error tuple needs wrapper
- `CrucibleAdversary.attack/2` returns `%AttackResult{}` struct
  - Need struct-to-map conversion for telemetry

#### Incompatible (Needs Redesign)
- Telemetry namespaces differ across all repos
  - `[:crucible_ensemble, :predict, :stop]`
  - `[:crucible_hedging, :request, :stop]`
  - `[:req_llm, :request, :stop]`
  - Need unified `[:crucible, :component, :action, :phase]` pattern

---

## 3. Architectural Brainstorms

### Brainstorm 1: Unified Telemetry System

**Problem:** 14 repos emit telemetry with inconsistent namespaces, metadata schemas, and storage patterns.

**Solution:** Create `CrucibleTelemetry.Unified` that:
1. Defines standard event schema:
```elixir
%{
  experiment_id: String.t(),
  component: :ensemble | :hedging | :training | :security | ...,
  action: :predict | :request | :validate | :train | ...,
  phase: :start | :stop | :exception,
  measurements: map(),
  metadata: map()
}
```

2. Provides namespace translator for backward compatibility
3. Auto-enriches events with experiment context
4. Routes to appropriate storage backends

**Estimated Effort:** 2 weeks
**Impact:** HIGH - Foundation for all other integrations

---

### Brainstorm 2: End-to-End Experiment Pipeline

**Problem:** Running a complete experiment requires manual coordination of datasets, ensemble, validation, analysis, and reporting.

**Solution:** Extend crucible_harness DSL:
```elixir
defmodule FullPipelineExperiment do
  use CrucibleHarness.Experiment

  experiment "Complete LLM Evaluation" do
    # Data layer
    dataset :mmlu, sample_size: 1000
    validate_data with: ExDataCheck, expectations: [...]

    # Security layer
    pre_process with: LlmGuard, config: %{...}

    # Execution layer
    conditions [
      %{name: "baseline", fn: &single_model/1},
      %{name: "ensemble", fn: &ensemble_vote/1}
    ]

    # Reliability layer
    hedging strategy: :percentile, percentile: 95

    # Analysis layer
    metrics [:accuracy, :latency_p99, :cost, :fairness]
    fairness_check ExFairness, sensitive_attrs: [:gender]
    statistical_tests CrucibleBench, alpha: 0.05

    # Robustness layer
    adversarial_test CrucibleAdversary, attacks: [:character_swap, :prompt_injection]

    # Explainability layer
    explain_with CrucibleXai, method: :lime

    # Output layer
    report formats: [:markdown, :latex, :html, :jupyter]
  end
end
```

**Estimated Effort:** 4-6 weeks
**Impact:** VERY HIGH - Complete research workflow automation

---

### Brainstorm 3: Security + Fairness + Quality Combined Workflow

**Problem:** These three concerns are typically evaluated separately, missing interactions.

**Solution:** Create `CrucibleFramework.ResponsibleAI` module:

```elixir
defmodule Crucible.ResponsibleAI do
  @moduledoc """
  Combined security, fairness, and data quality evaluation.
  """

  def evaluate(model, test_set, sensitive_attrs, opts \\ []) do
    # Phase 1: Data Quality
    quality = ExDataCheck.profile(test_set)
    quality_score = quality.quality_score

    # Phase 2: Security (input sanitization)
    {clean_inputs, security_report} = security_pass(test_set)

    # Phase 3: Model Evaluation
    predictions = Enum.map(clean_inputs, &model.(&1))

    # Phase 4: Fairness Analysis
    fairness = ExFairness.fairness_report(predictions, labels, sensitive_attrs)

    # Phase 5: Adversarial Robustness per demographic group
    robustness_by_group = evaluate_robustness_by_group(
      model, test_set, sensitive_attrs
    )

    %{
      data_quality: quality,
      security: security_report,
      fairness: fairness,
      robustness_equity: robustness_by_group,
      overall_score: compute_overall_score(...)
    }
  end
end
```

**Key Insight:** Adversarial robustness may differ across demographic groups - this is a fairness concern!

**Estimated Effort:** 3 weeks
**Impact:** HIGH - Novel research contribution

---

### Brainstorm 4: Training Integration with Experiment Harness

**Problem:** Tinkex training is separate from experiment orchestration.

**Solution:** Create `Crucible.Training` context:

```elixir
defmodule Crucible.Training do
  alias Tinkex.{ServiceClient, TrainingClient}

  def create_run(experiment_id, config) do
    # Create Tinkex clients
    {:ok, service} = ServiceClient.start_link(config: tinkex_config())
    {:ok, trainer} = ServiceClient.create_lora_training_client(service, ...)

    # Register with telemetry
    CrucibleTelemetry.register_training_run(experiment_id, trainer)

    # Return supervised process
    %TrainingRun{
      experiment_id: experiment_id,
      trainer: trainer,
      sampler: sampler
    }
  end

  def train_step(run, batch) do
    # Forward-backward pass
    {:ok, output} = TrainingClient.forward_backward(run.trainer, batch)

    # Emit telemetry
    emit_training_telemetry(run.experiment_id, output)

    # Optim step
    TrainingClient.optim_step(run.trainer, ...)
  end
end
```

**Estimated Effort:** 3 weeks
**Impact:** MEDIUM-HIGH - Enables LoRA experiments in harness

---

### Brainstorm 5: Red-Team/Blue-Team Automation

**Problem:** Security testing requires manual coordination between attack generation (crucible_adversary) and defense testing (LlmGuard).

**Solution:** Automated adversarial security testing:

```elixir
defmodule Crucible.Security.RedBlueTeam do
  @attack_types ~w(
    prompt_injection_basic prompt_injection_overflow prompt_injection_delimiter
    jailbreak_roleplay jailbreak_encode jailbreak_hypothetical
  )a

  def evaluate_defenses(guard_config, opts \\ []) do
    # Generate attack corpus
    base_inputs = opts[:inputs] || generate_benign_inputs(100)

    attacks = Enum.flat_map(@attack_types, fn type ->
      {:ok, results} = CrucibleAdversary.attack_batch(base_inputs, type: type)
      Enum.map(results, &{type, &1.attacked})
    end)

    # Test defenses
    results = Enum.map(attacks, fn {type, payload} ->
      case LlmGuard.validate_input(payload, guard_config) do
        {:ok, _} -> {:bypassed, type, payload}
        {:error, :detected, details} -> {:blocked, type, details}
      end
    end)

    # Calculate metrics
    %{
      total_attacks: length(attacks),
      blocked: count_blocked(results),
      bypassed: count_bypassed(results),
      by_attack_type: group_by_type(results),
      defense_rate: blocked / total,
      recommendations: generate_recommendations(results)
    }
  end

  def continuous_testing(guard_config, interval_ms \\ 3600_000) do
    # Periodic testing with new attack variations
  end
end
```

**Estimated Effort:** 2 weeks
**Impact:** HIGH - Quantified security posture

---

### Brainstorm 6: Unified Configuration System

**Problem:** Each repo has its own configuration approach (NimbleOptions, custom structs, keyword lists).

**Solution:** Create `CrucibleConfig` with schema inheritance:

```elixir
defmodule CrucibleConfig do
  use NimbleOptions

  @base_schema [
    experiment_id: [type: :string],
    timeout_ms: [type: :pos_integer, default: 30_000],
    telemetry_prefix: [type: {:list, :atom}, default: [:crucible]]
  ]

  @ensemble_schema @base_schema ++ [
    models: [type: {:list, :atom}, required: true],
    strategy: [type: {:in, [:majority, :weighted, :unanimous]}, default: :majority]
  ]

  @hedging_schema @base_schema ++ [
    strategy: [type: {:in, [:fixed, :percentile, :adaptive]}, default: :percentile],
    percentile: [type: :pos_integer, default: 95]
  ]

  # Nested configuration
  @experiment_schema [
    ensemble: [type: {:map, @ensemble_schema}],
    hedging: [type: {:map, @hedging_schema}],
    security: [type: {:map, @security_schema}],
    fairness: [type: {:map, @fairness_schema}]
  ]

  def validate(config, schema \\ @experiment_schema) do
    NimbleOptions.validate(config, schema)
  end
end
```

**Estimated Effort:** 2 weeks
**Impact:** MEDIUM - Consistent configuration across ecosystem

---

### Brainstorm 7: Shared Error Handling

**Problem:** Different error representations across repos.

**Solution:** Standardized error module:

```elixir
defmodule CrucibleError do
  defexception [:code, :message, :details, :source, :recoverable]

  @type t :: %__MODULE__{
    code: atom(),
    message: String.t(),
    details: map(),
    source: module(),
    recoverable: boolean()
  }

  # Error categories
  def validation_error(message, details \\ %{}) do
    %__MODULE__{code: :validation_error, message: message, details: details, recoverable: true}
  end

  def security_threat(message, details) do
    %__MODULE__{code: :security_threat, message: message, details: details, recoverable: false}
  end

  def timeout_error(component, elapsed_ms) do
    %__MODULE__{code: :timeout, message: "#{component} timed out", details: %{elapsed: elapsed_ms}, recoverable: true}
  end
end
```

All components return `{:ok, result} | {:error, %CrucibleError{}}`.

**Estimated Effort:** 1 week
**Impact:** MEDIUM - Better debugging and error recovery

---

### Brainstorm 8: Plugin/Extension Architecture

**Problem:** Adding new capabilities requires modifying core modules.

**Solution:** Behaviour-based plugin system:

```elixir
defmodule Crucible.Plugin do
  @callback name() :: atom()
  @callback version() :: String.t()
  @callback init(config :: map()) :: {:ok, state} | {:error, term()}
  @callback handle_event(event :: atom(), data :: map(), state) :: {:ok, state}
  @callback cleanup(state) :: :ok
end

defmodule Crucible.PluginManager do
  use GenServer

  def register(plugin_module, config) do
    GenServer.call(__MODULE__, {:register, plugin_module, config})
  end

  def broadcast_event(event, data) do
    GenServer.cast(__MODULE__, {:broadcast, event, data})
  end
end

# Example plugins
defmodule Crucible.Plugins.SlackNotifier do
  @behaviour Crucible.Plugin

  def handle_event(:experiment_complete, data, state) do
    Slack.post_message(state.channel, format_results(data))
    {:ok, state}
  end
end

defmodule Crucible.Plugins.MLflowExporter do
  @behaviour Crucible.Plugin

  def handle_event(:run_complete, data, state) do
    MLflow.log_metrics(state.tracking_uri, data.metrics)
    {:ok, state}
  end
end
```

**Estimated Effort:** 3 weeks
**Impact:** MEDIUM-HIGH - Extensibility without core changes

---

### Brainstorm 9: CLI Unification

**Problem:** Multiple CLIs (tinkex, potential crucible CLI) with different patterns.

**Solution:** Unified `crucible` CLI:

```bash
# Training
crucible train --model llama-3.1-8b --dataset data.jsonl --lora-rank 32

# Experiments
crucible experiment run my_experiment.exs
crucible experiment status <experiment_id>
crucible experiment report <experiment_id> --format latex

# Security
crucible security test --config guard.yml --attacks all
crucible security scan input.txt

# Analysis
crucible analyze compare results_a.json results_b.json
crucible analyze fairness predictions.json --sensitive gender

# Datasets
crucible dataset download mmlu --sample 1000
crucible dataset validate data.jsonl --expectations expectations.yml

# UI
crucible ui start --port 4000
```

Implementation via `Burrito` or `Bakeware` for standalone binary.

**Estimated Effort:** 4 weeks
**Impact:** MEDIUM - Better developer experience

---

### Brainstorm 10: Dashboard Consolidation

**Problem:** crucible_ui and cns_ui are separate Phoenix apps with duplicated components.

**Solution:** Pluggable dashboard architecture:

```elixir
# Core dashboard provides base views
defmodule CrucibleUI.Dashboard do
  use Phoenix.LiveView

  def mount(_params, _session, socket) do
    # Load registered panels
    panels = Crucible.Dashboard.list_panels()
    {:ok, assign(socket, panels: panels)}
  end
end

# Domain-specific panels register themselves
defmodule CnsUI.Panels.SNOExplorer do
  use Crucible.Dashboard.Panel

  @impl true
  def panel_info do
    %{
      name: "SNO Explorer",
      icon: "document-text",
      route: "/sno",
      category: :research
    }
  end
end

# Plugin registration
config :crucible_ui, :panels, [
  CrucibleUI.Panels.Experiments,
  CrucibleUI.Panels.Runs,
  CrucibleUI.Panels.Statistics,
  CnsUI.Panels.SNOExplorer,
  CnsUI.Panels.ChiralityGraph
]
```

**Estimated Effort:** 6 weeks
**Impact:** HIGH - Unified research dashboard

---

### Brainstorm 11: Distributed Execution Layer

**Problem:** Large experiments need multi-node execution.

**Solution:** Use libcluster + Horde for distributed orchestration:

```elixir
defmodule Crucible.Distributed.Coordinator do
  use Horde.DynamicSupervisor

  def distribute_tasks(tasks, opts) do
    # Partition tasks across cluster
    nodes = Node.list() ++ [node()]
    partitions = partition_tasks(tasks, nodes)

    # Start workers on each node
    Enum.flat_map(partitions, fn {node, node_tasks} ->
      :rpc.call(node, __MODULE__, :start_local_workers, [node_tasks, opts])
    end)
  end
end

defmodule Crucible.Distributed.TelemetryAggregator do
  @moduledoc "Aggregate telemetry from all nodes"

  def aggregate(experiment_id) do
    nodes = Node.list() ++ [node()]

    results = Enum.map(nodes, fn node ->
      :rpc.call(node, CrucibleTelemetry.Store.ETS, :get_events, [experiment_id])
    end)

    merge_results(results)
  end
end
```

**Estimated Effort:** 8 weeks
**Impact:** HIGH - Scale to large experiments

---

### Brainstorm 12: Reproducibility Framework

**Problem:** Experiments need to be fully reproducible for publication.

**Solution:** Comprehensive provenance tracking:

```elixir
defmodule Crucible.Reproducibility do
  def capture_environment do
    %{
      elixir_version: System.version(),
      otp_version: :erlang.system_info(:otp_release),
      dependencies: get_locked_deps(),
      system: %{
        os: :os.type(),
        cpu: System.schedulers_online(),
        memory: :erlang.memory(:total)
      },
      git: %{
        commit: get_git_commit(),
        branch: get_git_branch(),
        dirty: git_dirty?()
      },
      timestamp: DateTime.utc_now(),
      random_seed: :rand.export_seed()
    }
  end

  def create_manifest(experiment) do
    %{
      environment: capture_environment(),
      config: experiment.config,
      dataset: %{
        name: experiment.dataset.name,
        version: experiment.dataset.version,
        checksum: experiment.dataset.checksum
      },
      model_checkpoints: experiment.checkpoints
    }
  end

  def reproduce(manifest_path) do
    manifest = load_manifest(manifest_path)
    verify_environment(manifest.environment)
    restore_random_seed(manifest.environment.random_seed)
    # Re-run experiment
  end
end
```

**Estimated Effort:** 3 weeks
**Impact:** HIGH - Publication-ready experiments

---

## 4. Dependency Graph

### 4.1 Proposed Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER 6: APPLICATIONS                       │
│  crucible_ui    cns_ui    crucible_examples    CLI                 │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────┐
│                       LAYER 5: ORCHESTRATION                        │
│                     crucible_harness (DSL)                          │
│                     crucible_framework (facade)                     │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
┌──────────────┬──────────────┬───────────▼─────┬─────────────────────┐
│   LAYER 4:   │   LAYER 4:   │    LAYER 4:     │      LAYER 4:       │
│  EXECUTION   │   ANALYSIS   │    SECURITY     │      QUALITY        │
├──────────────┼──────────────┼─────────────────┼─────────────────────┤
│ ensemble     │ crucible_    │ LlmGuard        │ ExDataCheck         │
│ hedging      │   bench      │ crucible_       │ ExFairness          │
│ Tinkex       │ crucible_xai │   adversary     │                     │
└──────┬───────┴──────┬───────┴────────┬────────┴──────────┬──────────┘
       │              │                │                   │
┌──────▼──────────────▼────────────────▼───────────────────▼──────────┐
│                         LAYER 3: DATA                               │
│  crucible_datasets    crucible_trace                                │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────┐
│                      LAYER 2: OBSERVABILITY                         │
│                     crucible_telemetry                              │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────┐
│                         LAYER 1: CORE                               │
│  CrucibleConfig  CrucibleError  CrucibleStats  CrucibleExport       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Dependency Matrix

| Repository | Depends On | Depended By |
|------------|------------|-------------|
| **CrucibleCore** (new) | - | ALL |
| crucible_telemetry | CrucibleCore | harness, framework, datasets |
| crucible_trace | CrucibleCore, telemetry | xai, harness |
| crucible_datasets | CrucibleCore, telemetry | harness, framework |
| crucible_bench | CrucibleCore, nx, statistex | harness, framework |
| crucible_ensemble | CrucibleCore, telemetry | harness, framework |
| crucible_hedging | CrucibleCore, telemetry | ensemble, framework |
| Tinkex | CrucibleCore, nx, finch | framework |
| LlmGuard | CrucibleCore, telemetry | harness, framework |
| crucible_adversary | CrucibleCore | harness, framework |
| ExDataCheck | CrucibleCore, jason | harness, datasets, framework |
| ExFairness | CrucibleCore, nx | harness, bench, framework |
| crucible_xai | CrucibleCore, nx, trace | harness, framework |
| crucible_harness | ALL Layer 3-4 | framework |
| crucible_framework | harness + optional components | UI apps |
| crucible_ui | framework | cns_ui |
| cns_ui | crucible_ui, framework | - |

### 4.3 Package Publishing Strategy

**Phase 1: Core Foundation**
1. `crucible_core` - Shared utilities (MUST publish first)
2. `crucible_telemetry` - Observation layer

**Phase 2: Data Layer**
3. `crucible_datasets` - Benchmark management
4. `crucible_trace` - Decision provenance
5. `ex_data_check` - Data validation

**Phase 3: Computation Layer**
6. `crucible_bench` - Statistical analysis
7. `crucible_ensemble` - Multi-model voting
8. `crucible_hedging` - Tail latency
9. `ex_fairness` - Bias detection

**Phase 4: Security/XAI**
10. `llm_guard` - AI firewall
11. `crucible_adversary` - Attack generation
12. `crucible_xai` - Explainability

**Phase 5: Integration**
13. `crucible_harness` - Orchestration
14. `tinkex` - Training SDK
15. `crucible_framework` - Meta-package

---

## 5. Key Questions for Design Phase

### Architecture Questions

1. **Mono-repo vs Multi-repo**: Should crucible_framework become a mono-repo umbrella, or maintain separate packages with strict versioning?

2. **Optional vs Required Dependencies**: Which integrations should be required vs optional? (Currently all optional)

3. **Nx Version Strategy**: How to handle Nx version constraints across ExFairness, crucible_xai, crucible_bench, and Tinkex?

4. **OTP Application Boundaries**: Should crucible_ensemble and crucible_hedging start OTP applications, or remain library-only?

5. **Configuration Source of Truth**: Who owns experiment configuration - harness or framework?

### Integration Questions

6. **Telemetry Event Schema**: What's the canonical event structure that all components must emit?

7. **Error Propagation**: How should errors bubble up through the layer stack?

8. **State Management**: Should there be a global experiment state supervisor?

9. **Async Boundaries**: Where are the async/sync boundaries in the pipeline?

10. **Testing Strategy**: Integration tests across packages - how to structure?

### API Design Questions

11. **Facade vs Direct Access**: Should users always go through `Crucible.*` facades, or can they use underlying libraries directly?

12. **Result Types**: Standardize on structs (`%Result{}`) or maps?

13. **Streaming Support**: How to propagate streaming through the pipeline (ensemble → hedging → telemetry)?

14. **Callback/Hook System**: How to allow user-defined callbacks at various pipeline stages?

15. **Configuration Inheritance**: Should component configs inherit from experiment config?

### Security Questions

16. **PII Handling**: How to handle PII in telemetry events?

17. **Credential Management**: Unified approach for API keys (Tinkex, LLM providers)?

18. **Audit Logging**: What events require audit-level logging?

### Performance Questions

19. **ETS vs PostgreSQL**: When to use ETS (speed) vs PostgreSQL (persistence)?

20. **Connection Pooling**: Shared pool for HTTP requests across components?

21. **Batch Size Strategy**: How to optimize batch sizes across ensemble/training/evaluation?

### UX Questions

22. **Progressive Disclosure**: How to make the framework approachable while exposing power user features?

23. **Error Messages**: Consistent error message format across ecosystem?

24. **Documentation Structure**: Unified documentation site or per-package?

25. **Migration Paths**: How to migrate existing users when APIs change?

---

## 6. Implementation Priority Matrix

| Initiative | Impact | Effort | Priority | Dependencies |
|------------|--------|--------|----------|--------------|
| Unified Telemetry | HIGH | 2 weeks | **P0** | None |
| CrucibleCore Package | HIGH | 1 week | **P0** | None |
| Shared Statistics | MEDIUM | 1 week | **P1** | CrucibleCore |
| Red-Team/Blue-Team | HIGH | 2 weeks | **P1** | None |
| Training Integration | HIGH | 3 weeks | **P1** | Telemetry |
| Full Pipeline DSL | VERY HIGH | 6 weeks | **P2** | All P1 |
| ResponsibleAI Module | HIGH | 3 weeks | **P2** | Fairness, Security |
| Unified Config | MEDIUM | 2 weeks | **P2** | CrucibleCore |
| CLI Unification | MEDIUM | 4 weeks | **P3** | P1-P2 complete |
| Dashboard Plugins | HIGH | 6 weeks | **P3** | P1-P2 complete |
| Distributed Execution | HIGH | 8 weeks | **P3** | P2 complete |
| Reproducibility | HIGH | 3 weeks | **P2** | Telemetry |

---

## 7. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Nx version conflicts | HIGH | HIGH | Pin to specific Nx version, test matrix |
| Telemetry migration breaks existing users | MEDIUM | HIGH | Backward-compatible adapter layer |
| Performance regression from abstraction | MEDIUM | MEDIUM | Benchmarking suite, optional bypasses |
| OTP supervision tree complexity | LOW | MEDIUM | Clear documentation, example configurations |

### Process Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep in integration | HIGH | MEDIUM | Strict phase boundaries, MVP focus |
| API churn during design | MEDIUM | HIGH | Design review gates, deprecation policy |
| Documentation lag | HIGH | MEDIUM | Doc-driven development, auto-generation |

---

## 8. Success Metrics

### Integration Success
- [ ] All components emit telemetry to unified namespace
- [ ] Single `crucible_framework` dependency provides full functionality
- [ ] End-to-end experiment can be defined in <50 lines of DSL

### Quality Success
- [ ] >95% test coverage on integration code
- [ ] Zero compilation warnings
- [ ] All public APIs have typespecs and docs

### Adoption Success
- [ ] Example notebooks for all major workflows
- [ ] Migration guides from standalone libraries
- [ ] Community feedback incorporated in design

---

## Conclusion

The Crucible ecosystem represents a significant opportunity to create the definitive Elixir platform for AI/ML research. The individual components are well-designed and production-ready; integration is the key to unlocking their full potential.

**Immediate priorities:**
1. Establish unified telemetry (foundation for everything)
2. Create CrucibleCore shared package (eliminate duplication)
3. Build red-team/blue-team automation (high value, moderate effort)
4. Integrate Tinkex training (enable training experiments)

**Design agents should focus on:**
- Telemetry event schema standardization
- Error handling and propagation patterns
- Configuration inheritance model
- Testing strategy across packages

With coordinated effort, the ecosystem can evolve from "collection of excellent libraries" to "unified research platform" - a compelling alternative to Python-based tooling for the Elixir ML community.

---

*End of Synthesis Document*
