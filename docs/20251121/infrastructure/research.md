# Crucible Infrastructure Components Research Report

**Date:** November 21, 2025
**Author:** Research Agent
**Purpose:** Deep investigation of infrastructure libraries for integration into crucible_framework

---

## Executive Summary

This report analyzes four critical infrastructure components of the Crucible ecosystem:

1. **crucible_telemetry** - Research-grade instrumentation and metrics collection
2. **crucible_trace** - Causal reasoning chain logging for LLM transparency
3. **crucible_datasets** - Unified dataset management for benchmarks
4. **crucible_harness** - Automated experiment orchestration

All four libraries are well-architected, maintain consistent patterns, and are designed for seamless integration. Together they form a complete research infrastructure stack for AI/ML experimentation in Elixir.

---

## 1. CrucibleTelemetry

### Overview
Research-grade instrumentation and metrics collection for AI/ML experiments, designed specifically for rigorous scientific experimentation rather than production monitoring.

**Version:** 0.1.0
**Dependencies:** `jason ~> 1.4`, `telemetry ~> 1.3`

### Key Differentiators from Production Telemetry
- Complete event capture (no sampling by default) for reproducibility
- Experiment isolation for concurrent A/B testing
- Rich metadata enrichment with experiment context
- Statistical analysis and export to data science tools

### Architecture

```
CrucibleTelemetry
├── CrucibleTelemetry.Experiment     # Lifecycle management
├── CrucibleTelemetry.Handler        # Event collection pipeline
├── CrucibleTelemetry.Store          # Multi-backend storage (ETS)
│   └── Store.ETS                    # In-memory ETS backend
├── CrucibleTelemetry.Export         # Format converters
│   ├── Export.CSV
│   └── Export.JSONL
└── CrucibleTelemetry.Analysis       # Statistical analysis
```

### Core Data Structures

#### Experiment Struct
```elixir
%CrucibleTelemetry.Experiment{
  id: String.t(),
  name: String.t(),
  hypothesis: String.t() | nil,
  condition: String.t(),           # "treatment", "control"
  metadata: map(),
  tags: list(String.t()),
  started_at: DateTime.t(),
  stopped_at: DateTime.t() | nil,
  status: :running | :stopped | :archived,
  sample_size: integer() | nil,
  metrics_config: map(),
  storage_backend: :ets | :postgres
}
```

#### Enriched Event Structure
```elixir
%{
  event_id: String.t(),
  event_name: list(atom()),          # e.g., [:req_llm, :request, :stop]
  timestamp: integer(),              # microseconds
  experiment_id: String.t(),
  experiment_name: String.t(),
  condition: String.t(),
  tags: list(String.t()),
  measurements: map(),
  metadata: map(),
  # Computed fields
  latency_ms: float() | nil,
  cost_usd: float() | nil,
  success: boolean() | nil,
  model: String.t() | nil,
  provider: String.t() | nil
}
```

### Key APIs

#### Main Module
- `start_experiment(opts)` - Start isolated experiment with handlers
- `stop_experiment(experiment_id)` - Finalize data collection
- `get_experiment(experiment_id)` - Retrieve experiment details
- `list_experiments()` - List all experiments
- `export(experiment_id, format, opts)` - Export to CSV/JSONL
- `calculate_metrics(experiment_id)` - Comprehensive metrics

#### Analysis Module
Returns metrics organized by category:
```elixir
%{
  summary: %{total_events, time_range, duration_seconds, event_types},
  latency: %{mean, median, p50, p90, p95, p99, std_dev, min, max},
  cost: %{total, mean_per_request, cost_per_1k_requests, cost_per_1m_requests},
  reliability: %{success_rate, failure_rate, sla_99, sla_999},
  tokens: %{total_prompt, total_completion, mean_total}
}
```

### Telemetry Events Captured
- `[:req_llm, :request, :start|stop|exception]` - LLM API calls
- `[:ensemble, :prediction, :start|stop]` - Ensemble predictions
- `[:ensemble, :vote, :completed]` - Voting results
- `[:hedging, :request, :*]` - Request hedging events
- `[:causal_trace, :event, :created]` - Reasoning traces
- `[:altar, :tool, :*]` - Tool invocations

### Integration Points
1. **With crucible_harness:** Runner emits telemetry events automatically captured
2. **With crucible_ensemble:** Ensemble events tracked for accuracy analysis
3. **With crucible_trace:** CausalTrace events captured for provenance

### Cost Calculation
Built-in pricing for major models:
- Gemini 2.0 Flash: $0.075/$0.30 per 1M tokens
- GPT-4: $30/$60 per 1M tokens
- Claude 3 Opus: $15/$75 per 1M tokens

---

## 2. CrucibleTrace

### Overview
Structured causal reasoning chain logging for LLM code generation, enabling transparency and debugging by capturing decision-making processes.

**Version:** 0.1.0
**Dependencies:** `jason ~> 1.4`

### Key Features
- Event tracking with alternatives and reasoning
- LLM output parsing with XML tags
- Interactive HTML visualization
- Multiple export formats (JSON, Markdown, CSV)
- Chain statistics and analysis

### Architecture

```
CrucibleTrace
├── CrucibleTrace.Event      # Event struct and operations
├── CrucibleTrace.Chain      # Chain collection management
├── CrucibleTrace.Parser     # LLM output parsing
├── CrucibleTrace.Storage    # Persistence and retrieval
└── CrucibleTrace.Viewer     # HTML visualization
```

### Core Data Structures

#### Event Types
Six predefined event types for reasoning stages:
- `:hypothesis_formed` - Initial approach or solution
- `:alternative_rejected` - Explicit rejection of alternatives
- `:constraint_evaluated` - Evaluation of requirements
- `:pattern_applied` - Design pattern application
- `:ambiguity_flagged` - Specification ambiguities
- `:confidence_updated` - Confidence level changes

#### Event Struct
```elixir
%CrucibleTrace.Event{
  id: String.t(),
  timestamp: DateTime.t(),
  type: atom(),                    # Event type
  decision: String.t(),            # What was decided
  alternatives: [String.t()],      # Options considered
  reasoning: String.t(),           # Why this decision
  confidence: float(),             # 0.0 to 1.0
  code_section: String.t() | nil,
  spec_reference: String.t() | nil,
  metadata: map()
}
```

#### Chain Struct
```elixir
%CrucibleTrace.Chain{
  id: String.t(),
  name: String.t(),
  events: [Event.t()],
  created_at: DateTime.t(),
  metadata: map()
}
```

### Key APIs

#### Chain Operations
- `new_chain(name, opts)` - Create empty chain
- `add_event(chain, event)` - Add single event
- `add_events(chain, events)` - Add multiple events
- `merge_chains(chain1, chain2)` - Combine chains

#### Event Operations
- `create_event(type, decision, reasoning, opts)` - Create new event
- `validate_event(event)` - Validate event structure

#### LLM Integration
- `parse_llm_output(text, chain_name, opts)` - Extract events from LLM response
- `extract_code(text)` - Get clean code without event tags
- `build_causal_prompt(base_spec)` - Generate instrumented prompt

#### Storage
- `save(chain, opts)` - Persist to disk (JSON)
- `load(chain_id, opts)` - Load by ID
- `list_chains(opts)` - List all saved chains
- `search(query, opts)` - Search with filters
- `export(chain, format, opts)` - Export to JSON/Markdown/CSV

#### Analysis
- `statistics(chain)` - Event counts, avg confidence, duration
- `find_decision_points(chain)` - Events with alternatives
- `find_low_confidence(chain, threshold)` - Below threshold
- `get_events_by_type(chain, type)` - Filter by type

### XML Tag Format for LLM
```xml
<event type="hypothesis_formed">
  <decision>Use GenServer for state management</decision>
  <alternatives>Agent, ETS table, Database</alternatives>
  <reasoning>GenServer provides good balance of simplicity and features</reasoning>
  <confidence>0.85</confidence>
  <code_section>StateManager</code_section>
</event>
```

### Integration Points
1. **With crucible_telemetry:** Events emit `[:causal_trace, :event, :created]` telemetry
2. **With crucible_harness:** Track reasoning in experiment conditions
3. **Direct LLM integration:** Parse responses automatically

---

## 3. CrucibleDatasets

### Overview
Centralized dataset management library providing a unified interface for loading, caching, evaluating, and sampling AI benchmark datasets.

**Version:** 0.1.0
**Dependencies:** `jason ~> 1.4`, `telemetry ~> 1.3`

### Supported Datasets
- **MMLU** - Massive Multitask Language Understanding (57 subjects)
- **HumanEval** - Code generation (164 problems)
- **GSM8K** - Grade school math (8,500 problems)
- **Custom** - Load from local JSONL files

### Architecture

```
CrucibleDatasets
├── CrucibleDatasets.Dataset           # Dataset schema
├── CrucibleDatasets.EvaluationResult  # Result schema
├── CrucibleDatasets.Loader            # Dataset loading
│   ├── Loader.MMLU
│   ├── Loader.HumanEval
│   └── Loader.GSM8K
├── CrucibleDatasets.Cache             # Local caching
├── CrucibleDatasets.Evaluator         # Evaluation engine
│   ├── Evaluator.ExactMatch
│   └── Evaluator.F1
└── CrucibleDatasets.Sampler           # Sampling utilities
```

### Core Data Structures

#### Dataset Struct
```elixir
%CrucibleDatasets.Dataset{
  name: String.t(),
  version: String.t(),
  items: [
    %{
      id: String.t(),
      input: map(),           # Question/problem data
      expected: any(),        # Ground truth
      metadata: map()         # Subject, difficulty, etc.
    }
  ],
  metadata: %{
    source: String.t(),
    license: String.t(),
    domain: String.t(),
    total_items: integer(),
    loaded_at: DateTime.t(),
    checksum: String.t()
  }
}
```

#### Prediction Format
```elixir
%{
  id: String.t(),         # Must match dataset item ID
  predicted: any(),       # Model's prediction
  metadata: %{            # Optional
    latency_ms: integer(),
    confidence: float(),
    tokens_used: integer()
  }
}
```

#### EvaluationResult Struct
```elixir
%CrucibleDatasets.EvaluationResult{
  dataset: String.t(),
  version: String.t(),
  model: String.t(),
  accuracy: float(),
  metrics: %{exact_match: float(), f1: float()},
  item_results: [map()],
  duration_ms: integer()
}
```

### Key APIs

#### Loading
- `load(dataset_name, opts)` - Load with optional sampling
  - Options: `:sample_size`, `:source` (for custom)

#### Evaluation
- `evaluate(predictions, opts)` - Evaluate predictions
  - Options: `:dataset`, `:metrics`, `:model_name`
- `evaluate_batch(model_predictions, opts)` - Compare multiple models

#### Sampling
- `random_sample(dataset, opts)` - Random sampling
  - Options: `:size`, `:seed`
- `stratified_sample(dataset, opts)` - Maintain distributions
  - Options: `:size`, `:strata_field`
- `train_test_split(dataset, opts)` - Train/test split
  - Options: `:test_size`, `:shuffle`, `:seed`
- `k_fold(dataset, opts)` - Cross-validation folds
  - Options: `:k`, `:shuffle`, `:seed`

#### Cache Management
- `list_cached()` - List cached datasets
- `clear_cache()` - Clear all cache
- `invalidate_cache(dataset_name)` - Clear specific dataset

### Evaluation Metrics

#### Exact Match
- Case-insensitive comparison
- Whitespace normalization
- Numerical tolerance
- Type coercion (string <-> number)

#### F1 Score
- Token-level precision and recall
- Word boundary tokenization

### Integration Points
1. **With crucible_harness:** Primary data source for experiments
2. **With crucible_telemetry:** Emit metrics via telemetry
3. **Standalone usage:** Direct evaluation workflows

---

## 4. CrucibleHarness

### Overview
Automated experiment orchestration framework providing declarative DSL for defining, executing, and analyzing large-scale AI research experiments.

**Version:** 0.1.1
**Dependencies:** `gen_stage ~> 1.2`, `flow ~> 1.2`, `jason ~> 1.4`, `nimble_csv ~> 1.2`, `statistex ~> 1.0`, `telemetry ~> 1.2`

Positioned as "pytest + MLflow + Weights & Biases" for Elixir AI research.

### Key Features
- Declarative experiment definition via DSL
- Parallel execution using GenStage/Flow
- Fault tolerance and checkpointing
- Statistical significance testing
- Multi-format reporting (Markdown, LaTeX, HTML, Jupyter)
- Cost estimation and budget management

### Architecture

```
CrucibleHarness
├── CrucibleHarness.Experiment          # DSL and definition
│   └── Experiment.Validator            # Config validation
├── CrucibleHarness.Runner              # Execution engine
│   ├── Runner.ProgressTracker
│   └── Runner.RateLimiter
├── CrucibleHarness.Collector           # Results aggregation
│   ├── Collector.MetricsAggregator
│   ├── Collector.StatisticalAnalyzer
│   └── Collector.ComparisonMatrix
├── CrucibleHarness.Reporter            # Output generation
│   ├── Reporter.MarkdownGenerator
│   ├── Reporter.LatexGenerator
│   ├── Reporter.HtmlGenerator
│   └── Reporter.JupyterGenerator
└── CrucibleHarness.Utilities
    ├── Utilities.CheckpointManager
    ├── Utilities.CostEstimator
    └── Utilities.TimeEstimator
```

### Experiment DSL

```elixir
defmodule MyExperiment do
  use CrucibleHarness.Experiment

  name "My Research Experiment"
  description "Comparing baseline vs treatment"
  author "researcher"
  version "1.0.0"
  tags ["ensemble", "reliability"]

  dataset :mmlu_200
  dataset_config %{sample_size: 200}

  conditions [
    %{name: "baseline", fn: &baseline_condition/1},
    %{name: "treatment", fn: &treatment_condition/1}
  ]

  metrics [:accuracy, :latency_p99, :cost_per_query]
  repeat 3

  config %{
    timeout: 30_000,
    rate_limit: 10,
    max_parallel: 10,
    random_seed: 42
  }

  cost_budget %{
    max_total: 100.00,
    max_per_condition: 25.00,
    currency: :usd
  }

  statistical_analysis %{
    significance_level: 0.05,
    multiple_testing_correction: :bonferroni,
    confidence_interval: 0.95
  }

  # Condition implementations
  def baseline_condition(query), do: %{prediction: "A", latency: 100, cost: 0.01}
  def treatment_condition(query), do: %{prediction: "B", latency: 150, cost: 0.02}
end
```

### Core Data Structures

#### Experiment Config (from DSL)
```elixir
%{
  experiment_id: String.t(),
  name: String.t(),
  description: String.t(),
  dataset: atom(),
  conditions: [%{name: String.t(), fn: function()}],
  metrics: [atom()],
  repeat: integer(),
  config: %{timeout: ms, rate_limit: rps, max_parallel: n},
  tags: [String.t()],
  author: String.t(),
  version: String.t(),
  dataset_config: map(),
  cost_budget: map() | nil,
  statistical_analysis: %{
    significance_level: float(),
    multiple_testing_correction: atom(),
    confidence_interval: float()
  }
}
```

#### Task Structure
```elixir
%{
  experiment_id: String.t(),
  condition: %{name: String.t(), fn: function()},
  repeat: integer(),
  query: map(),
  timeout: integer()
}
```

#### Result Structure
```elixir
%{
  experiment_id: String.t(),
  condition: String.t(),
  repeat: integer(),
  query_id: String.t(),
  result: map() | {:error, term()},
  elapsed_time: integer(),
  timestamp: DateTime.t()
}
```

### Key APIs

#### Main Functions
- `run(experiment_module, opts)` - Execute experiment
  - Options: `:output_dir`, `:formats`, `:checkpoint_dir`, `:dry_run`, `:confirm`
- `run_async(experiment_module, opts)` - Async execution
- `status(task_ref)` - Check async status
- `estimate(experiment_module)` - Cost/time estimation
- `resume(experiment_id)` - Resume from checkpoint

#### Pipeline Flow
1. Validate experiment config
2. Estimate cost and time
3. Confirm execution (optional)
4. Execute tasks via Flow
5. Aggregate metrics
6. Perform statistical analysis
7. Generate comparison matrices
8. Produce reports

### Statistical Analysis

#### Pairwise Comparisons
- Welch's t-test (unequal variances)
- P-value calculation
- Significance testing with alpha level
- Multiple testing correction (Bonferroni)

#### Effect Sizes
- Cohen's d calculation
- Pooled standard deviation

#### Confidence Intervals
- t-distribution for small samples
- Configurable confidence level

### Report Formats
1. **Markdown** - GitHub-compatible tables and charts
2. **LaTeX** - Publication-ready tables
3. **HTML** - Interactive web report
4. **Jupyter** - Notebook with code cells

### Integration Points
1. **With crucible_datasets:** Load experimental datasets
2. **With crucible_telemetry:** Emit execution telemetry
3. **With crucible_bench:** Leverage statistical tests
4. **Standalone:** Complete experiment lifecycle

---

## Integration Opportunities

### 1. Unified Experiment Pipeline

Currently, the harness has a mock dataset loader. Integration opportunity:

```elixir
# In CrucibleHarness.Runner
defp load_dataset(config) do
  {:ok, dataset} = CrucibleDatasets.load(
    config.dataset,
    sample_size: config.dataset_config[:sample_size]
  )
  dataset.items
end
```

### 2. Automatic Telemetry Integration

The harness emits `[:research_harness, :task, :complete]`. This should be captured by telemetry:

```elixir
# Add to CrucibleTelemetry.Experiment events list
[:research_harness, :task, :complete],
[:research_harness, :experiment, :start],
[:research_harness, :experiment, :stop]
```

### 3. Causal Trace in Conditions

Allow experiment conditions to emit CausalTrace events:

```elixir
def ensemble_condition(query) do
  chain = CrucibleTrace.new_chain("ensemble_decision_#{query.id}")

  # Log reasoning
  event = CrucibleTrace.create_event(
    :hypothesis_formed,
    "Use majority voting",
    "Best balance of accuracy and latency",
    confidence: 0.9
  )
  chain = CrucibleTrace.add_event(chain, event)

  # Execute and return
  result = Ensemble.predict(query)
  {result, chain}
end
```

### 4. Shared Statistical Functions

Both `CrucibleTelemetry.Analysis` and `CrucibleHarness.Collector.StatisticalAnalyzer` implement:
- Mean, median, variance, std_dev
- Percentiles
- T-tests
- Confidence intervals

**Opportunity:** Extract to `CrucibleStats` shared module.

### 5. Unified Export Pipeline

All libraries export data in similar formats:
- CSV for spreadsheets
- JSON/JSONL for data tools
- Markdown for documentation

**Opportunity:** Shared export utilities with consistent formatting.

### 6. Cross-Library Metadata

Establish shared metadata conventions:
- Experiment IDs
- Timestamps
- Model identifiers
- Cost tracking

### 7. Combined Visualization

CrucibleTrace generates HTML visualizations. Extend to include:
- Telemetry metrics dashboards
- Harness experiment results
- Dataset statistics

---

## Recommendations for Tighter Integration

### High Priority

1. **Create CrucibleCore Package**
   - Shared statistical functions
   - Common data structures
   - Export utilities
   - ID generation

2. **Dataset Integration in Harness**
   - Replace mock loader with CrucibleDatasets
   - Use dataset metadata for cost estimation
   - Enable evaluation integration

3. **Telemetry Event Standardization**
   - Define all event types across packages
   - Consistent metadata schemas
   - Unified handler registration

### Medium Priority

4. **Unified Reporting**
   - Combine harness reports with telemetry analysis
   - Include trace visualizations
   - Dataset statistics integration

5. **Config Schema Standardization**
   - Shared experiment configuration format
   - Cross-library configuration inheritance
   - Validation utilities

6. **Error Handling Patterns**
   - Consistent `{:ok, result} | {:error, reason}`
   - Error categorization
   - Recovery strategies

### Future Enhancements

7. **PostgreSQL Backend**
   - CrucibleTelemetry has planned Postgres support
   - Share connection pool across libraries
   - Enable cross-experiment queries

8. **LiveView Dashboard**
   - Real-time experiment monitoring
   - Interactive trace exploration
   - Dataset browser

9. **Distributed Execution**
   - Multi-node experiment execution
   - Distributed telemetry aggregation
   - Coordinated checkpointing

---

## Dependency Matrix

| Library | jason | telemetry | gen_stage | flow | statistex |
|---------|-------|-----------|-----------|------|-----------|
| crucible_telemetry | v | v | - | - | - |
| crucible_trace | v | - | - | - | - |
| crucible_datasets | v | v | - | - | - |
| crucible_harness | v | v | v | v | v |

All libraries share:
- Elixir ~> 1.14
- MIT License
- ex_doc for documentation

---

## Conclusion

The four infrastructure libraries form a cohesive research platform:

- **crucible_telemetry** provides the observation layer
- **crucible_trace** adds transparency and debugging
- **crucible_datasets** manages experimental data
- **crucible_harness** orchestrates the entire workflow

With targeted integration work, particularly in dataset loading and telemetry capture, these components can function as a seamless research infrastructure. The shared patterns (ETS storage, JSON serialization, telemetry events) make integration straightforward.

Priority should be given to:
1. Connecting harness to actual datasets
2. Capturing harness events in telemetry
3. Extracting shared statistical utilities

This would establish crucible_framework as a complete, production-ready research platform for AI/ML experimentation in Elixir.
