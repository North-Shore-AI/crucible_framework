# Core Crucible Components Research Report

**Date:** November 21, 2025
**Author:** Research Agent
**Purpose:** Integration planning for tighter coupling between crucible_framework and core components

---

## Executive Summary

This report analyzes four core Crucible ecosystem repositories to identify integration opportunities and architectural insights for enhancing crucible_framework. The components form a cohesive research infrastructure for LLM reliability and performance optimization.

### Key Findings

1. **crucible_framework** (v0.2.1) serves as the orchestration layer with LoRA training, but has limited direct integration with ensemble/hedging/bench
2. **crucible_ensemble** (v0.1.0) is a standalone multi-model voting library with comprehensive execution strategies
3. **crucible_hedging** (v0.1.0) implements Google's tail latency research with adaptive learning
4. **crucible_bench** (v0.1.0) provides publication-ready statistical testing

### Primary Integration Opportunities

- Unify telemetry event namespaces across all components
- Create shared adapter/behaviour patterns for ensemble execution
- Integrate bench statistics into framework reporting pipeline
- Leverage hedging strategies within ensemble execution modes

---

## Repository Analysis

### 1. CrucibleFramework (v0.2.1)

**Path:** `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\crucible_framework\`

#### Purpose
A scientifically-rigorous infrastructure for LLM reliability and performance research. Acts as the top-level orchestration layer that coordinates the other components.

#### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `CrucibleFramework` | `lib/crucible_framework.ex` | Version info, component status |
| `Crucible.Lora` | `lib/crucible/lora.ex` | Adapter-agnostic LoRA entry point |
| `Crucible.Tinkex` | `lib/crucible/tinkex.ex` | Default LoRA adapter |
| `Crucible.Ensemble.*` | `lib/crucible/ensemble/` | ML-specific ensemble (model_ensemble, adapter_pool, ml_voting) |
| `Crucible.Hedging.*` | `lib/crucible/hedging/` | ML-specific hedging (inference_hedger, adaptive_routing) |

#### Dependencies (mix.exs)

```elixir
defp deps do
  [
    {:tinkex, "~> 0.1.1"},           # Core ML backend
    {:supertester, "~> 0.3.1"},      # Testing only
    {:mox, "~> 1.1"},                # Testing only
    {:stream_data, "~> 1.0"},        # Testing only
    {:ex_doc, "~> 0.38"},            # Docs only
    {:dialyxir, "~> 1.4"}            # Analysis only
  ]
end
```

**Notable:** Does NOT currently depend on crucible_ensemble, crucible_hedging, or crucible_bench as hex packages.

#### Public APIs

```elixir
# Core framework functions
CrucibleFramework.version()            # => "0.2.1"
CrucibleFramework.components()         # => [:ensemble, :hedging, :bench, ...]
CrucibleFramework.component_status(:ensemble)  # => :available | :not_loaded

# LoRA training (adapter-agnostic)
Crucible.Lora.create_experiment(opts)
Crucible.Lora.batch_dataset(dataset, batch_size)
Crucible.Lora.format_training_data(batch, opts)
Crucible.Lora.calculate_metrics(results)
Crucible.Lora.validate_quality(results, config)
```

#### Architecture Insights

1. **Adapter Pattern:** `Crucible.Lora` delegates to configurable adapters via `Crucible.Lora.Adapter` behaviour
2. **Internal Ensemble/Hedging:** Has its own ML-specific implementations in `lib/crucible/ensemble/` and `lib/crucible/hedging/` that differ from the standalone libraries
3. **Telemetry Integration:** Uses `[:crucible, :tinkex, ...]` event namespace

#### Integration Gap

The framework has internal implementations but doesn't leverage the standalone crucible_ensemble/hedging libraries directly. This creates code duplication and misses the rich features of those libraries.

---

### 2. CrucibleEnsemble (v0.1.0)

**Path:** `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\crucible_ensemble\`

#### Purpose
Multi-model ensemble prediction with configurable voting strategies for AI reliability research. Designed to achieve 99.9%+ reliability through model consensus.

#### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `CrucibleEnsemble` | `lib/ensemble.ex` | Main API (predict, predict_async, predict_stream) |
| `CrucibleEnsemble.Strategy` | `lib/crucible_ensemble/strategy.ex` | Execution strategies (parallel, sequential, hedged, cascade) |
| `CrucibleEnsemble.Vote` | `lib/crucible_ensemble/vote.ex` | Voting aggregation |
| `CrucibleEnsemble.Executor` | `lib/crucible_ensemble/executor.ex` | Model execution |
| `CrucibleEnsemble.Metrics` | `lib/crucible_ensemble/metrics.ex` | Telemetry and stats |
| `CrucibleEnsemble.Pricing` | `lib/crucible_ensemble/pricing.ex` | Cost tracking |
| `CrucibleEnsemble.Normalize` | `lib/crucible_ensemble/normalize.ex` | Response normalization |

#### Dependencies (mix.exs)

```elixir
defp deps do
  [
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},
    {:ex_doc, "~> 0.31"},   # Dev only
    {:mox, "~> 1.1"}        # Test only
  ]
end
```

**Minimal dependencies** - self-contained and lightweight.

#### Public APIs

```elixir
# Synchronous prediction
CrucibleEnsemble.predict(query, opts)
# => {:ok, %{answer: "4", metadata: %{consensus: 1.0, cost_usd: 0.00015, ...}}}

# Asynchronous prediction
task = CrucibleEnsemble.predict_async(query, opts)
{:ok, result} = Task.await(task)

# Streaming results
stream = CrucibleEnsemble.predict_stream(query, opts)
Enum.each(stream, fn event -> ... end)
```

#### Key Options

```elixir
[
  models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
  strategy: :majority | :weighted | :best_confidence | :unanimous,
  execution: :parallel | :sequential | :hedged | :cascade,
  timeout: 5_000,
  min_responses: ceil(n/2),
  normalization: :lowercase_trim | :numeric | :boolean | :json
]
```

#### Execution Strategies Detail

```elixir
# Parallel - All models simultaneously
CrucibleEnsemble.Strategy.parallel(query, models, opts)

# Sequential - Stop on consensus
CrucibleEnsemble.Strategy.sequential(query, models, opts)

# Hedged - Primary with backup hedges
CrucibleEnsemble.Strategy.hedged(query, primary, backups, opts)

# Cascade - Priority order with early stopping
CrucibleEnsemble.Strategy.cascade(query, models, opts)
```

#### Telemetry Events

```elixir
[:crucible_ensemble, :predict, :start]
[:crucible_ensemble, :predict, :stop]
[:crucible_ensemble, :predict, :exception]
[:crucible_ensemble, :model, :start]
[:crucible_ensemble, :model, :stop]
[:crucible_ensemble, :vote, :complete]
[:crucible_ensemble, :consensus, :reached]
```

#### Integration Opportunities

1. **Direct API Exposure:** Expose `CrucibleEnsemble.predict/2` through framework facade
2. **Strategy Sharing:** Share execution strategies with framework's internal ensemble
3. **Telemetry Unification:** Use consistent namespace `[:crucible, :ensemble, ...]`
4. **Bench Integration:** Feed ensemble results into statistical analysis

---

### 3. CrucibleHedging (v0.1.0)

**Path:** `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\crucible_hedging\`

#### Purpose
Request hedging for tail latency reduction. Implements Google's "Tail at Scale" research to reduce P99 latency by 75-96% with only 5-10% cost overhead.

#### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `CrucibleHedging` | `lib/hedging.ex` | Main API |
| `CrucibleHedging.Strategy` | `lib/crucible_hedging/strategy.ex` | Strategy behaviour |
| `CrucibleHedging.Strategy.Fixed` | `lib/crucible_hedging/strategy/fixed.ex` | Fixed delay |
| `CrucibleHedging.Strategy.Percentile` | `lib/crucible_hedging/strategy/percentile.ex` | Percentile-based |
| `CrucibleHedging.Strategy.Adaptive` | `lib/crucible_hedging/strategy/adaptive.ex` | Thompson Sampling |
| `CrucibleHedging.Strategy.WorkloadAware` | `lib/crucible_hedging/strategy/workload_aware.ex` | Context-sensitive |
| `CrucibleHedging.MultiLevel` | `lib/crucible_hedging/multi_level.ex` | Multi-tier cascades |
| `CrucibleHedging.Metrics` | `lib/crucible_hedging/metrics.ex` | Statistics collection |
| `CrucibleHedging.Config` | `lib/crucible_hedging/config.ex` | NimbleOptions validation |

#### Dependencies (mix.exs)

```elixir
defp deps do
  [
    {:telemetry, "~> 1.2"},
    {:nimble_options, "~> 1.0"},
    {:ex_doc, "~> 0.31"}  # Dev only
  ]
end
```

**Has OTP application:** Starts `CrucibleHedging.Application`.

#### Public APIs

```elixir
# Main entry point
{:ok, result, metadata} = CrucibleHedging.request(
  fn -> make_api_call() end,
  strategy: :percentile,
  percentile: 95,
  timeout_ms: 30_000,
  enable_cancellation: true
)

# Multi-level hedging across providers
{:ok, result, metadata} = CrucibleHedging.MultiLevel.execute(tiers)
```

#### Strategy Behaviour

```elixir
@callback calculate_delay(opts) :: delay_ms()
@callback update(metrics, state) :: state()
```

#### Configuration Options

```elixir
[
  strategy: :fixed | :percentile | :adaptive | :workload_aware,
  delay_ms: 100,           # Fixed strategy
  percentile: 95,          # Percentile strategy
  window_size: 1000,       # Rolling window
  delay_candidates: [50, 100, 200, 500],  # Adaptive
  max_hedges: 1,
  timeout_ms: 30_000,
  enable_cancellation: true,
  telemetry_prefix: [:crucible_hedging]
]
```

#### Telemetry Events

```elixir
[:crucible_hedging, :request, :start]
[:crucible_hedging, :request, :stop]
[:crucible_hedging, :request, :exception]
[:crucible_hedging, :hedge, :fired]
[:crucible_hedging, :hedge, :won]
[:crucible_hedging, :request, :cancelled]
```

#### Metadata Returned

```elixir
%{
  hedged: true,
  hedge_won: false,
  primary_latency: 150,
  backup_latency: nil,
  hedge_delay: 95,
  cost: 1.0,
  total_latency: 150
}
```

#### Integration Opportunities

1. **Ensemble Integration:** Use hedging within ensemble's `:hedged` execution strategy
2. **Framework Hedging:** Replace internal `Crucible.Hedging.InferenceHedger` with this library
3. **Metrics Aggregation:** Feed hedging metrics into bench for statistical analysis
4. **Adaptive Learning:** Share learned delay parameters across experiments

---

### 4. CrucibleBench (v0.1.0)

**Path:** `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\crucible_bench\`

#### Purpose
Statistical testing framework for AI research. Provides rigorous tests, effect size measures, power analysis, and publication-ready reporting.

#### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `CrucibleBench` | `lib/bench.ex` | Main API |
| `CrucibleBench.Analysis` | `lib/crucible_bench/analysis.ex` | Automatic test selection |
| `CrucibleBench.Stats` | `lib/crucible_bench/stats.ex` | Core statistics |
| `CrucibleBench.Stats.TTest` | `lib/crucible_bench/stats/t_test.ex` | t-tests |
| `CrucibleBench.Stats.PairedTTest` | `lib/crucible_bench/stats/paired_t_test.ex` | Paired t-test |
| `CrucibleBench.Stats.ANOVA` | `lib/crucible_bench/stats/anova.ex` | ANOVA |
| `CrucibleBench.Stats.MannWhitney` | `lib/crucible_bench/stats/mann_whitney.ex` | Mann-Whitney U |
| `CrucibleBench.Stats.Wilcoxon` | `lib/crucible_bench/stats/wilcoxon.ex` | Wilcoxon signed-rank |
| `CrucibleBench.Stats.KruskalWallis` | `lib/crucible_bench/stats/kruskal_wallis.ex` | Kruskal-Wallis |
| `CrucibleBench.Stats.EffectSize` | `lib/crucible_bench/stats/effect_size.ex` | Cohen's d, etc. |
| `CrucibleBench.Stats.ConfidenceInterval` | `lib/crucible_bench/stats/confidence_interval.ex` | Bootstrap/analytical CI |
| `CrucibleBench.Stats.Power` | `lib/crucible_bench/stats/power.ex` | Power analysis |
| `CrucibleBench.Experiment` | `lib/crucible_bench/experiment.ex` | High-level DSL |
| `CrucibleBench.Export` | `lib/crucible_bench/export.ex` | Markdown/LaTeX/HTML |
| `CrucibleBench.Result` | `lib/crucible_bench/result.ex` | Result struct |

#### Dependencies (mix.exs)

```elixir
defp deps do
  [
    {:statistex, "~> 1.0"},
    {:nx, "~> 0.7"},
    {:ex_doc, "~> 0.31"}  # Dev only
  ]
end
```

**Uses Nx** for numerical computations.

#### Public APIs

```elixir
# Compare two groups
result = CrucibleBench.compare(control, treatment)

# Paired comparison
result = CrucibleBench.compare_paired(before, after)

# Multiple groups
result = CrucibleBench.compare_multiple([group1, group2, group3])

# Effect size
effect = CrucibleBench.effect_size(group1, group2)

# Confidence interval
ci = CrucibleBench.confidence_interval(data, :mean, method: :bootstrap)

# Power analysis
power = CrucibleBench.power_analysis(:t_test,
  analysis_type: :a_priori,
  effect_size: 0.5,
  power: 0.80
)

# High-level experiments
result = CrucibleBench.experiment(:ab_test,
  control: control,
  treatment: treatment,
  name: "Prompt Test"
)
```

#### Experiment Types

- `:ab_test` - A/B testing
- `:ablation` - Ablation study
- `:hyperparameter_sweep` - Hyperparameter optimization

#### Export Formats

```elixir
CrucibleBench.Export.to_markdown(result)
CrucibleBench.Export.to_latex(result)
CrucibleBench.Export.to_html(result)
CrucibleBench.Export.experiment_to_markdown(result)
```

#### Result Structure

```elixir
%CrucibleBench.Result{
  test: :welch_t_test,
  statistic: 8.45,
  p_value: 0.00012,
  effect_size: %{cohens_d: 4.52, interpretation: "very large"},
  confidence_interval: {0.051, 0.089},
  interpretation: "Treatment shows significantly higher..."
}
```

#### Integration Opportunities

1. **Framework Reporter:** Use bench exports in framework's multi-format reporting
2. **Experiment Harness:** Integrate experiment DSL with framework's ResearchHarness
3. **Automatic Analysis:** Feed ensemble/hedging metrics into bench for analysis
4. **Quality Gates:** Use statistical tests for quality validation in training

---

## Integration Architecture Recommendations

### 1. Dependency Integration

Add the three core libraries as optional dependencies to crucible_framework:

```elixir
# mix.exs
defp deps do
  [
    {:tinkex, "~> 0.1.1"},
    {:crucible_ensemble, "~> 0.1.0", optional: true},
    {:crucible_hedging, "~> 0.1.0", optional: true},
    {:crucible_bench, "~> 0.1.0", optional: true},
    # ...
  ]
end
```

### 2. Unified Telemetry Namespace

Standardize all telemetry events under `[:crucible, ...]`:

```elixir
# Current namespaces:
[:crucible_ensemble, :predict, :stop]
[:crucible_hedging, :request, :stop]
[:crucible, :tinkex, :train, :stop]

# Proposed unified namespace:
[:crucible, :ensemble, :predict, :stop]
[:crucible, :hedging, :request, :stop]
[:crucible, :tinkex, :train, :stop]
[:crucible, :bench, :analysis, :complete]
```

### 3. Facade Module Architecture

Create a unified facade in crucible_framework:

```elixir
defmodule Crucible do
  @moduledoc """
  Unified entry point for all Crucible operations.
  """

  # Ensemble operations
  defdelegate predict(query, opts \\ []), to: CrucibleEnsemble
  defdelegate predict_async(query, opts \\ []), to: CrucibleEnsemble

  # Hedging operations
  defdelegate hedged_request(fn, opts \\ []), to: CrucibleHedging, as: :request

  # Statistical analysis
  defdelegate compare(g1, g2, opts \\ []), to: CrucibleBench
  defdelegate experiment(type, opts), to: CrucibleBench

  # Training operations
  defdelegate train(experiment, dataset, opts), to: Crucible.Lora
end
```

### 4. Hedging-Enhanced Ensemble

Integrate CrucibleHedging into CrucibleEnsemble's `:hedged` execution:

```elixir
# In CrucibleEnsemble.Strategy
def hedged(query, primary, backups, opts) do
  # Use CrucibleHedging for the actual hedging logic
  CrucibleHedging.request(
    fn -> Executor.call_model(primary, query, opts) end,
    strategy: Keyword.get(opts, :hedge_strategy, :percentile),
    percentile: Keyword.get(opts, :hedge_percentile, 95)
  )
end
```

### 5. Bench Integration with Experiment Results

Create automatic statistical analysis of experimental results:

```elixir
defmodule Crucible.ExperimentAnalyzer do
  def analyze_ensemble_comparison(control_results, treatment_results) do
    control_scores = Enum.map(control_results, & &1.metadata.consensus)
    treatment_scores = Enum.map(treatment_results, & &1.metadata.consensus)

    %{
      comparison: CrucibleBench.compare(control_scores, treatment_scores),
      effect: CrucibleBench.effect_size(control_scores, treatment_scores),
      power: CrucibleBench.power_analysis(:t_test,
        analysis_type: :post_hoc,
        n_per_group: length(control_scores)
      )
    }
  end
end
```

### 6. Shared Configuration

Create shared configuration module:

```elixir
# config/config.exs
config :crucible_framework,
  ensemble: [
    default_models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
    default_strategy: :majority,
    timeout: 5_000
  ],
  hedging: [
    default_strategy: :percentile,
    percentile: 95,
    enable_cancellation: true
  ],
  bench: [
    confidence_level: 0.95,
    default_test: :welch_t_test
  ]
```

### 7. Research Pipeline

Create end-to-end research pipeline:

```elixir
defmodule Crucible.ResearchPipeline do
  def run_experiment(config) do
    # 1. Load dataset
    {:ok, dataset} = Crucible.Datasets.load(config.dataset)

    # 2. Run ensemble predictions with hedging
    results = Enum.map(dataset.items, fn item ->
      CrucibleHedging.request(
        fn -> CrucibleEnsemble.predict(item.input, config.ensemble_opts) end,
        config.hedging_opts
      )
    end)

    # 3. Extract metrics
    control = extract_metrics(results.control)
    treatment = extract_metrics(results.treatment)

    # 4. Statistical analysis
    analysis = CrucibleBench.experiment(:ab_test,
      control: control,
      treatment: treatment,
      name: config.name
    )

    # 5. Generate report
    CrucibleBench.Export.experiment_to_markdown(analysis)
  end
end
```

---

## Code Quality Assessment

### Test Coverage

| Repository | Tests | Coverage Notes |
|------------|-------|----------------|
| crucible_framework | Uses supertester, mox | Unknown coverage |
| crucible_ensemble | Uses mox | Comprehensive mocks for API calls |
| crucible_hedging | Standard tests | Metrics and strategy tests |
| crucible_bench | Standard tests | Statistical validation against R/SciPy |

### Documentation Quality

All repositories have:
- Comprehensive `@moduledoc` and `@doc` annotations
- Working examples in documentation
- README files with quick start guides

### Code Patterns

**Consistent patterns across libraries:**
- Telemetry integration
- Keyword options with defaults
- Result tuples `{:ok, result}` / `{:error, reason}`
- Type specifications

**Inconsistencies:**
- Different telemetry namespace prefixes
- Different return value structures (metadata placement)
- Different configuration approaches

---

## Recommended Integration Priorities

### Phase 1: Foundation (Week 1-2)

1. Add optional dependencies to crucible_framework
2. Create unified telemetry namespace
3. Create `Crucible` facade module with delegations

### Phase 2: Deep Integration (Week 3-4)

1. Replace internal ensemble/hedging with library calls
2. Integrate bench into framework reporter
3. Create shared configuration system

### Phase 3: Research Pipeline (Week 5-6)

1. Build end-to-end experiment runner
2. Create automated statistical analysis
3. Generate publication-ready reports

### Phase 4: Documentation (Week 7)

1. Update all documentation for unified API
2. Create integration examples
3. Write migration guide

---

## Conclusion

The four Crucible repositories represent a well-designed, modular research infrastructure. However, the current architecture has unnecessary duplication between the framework's internal implementations and the standalone libraries.

By integrating CrucibleEnsemble, CrucibleHedging, and CrucibleBench directly into crucible_framework as first-class dependencies, the ecosystem would benefit from:

1. **Reduced code duplication** - Single implementation of ensemble/hedging logic
2. **Richer features** - Framework gains adaptive learning, multi-tier hedging, advanced voting
3. **Better statistical rigor** - Direct access to publication-ready analysis
4. **Unified telemetry** - Consistent event naming for observability
5. **Simpler API** - Single entry point for all Crucible operations

The recommended integration approach maintains backward compatibility while enabling more powerful research workflows through the unified `Crucible` facade module.

---

## Appendix: File Inventory

### crucible_framework/lib/ (45+ files)

```
lib/crucible_framework.ex
lib/crucible/lora.ex
lib/crucible/lora/*.ex (5 files)
lib/crucible/tinkex.ex
lib/crucible/tinkex/*.ex (14 files)
lib/crucible/ensemble/*.ex (3 files)
lib/crucible/hedging/*.ex (3 files)
lib/crucible/harness/*.ex (3 files)
lib/crucible/datasets/*.ex (3 files)
lib/crucible/telemetry/*.ex (4 files)
```

### crucible_ensemble/lib/ (8 files)

```
lib/ensemble.ex
lib/crucible_ensemble/application.ex
lib/crucible_ensemble/executor.ex
lib/crucible_ensemble/metrics.ex
lib/crucible_ensemble/normalize.ex
lib/crucible_ensemble/pricing.ex
lib/crucible_ensemble/strategy.ex
lib/crucible_ensemble/vote.ex
```

### crucible_hedging/lib/ (10 files)

```
lib/hedging.ex
lib/crucible_hedging/application.ex
lib/crucible_hedging/config.ex
lib/crucible_hedging/metrics.ex
lib/crucible_hedging/multi_level.ex
lib/crucible_hedging/strategy.ex
lib/crucible_hedging/strategy/*.ex (4 files)
```

### crucible_bench/lib/ (16 files)

```
lib/bench.ex
lib/crucible_bench/analysis.ex
lib/crucible_bench/experiment.ex
lib/crucible_bench/export.ex
lib/crucible_bench/result.ex
lib/crucible_bench/stats.ex
lib/crucible_bench/stats/*.ex (10 files)
```
