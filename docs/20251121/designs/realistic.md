# Crucible Framework Realistic Integration Design

**Date:** November 21, 2025
**Author:** Realistic Design Agent
**Purpose:** Pragmatic, achievable roadmap for crucible_framework integration

---

## 1. Executive Summary

**The Pragmatic Vision:** Transform crucible_framework from a collection of excellent standalone libraries into a unified research platform through *incremental, low-risk integration steps*.

**Key Principle:** Ship working software every 2 weeks. Each phase must deliver usable value, not just lay groundwork.

**Reality Check:**
- 19 repositories is a lot to coordinate
- No dedicated integration team (assumption)
- Existing users shouldn't break
- Technical debt exists and must be managed
- Perfect is the enemy of good

**Our approach:** Focus on the 20% of work that delivers 80% of value. Start with telemetry unification (everything depends on it), then integrate the most mature libraries first.

---

## 2. Phased Architecture

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Establish shared infrastructure without breaking existing code.

#### Week 1-2: Unified Telemetry Namespace

**Why First:** 14 of 19 repos emit telemetry with inconsistent namespaces. You can't integrate what you can't observe.

**Deliverable:** `CrucibleTelemetry.Unified` module that:
1. Translates old namespaces to new standard
2. Enriches events with experiment context
3. Maintains full backward compatibility

```elixir
defmodule CrucibleTelemetry.Unified do
  @moduledoc """
  Adapter layer for unified telemetry without breaking existing code.
  """

  @old_to_new %{
    [:crucible_ensemble, :predict] => [:crucible, :ensemble, :predict],
    [:crucible_hedging, :request] => [:crucible, :hedging, :request],
    [:req_llm, :request] => [:crucible, :llm, :request]
  }

  def attach_translators do
    # Attach handlers that re-emit events with standard namespace
    Enum.each(@old_to_new, fn {old, new} ->
      :telemetry.attach(
        "#{old}-translator",
        old ++ [:start],
        &translate_event/4,
        %{new_prefix: new}
      )
    end)
  end

  defp translate_event(old_event, measurements, metadata, config) do
    new_event = config.new_prefix ++ [List.last(old_event)]

    enriched_metadata = Map.merge(metadata, %{
      experiment_id: get_experiment_context(),
      original_event: old_event,
      translated_at: System.monotonic_time()
    })

    :telemetry.execute(new_event, measurements, enriched_metadata)
  end
end
```

**Complexity:** Low - Pure additive change
**Risk:** Low - Old code continues working
**Test Strategy:** Integration tests that verify both old and new events fire

#### Week 3-4: Shared Statistics Extraction

**Why:** Statistical functions duplicated in 3 places (telemetry, harness, bench). This is immediate technical debt reduction.

**Deliverable:** `CrucibleStats` module extracted from `crucible_bench`

```elixir
defmodule CrucibleStats do
  @moduledoc """
  Core statistical functions used across Crucible ecosystem.
  Extracted from crucible_bench to reduce duplication.
  """

  # Basic descriptive stats
  defdelegate mean(data), to: CrucibleStats.Descriptive
  defdelegate median(data), to: CrucibleStats.Descriptive
  defdelegate std_dev(data), to: CrucibleStats.Descriptive
  defdelegate variance(data), to: CrucibleStats.Descriptive
  defdelegate percentile(data, p), to: CrucibleStats.Descriptive

  # Percentiles commonly used
  def p50(data), do: percentile(data, 50)
  def p90(data), do: percentile(data, 90)
  def p95(data), do: percentile(data, 95)
  def p99(data), do: percentile(data, 99)
end
```

**Migration:** Update `CrucibleTelemetry.Analysis` and `CrucibleHarness.StatisticalAnalyzer` to delegate to this.

**Complexity:** Low - Extracting existing code
**Risk:** Low - No new functionality, just consolidation

---

### Phase 2: Core Integrations (Weeks 5-8)

**Goal:** Connect the most mature, valuable libraries to the framework.

#### Week 5-6: ExDataCheck + crucible_datasets

**Why:** Data validation is foundational. ExDataCheck is production-ready (273 tests, >90% coverage).

**Deliverable:** Automatic validation on dataset load

```elixir
defmodule CrucibleDatasets.Validator do
  @moduledoc """
  Validates datasets on load using ExDataCheck.
  """

  @default_expectations [
    ExDataCheck.expect_column_to_exist(:id),
    ExDataCheck.expect_column_to_exist(:input),
    ExDataCheck.expect_no_missing_values(:input)
  ]

  def validate_on_load(dataset, opts \\ []) do
    expectations = Keyword.get(opts, :expectations, @default_expectations)

    case ExDataCheck.validate(dataset.items, expectations) do
      %{all_passed: true} = result ->
        {:ok, dataset, result}

      %{all_passed: false} = result ->
        if Keyword.get(opts, :strict, true) do
          {:error, {:validation_failed, result}}
        else
          Logger.warning("Dataset validation warnings: #{inspect(result.failed)}")
          {:ok, dataset, result}
        end
    end
  end
end
```

**Integration point:** Add optional `:validate` flag to `CrucibleDatasets.load/2`

```elixir
# Usage
{:ok, dataset, validation} = CrucibleDatasets.load(:mmlu,
  sample_size: 1000,
  validate: true,
  expectations: [
    ExDataCheck.expect_column_values_to_not_be_null(:input),
    ExDataCheck.expect_column_values_to_be_in_set(:expected, ~w(A B C D))
  ]
)
```

**Complexity:** Low-Medium
**Risk:** Low - Optional feature, doesn't break existing loads

#### Week 7-8: CrucibleBench in Experiment Reports

**Why:** crucible_bench provides publication-ready statistical tests that harness needs.

**Deliverable:** Automatic statistical analysis in harness output

```elixir
defmodule CrucibleHarness.Collector.BenchIntegration do
  @moduledoc """
  Integrates crucible_bench statistical analysis into experiment results.
  """

  def analyze_conditions(results, opts \\ []) do
    conditions = group_by_condition(results)

    # Only run if we have enough data
    if Enum.all?(conditions, fn {_, data} -> length(data) >= 3 end) do
      comparisons = pairwise_comparisons(conditions, opts)
      effect_sizes = calculate_effect_sizes(conditions)

      %{
        statistical_tests: comparisons,
        effect_sizes: effect_sizes,
        power_analysis: maybe_power_analysis(conditions, opts)
      }
    else
      %{note: "Insufficient data for statistical analysis (need n >= 3 per condition)"}
    end
  end

  defp pairwise_comparisons(conditions, opts) do
    metric = Keyword.get(opts, :primary_metric, :accuracy)

    for {name_a, data_a} <- conditions,
        {name_b, data_b} <- conditions,
        name_a < name_b do
      scores_a = Enum.map(data_a, & &1[metric])
      scores_b = Enum.map(data_b, & &1[metric])

      result = CrucibleBench.compare(scores_a, scores_b)

      %{
        comparison: "#{name_a} vs #{name_b}",
        test: result.test,
        p_value: result.p_value,
        significant: result.p_value < Keyword.get(opts, :alpha, 0.05),
        effect_size: result.effect_size
      }
    end
  end
end
```

**Complexity:** Medium
**Risk:** Low - Additive feature to existing reports

---

### Phase 3: Security & Fairness (Weeks 9-12)

**Goal:** Add responsible AI capabilities that differentiate the platform.

#### Week 9-10: LlmGuard Pre-Processing Hook

**Why:** Security validation before LLM calls is essential for production use.

**Deliverable:** Optional security pipeline in harness

```elixir
defmodule CrucibleHarness.Hooks.Security do
  @moduledoc """
  Security pre-processing hook for experiment conditions.
  """

  def validate_input(input, config) do
    guard_config = Map.get(config, :security, default_security_config())

    case LlmGuard.validate_input(input, guard_config) do
      {:ok, sanitized} ->
        {:ok, sanitized}

      {:error, :detected, details} ->
        # Log the security event
        :telemetry.execute(
          [:crucible, :security, :threat_detected],
          %{count: 1},
          %{details: details, input_hash: hash(input)}
        )

        {:halt, {:security_blocked, details}}
    end
  end

  defp default_security_config do
    LlmGuard.Config.new(
      prompt_injection_detection: true,
      confidence_threshold: 0.7
    )
  end
end
```

**Usage in DSL:**

```elixir
defmodule MyExperiment do
  use CrucibleHarness.Experiment

  # ... other config ...

  config %{
    security: %{
      enabled: true,
      prompt_injection_detection: true,
      pii_redaction: true
    }
  }
end
```

**Complexity:** Medium
**Risk:** Low - Optional feature

#### Week 11-12: ExFairness in Experiment Analysis

**Why:** Fairness metrics are increasingly required for responsible AI research.

**Deliverable:** Fairness analysis section in experiment reports

```elixir
defmodule CrucibleHarness.Collector.FairnessAnalysis do
  @moduledoc """
  Fairness analysis for experiment results.
  Requires sensitive_attr to be present in result metadata.
  """

  def analyze(results, opts \\ []) do
    sensitive_attr = Keyword.fetch!(opts, :sensitive_attr)

    # Extract predictions, labels, and sensitive attributes
    {predictions, labels, attrs} = extract_fairness_data(results, sensitive_attr)

    if length(predictions) < 30 do
      %{note: "Insufficient data for fairness analysis (need n >= 30)"}
    else
      report = ExFairness.fairness_report(predictions, labels, attrs,
        threshold: Keyword.get(opts, :fairness_threshold, 0.2)
      )

      %{
        metrics: report.metrics,
        passed: report.passed_count,
        failed: report.failed_count,
        overall: report.overall_assessment,
        disparate_impact: check_disparate_impact(predictions, attrs)
      }
    end
  end

  defp check_disparate_impact(predictions, attrs) do
    result = ExFairness.Detection.DisparateImpact.detect(predictions, attrs)

    %{
      ratio: result.ratio,
      passes_80_percent_rule: result.passes_80_percent_rule,
      legal_note: if(result.passes_80_percent_rule,
        do: "Passes EEOC 80% rule",
        else: "WARNING: Potential disparate impact violation"
      )
    }
  end
end
```

**Usage:**

```elixir
defmodule FairnessExperiment do
  use CrucibleHarness.Experiment

  # ... other config ...

  fairness_analysis %{
    enabled: true,
    sensitive_attr: :gender,  # or :age_group, :ethnicity, etc.
    threshold: 0.15
  }
end
```

**Complexity:** Medium
**Risk:** Low - Optional analysis, doesn't affect experiment execution

---

### Phase 4: Training Integration (Weeks 13-16)

**Goal:** Connect Tinkex training SDK to experiment orchestration.

#### Week 13-14: Tinkex Telemetry Bridge

**Why:** Tinkex training events need to flow into unified telemetry.

```elixir
defmodule CrucibleTelemetry.TinkexBridge do
  @moduledoc """
  Bridges Tinkex telemetry events to Crucible unified namespace.
  """

  @tinkex_events [
    [:tinkex, :http, :request, :start],
    [:tinkex, :http, :request, :stop],
    [:tinkex, :sampling, :complete],
    [:tinkex, :training, :forward_backward, :complete],
    [:tinkex, :training, :optim_step, :complete]
  ]

  def attach do
    :telemetry.attach_many(
      "crucible-tinkex-bridge",
      @tinkex_events,
      &handle_event/4,
      nil
    )
  end

  defp handle_event(event, measurements, metadata, _config) do
    crucible_event = translate_event_name(event)

    enriched = Map.merge(metadata, %{
      experiment_id: Process.get(:crucible_experiment_id),
      source: :tinkex
    })

    :telemetry.execute(crucible_event, measurements, enriched)
  end

  defp translate_event_name([:tinkex | rest]), do: [:crucible, :training | rest]
end
```

**Complexity:** Low
**Risk:** Low - Pure observation, no side effects

#### Week 15-16: Basic Training Context

**Why:** Allow training experiments in harness DSL.

```elixir
defmodule Crucible.Training do
  @moduledoc """
  High-level training orchestration wrapping Tinkex.
  Simplified API for common use cases.
  """

  def create_lora_run(opts) do
    config = validate_config!(opts)

    # Create Tinkex service client
    {:ok, service} = Tinkex.ServiceClient.start_link(
      config: tinkex_config_from_crucible(config)
    )

    # Create training client
    {:ok, trainer} = Tinkex.ServiceClient.create_lora_training_client(
      service,
      base_model: config.model,
      lora_config: build_lora_config(config)
    )

    %{
      service: service,
      trainer: trainer,
      config: config
    }
  end

  def train_batch(run, batch) do
    # Format data for Tinkex
    formatted = format_batch(batch, run.config)

    # Forward-backward pass
    {:ok, output} = Tinkex.TrainingClient.forward_backward(
      run.trainer,
      formatted
    )

    # Optimizer step
    {:ok, _} = Tinkex.TrainingClient.optim_step(
      run.trainer,
      run.config.optimizer
    )

    output
  end

  defp build_lora_config(config) do
    %Tinkex.Types.LoraConfig{
      rank: Map.get(config, :lora_rank, 32),
      alpha: Map.get(config, :lora_alpha, 64),
      dropout: Map.get(config, :lora_dropout, 0.1)
    }
  end
end
```

**Complexity:** Medium-High
**Risk:** Medium - Depends on Tinkex stability

---

## 3. Low-Hanging Fruit

Quick wins that deliver immediate value with minimal effort:

### 3.1 Facade Delegations (1 day)

Create `Crucible` module that just delegates to existing libraries:

```elixir
defmodule Crucible do
  @moduledoc """
  Unified entry point for Crucible ecosystem.
  Start here - all functionality accessible through this module.
  """

  # Ensemble
  defdelegate predict(query, opts \\ []), to: CrucibleEnsemble
  defdelegate predict_async(query, opts \\ []), to: CrucibleEnsemble

  # Hedging
  defdelegate hedged_request(fn, opts \\ []), to: CrucibleHedging, as: :request

  # Statistical analysis
  defdelegate compare(group1, group2, opts \\ []), to: CrucibleBench

  # Data validation
  defdelegate validate_data(data, expectations), to: ExDataCheck, as: :validate

  # Fairness
  defdelegate fairness_report(preds, labels, attrs, opts \\ []), to: ExFairness
end
```

**Why:** Immediate improvement to developer experience. Users can start with `Crucible.` and discover everything.

### 3.2 Dependency Additions to mix.exs (30 minutes)

Add as optional dependencies now, document for future integration:

```elixir
defp deps do
  [
    {:tinkex, "~> 0.1.1"},

    # Optional integrations (will be used in future releases)
    {:ex_data_check, "~> 0.2.0", optional: true},
    {:ex_fairness, "~> 0.2.0", optional: true},
    {:llm_guard, "~> 0.2.0", optional: true},
    {:crucible_ensemble, "~> 0.1.0", optional: true},
    {:crucible_hedging, "~> 0.1.0", optional: true},
    {:crucible_bench, "~> 0.1.0", optional: true},

    # ... existing deps
  ]
end
```

### 3.3 Common Type Definitions (2 hours)

Share common types across ecosystem:

```elixir
defmodule Crucible.Types do
  @moduledoc """
  Common type definitions used across Crucible ecosystem.
  """

  @type experiment_id :: String.t()
  @type metric_name :: atom()
  @type metric_value :: number()

  @type result :: {:ok, term()} | {:error, term()}

  @type validation_result :: %{
    all_passed: boolean(),
    passed: [map()],
    failed: [map()]
  }

  @type fairness_report :: %{
    metrics: map(),
    passed_count: non_neg_integer(),
    failed_count: non_neg_integer(),
    overall_assessment: String.t()
  }
end
```

### 3.4 Configuration Documentation (4 hours)

Create comprehensive config example:

```elixir
# config/crucible_example.exs

config :crucible_framework,
  # Telemetry configuration
  telemetry: [
    namespace: [:crucible],
    storage: :ets,  # or :postgres
    retention_days: 30
  ],

  # Default ensemble settings
  ensemble: [
    default_models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
    default_strategy: :majority,
    timeout: 5_000
  ],

  # Default hedging settings
  hedging: [
    strategy: :percentile,
    percentile: 95,
    enable_cancellation: true
  ],

  # Security defaults
  security: [
    enabled: true,
    prompt_injection_detection: true,
    pii_redaction: false,
    confidence_threshold: 0.7
  ],

  # Fairness defaults
  fairness: [
    enabled: false,  # Opt-in
    threshold: 0.2,
    metrics: [:demographic_parity, :equalized_odds]
  ]
```

---

## 4. Technical Debt Considerations

### 4.1 Work Around (Don't Fix Yet)

**Inconsistent Return Values**
- LlmGuard returns `{:error, :detected, details}` (3-tuple)
- Others return `{:error, reason}` (2-tuple)
- **Decision:** Wrap LlmGuard in adapter that normalizes to 2-tuple
- **Why not fix:** Breaking change for LlmGuard users

**Different Configuration Approaches**
- Some use NimbleOptions, some use custom structs
- **Decision:** Accept any format at boundary, normalize internally
- **Why not fix:** Too much churn for existing users

### 4.2 Fix Incrementally

**Statistical Code Duplication**
- Extract to CrucibleStats (Phase 1, Week 3-4)
- Update one consumer at a time
- Keep original code as deprecated delegations

**Telemetry Namespaces**
- Add translator layer (Phase 1, Week 1-2)
- Libraries continue emitting old events
- Slowly migrate libraries to new namespace over 2-3 releases

### 4.3 Fix Before Major Release

**Export Format Inconsistencies**
- Different CSV/JSON/Markdown formats
- Schedule for unified export module in v1.0
- Document current formats clearly

**Error Type Standardization**
- Different error struct formats
- Schedule `CrucibleError` for v1.0
- Maintain backward compatibility during transition

---

## 5. Resource Estimates

### Complexity Ratings

| Task | Lines of Code | Complexity | Calendar Time | Dependencies |
|------|---------------|------------|---------------|--------------|
| Unified Telemetry | ~200 | Low | 2 weeks | None |
| CrucibleStats extraction | ~150 | Low | 1 week | None |
| ExDataCheck integration | ~100 | Low | 1 week | Telemetry |
| CrucibleBench in reports | ~250 | Medium | 2 weeks | Telemetry |
| LlmGuard hooks | ~150 | Medium | 2 weeks | Telemetry |
| ExFairness analysis | ~200 | Medium | 2 weeks | Bench integration |
| Tinkex bridge | ~100 | Low | 1 week | Telemetry |
| Training context | ~400 | Medium-High | 2 weeks | Tinkex bridge |
| Facade module | ~50 | Low | 1 day | None |

### Team Requirements (Estimated)

**Minimum Viable:** 1 developer, 16 weeks
- Slower pace, sequential phases
- Can skip Phase 4 initially

**Recommended:** 2 developers, 10 weeks
- Parallel work on independent integrations
- One focused on telemetry/infrastructure
- One focused on library integrations

**Ideal:** 3 developers, 6 weeks
- Dedicated testing/documentation person
- Faster iteration cycles
- Better cross-library coordination

---

## 6. Code Examples

### 6.1 Minimal Integration Example

```elixir
defmodule QuickStart do
  @moduledoc """
  Demonstrates basic Crucible integration in ~20 lines.
  """

  def run_simple_experiment do
    # Load validated dataset
    {:ok, dataset, _} = CrucibleDatasets.load(:mmlu,
      sample_size: 100,
      validate: true
    )

    # Run ensemble prediction with hedging
    results = Enum.map(dataset.items, fn item ->
      CrucibleHedging.request(
        fn -> CrucibleEnsemble.predict(item.input) end,
        strategy: :percentile,
        percentile: 95
      )
    end)

    # Analyze results
    scores = extract_accuracy(results)

    %{
      mean_accuracy: CrucibleStats.mean(scores),
      p95_latency: CrucibleStats.p95(extract_latency(results)),
      statistical_summary: CrucibleBench.summary(scores)
    }
  end
end
```

### 6.2 Full Experiment with Safety

```elixir
defmodule SafeExperiment do
  use CrucibleHarness.Experiment

  name "Production-Safe LLM Evaluation"
  description "Ensemble evaluation with security and fairness checks"

  dataset :mmlu, sample_size: 500

  # Pre-validation
  validate_data with: [
    ExDataCheck.expect_column_to_exist(:input),
    ExDataCheck.expect_no_missing_values(:input)
  ]

  # Security
  config %{
    security: %{
      enabled: true,
      prompt_injection_detection: true
    }
  }

  conditions [
    %{name: "single_model", fn: &single_model/1},
    %{name: "ensemble_3", fn: &ensemble_3/1}
  ]

  metrics [:accuracy, :latency_p99, :cost]
  repeat 3

  # Fairness (if sensitive_attr available)
  fairness_analysis %{
    enabled: true,
    sensitive_attr: :demographic_group
  }

  # Statistical analysis
  statistical_analysis %{
    alpha: 0.05,
    correction: :bonferroni
  }

  # Condition implementations
  def single_model(query) do
    {:ok, result, metadata} = CrucibleHedging.request(
      fn -> call_model(:gpt4, query) end,
      strategy: :percentile
    )

    %{
      prediction: result,
      latency: metadata.total_latency,
      cost: metadata.cost
    }
  end

  def ensemble_3(query) do
    {:ok, result} = CrucibleEnsemble.predict(query,
      models: [:gpt4, :claude, :gemini],
      strategy: :majority
    )

    %{
      prediction: result.answer,
      latency: result.metadata.latency,
      cost: result.metadata.cost_usd
    }
  end
end
```

### 6.3 Training Integration Example

```elixir
defmodule TrainingExample do
  @moduledoc """
  LoRA training with Crucible orchestration.
  """

  def train_model(dataset_path, opts \\ []) do
    # Create training run
    run = Crucible.Training.create_lora_run(%{
      model: "meta-llama/Llama-3.1-8B",
      lora_rank: opts[:rank] || 32,
      lora_alpha: opts[:alpha] || 64,
      optimizer: :adam,
      learning_rate: opts[:lr] || 1.0e-4
    })

    # Load and validate training data
    {:ok, data, _} = CrucibleDatasets.load(:custom,
      source: dataset_path,
      validate: true,
      expectations: [
        ExDataCheck.expect_column_to_exist(:prompt),
        ExDataCheck.expect_column_to_exist(:completion)
      ]
    )

    # Training loop
    epochs = opts[:epochs] || 3
    batch_size = opts[:batch_size] || 8

    for epoch <- 1..epochs do
      data.items
      |> Enum.shuffle()
      |> Enum.chunk_every(batch_size)
      |> Enum.each(fn batch ->
        output = Crucible.Training.train_batch(run, batch)

        # Loss is automatically captured via telemetry
        Logger.info("Epoch #{epoch}, Loss: #{output.loss}")
      end)
    end

    # Return trained model handle
    run
  end
end
```

---

## 7. Risk Mitigation

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **Nx version conflicts** | HIGH | HIGH | Pin to Nx 0.7.x across all libraries. Add CI matrix testing for 0.7 and 0.8. |
| **Telemetry migration breaks users** | MEDIUM | HIGH | Translator layer keeps old events firing. 3-release deprecation cycle. |
| **Performance regression** | MEDIUM | MEDIUM | Add benchmarks to CI. Document bypass options for hot paths. |
| **OTP supervision complexity** | LOW | MEDIUM | Provide working supervision tree examples. Default to simple configurations. |

### 7.2 Process Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **Scope creep** | HIGH | MEDIUM | Strict phase boundaries. Each phase must ship before next begins. |
| **Documentation lag** | HIGH | MEDIUM | Each PR must include docs. No merge without doc review. |
| **Breaking existing users** | MEDIUM | HIGH | Semantic versioning. Comprehensive changelog. Migration guides. |

### 7.3 Specific Mitigations

**For Nx Version Conflicts:**
```elixir
# In mix.exs, override Nx version for all deps
defp deps do
  [
    {:nx, "~> 0.7.0", override: true},
    # ... other deps
  ]
end
```

**For Telemetry Migration:**
```elixir
# Deprecation warning for old events
def handle_event([:crucible_ensemble | _] = event, _, _, _) do
  IO.warn("""
  Telemetry event #{inspect(event)} is deprecated.
  Use [:crucible, :ensemble, ...] instead.
  This warning will become an error in v1.0.
  """)
end
```

**For Performance:**
```elixir
# Allow bypassing validation in hot paths
def predict(query, opts \\ []) do
  if Keyword.get(opts, :skip_validation, false) do
    do_predict(query, opts)
  else
    with {:ok, query} <- validate_input(query, opts) do
      do_predict(query, opts)
    end
  end
end
```

---

## 8. Success Metrics

### 8.1 Phase 1 Success (Weeks 1-4)

- [ ] All repos emit events to `[:crucible, ...]` namespace (via translator)
- [ ] Old telemetry events still work (backward compatibility)
- [ ] `CrucibleStats` module extracted and used by telemetry
- [ ] Zero new compilation warnings
- [ ] 10 integration tests pass

### 8.2 Phase 2 Success (Weeks 5-8)

- [ ] `CrucibleDatasets.load/2` supports `:validate` option
- [ ] Harness reports include statistical comparisons from CrucibleBench
- [ ] At least one example experiment uses new features
- [ ] Documentation covers all new features
- [ ] 20 additional integration tests

### 8.3 Phase 3 Success (Weeks 9-12)

- [ ] Harness DSL supports `security` configuration
- [ ] Harness DSL supports `fairness_analysis` configuration
- [ ] Both features are optional and don't break existing experiments
- [ ] Example experiments demonstrate security + fairness
- [ ] 15 additional integration tests

### 8.4 Phase 4 Success (Weeks 13-16)

- [ ] Tinkex events appear in unified telemetry
- [ ] `Crucible.Training` module provides simplified training API
- [ ] At least one training example works end-to-end
- [ ] Training metrics visible in crucible_telemetry exports
- [ ] 10 additional integration tests

### 8.5 Overall Success Criteria

**Quantitative:**
- Total test count: 55+ new integration tests
- Documentation: 100% coverage of new public functions
- Performance: No operation >10% slower than baseline

**Qualitative:**
- New user can run first experiment in <15 minutes
- Existing users experience no breaking changes
- Community feedback incorporated in design decisions

---

## 9. What We're NOT Doing (Yet)

To keep scope realistic, explicitly deferring:

### 9.1 Deferred to v1.1+

- **Distributed execution** - Multi-node experiments (8+ weeks of work)
- **PostgreSQL backend for telemetry** - ETS is sufficient for now
- **Unified CLI** - Keep separate CLIs, document interop
- **Dashboard consolidation** - Keep crucible_ui and cns_ui separate
- **Plugin architecture** - Behaviors are sufficient

### 9.2 Deferred to v2.0

- **Full CrucibleCore package** - Just extract what we need now
- **Standardized error types** - Wrapper adapters are fine
- **Complete export unification** - Document current formats
- **Streaming pipeline** - Batch processing works

### 9.3 Explicitly Out of Scope

- Supporting Python interop
- GPU scheduling/orchestration
- MLflow/W&B integration
- Cloud deployment automation

---

## 10. Conclusion

This design prioritizes **shipping working software** over architectural purity. Each phase delivers usable features:

- **Phase 1:** Unified telemetry + shared statistics (foundation)
- **Phase 2:** Data validation + statistical analysis (immediate research value)
- **Phase 3:** Security + fairness (responsible AI differentiation)
- **Phase 4:** Training integration (complete workflow)

**The key insight:** The existing libraries are already excellent. Our job is to connect them incrementally, not redesign them. Each integration should:

1. Be optional and backward-compatible
2. Deliver immediate value
3. Not block other integrations
4. Ship within 2 weeks

By the end of Phase 4, crucible_framework will be a genuinely unified research platform - not because we rebuilt everything, but because we carefully wired together what already works.

**Recommended First Steps:**
1. Create `CrucibleTelemetry.Unified` translator (Week 1)
2. Add optional dependencies to mix.exs (Day 1)
3. Create `Crucible` facade module (Day 1)
4. Write integration test skeleton (Week 1)

Let's ship it.

---

## Appendix A: Quick Reference

### Mix.exs Changes

```elixir
defp deps do
  [
    {:tinkex, "~> 0.1.1"},
    {:ex_data_check, "~> 0.2.0", optional: true},
    {:ex_fairness, "~> 0.2.0", optional: true},
    {:llm_guard, "~> 0.2.0", optional: true},
    {:crucible_ensemble, "~> 0.1.0", optional: true},
    {:crucible_hedging, "~> 0.1.0", optional: true},
    {:crucible_bench, "~> 0.1.0", optional: true},
    {:crucible_telemetry, "~> 0.1.0", optional: true},
    {:crucible_datasets, "~> 0.1.0", optional: true},
    {:crucible_harness, "~> 0.1.1", optional: true},
    {:crucible_trace, "~> 0.1.0", optional: true},
    {:crucible_adversary, "~> 0.2.0", optional: true},
    {:crucible_xai, "~> 0.2.1", optional: true},
    # Existing deps...
  ]
end
```

### Telemetry Event Naming Convention

```
[:crucible, :component, :action, :phase]

Examples:
[:crucible, :ensemble, :predict, :start]
[:crucible, :ensemble, :predict, :stop]
[:crucible, :hedging, :request, :stop]
[:crucible, :training, :forward_backward, :complete]
[:crucible, :security, :threat_detected]
[:crucible, :fairness, :check, :complete]
```

### Module Naming Convention

```
CrucibleFramework.*     - Core framework modules
Crucible.*              - Unified API facades
Crucible.Training.*     - Training-specific modules
Crucible.Security.*     - Security-specific modules
```

---

*End of Realistic Design Document*
