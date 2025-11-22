# Useful Design: User-Centric Crucible Integration

**Date:** November 21, 2025
**Author:** Useful Design Agent
**Perspective:** User-centric, value-driven, focused on real problems

---

## 1. Executive Summary: A User-Centric Vision

The Crucible ecosystem has tremendous capability, but capability without usability is wasted potential. This design prioritizes **daily workflow improvement** over architectural elegance.

**Core Principle:** Make common tasks trivial, hard tasks possible, and impossible tasks visible.

**Vision Statement:** A researcher should be able to go from hypothesis to publication-ready results in a single afternoon, with the framework handling the complexity of multi-model orchestration, statistical rigor, fairness checks, and reproducibility.

**Design Philosophy:**
- **Progressive disclosure** - Simple things stay simple
- **Fail fast, fail helpfully** - Errors guide toward solutions
- **Convention over configuration** - Sensible defaults everywhere
- **Documentation as interface** - If it's not documented, it doesn't exist

---

## 2. User Personas: Who Will Use This and What They Need

### Persona 1: Dr. Sarah Chen - Research Scientist (Primary)

**Background:** PhD in ML, 5 years post-doc experience, publishes 3-4 papers/year
**Technical Level:** Strong in Python/PyTorch, learning Elixir for concurrency benefits
**Primary Goals:**
- Run rigorous experiments that pass peer review
- Generate publication-ready tables and figures
- Ensure reproducibility for artifact evaluation
- Meet fairness requirements from ethics board

**Pain Points:**
- "I spend more time on infrastructure than research"
- "I can never remember which statistical test to use"
- "My experiments from 6 months ago won't reproduce"
- "I need to check fairness but don't know where to start"

**Daily Tasks:**
1. Design experiment comparing 3 models on benchmark
2. Run experiment with proper controls
3. Analyze results for statistical significance
4. Generate LaTeX tables for paper
5. Create reproducibility package for reviewers

**Success Metric:** Time from hypothesis to paper-ready results < 4 hours

---

### Persona 2: Marcus Thompson - ML Engineer (Secondary)

**Background:** BS in CS, 3 years at AI startup, ships models to production
**Technical Level:** Strong in Elixir/Phoenix, moderate ML knowledge
**Primary Goals:**
- Validate model quality before deployment
- Monitor production model health
- Ensure security against adversarial inputs
- Maintain low latency at high throughput

**Pain Points:**
- "How do I know if this model is actually better?"
- "Our API gets prompt injection attacks constantly"
- "P99 latency spikes kill our SLAs"
- "I need dashboards but don't have time to build them"

**Daily Tasks:**
1. Run A/B test on new model version
2. Configure request hedging for latency
3. Set up security validation pipeline
4. Monitor ensemble voting accuracy
5. Generate performance report for stakeholders

**Success Metric:** Time from "model ready" to "deployed with monitoring" < 2 hours

---

### Persona 3: Jamie Rodriguez - Graduate Student (Tertiary)

**Background:** 2nd year PhD, first ML project, learning research methodology
**Technical Level:** Moderate Python, beginner Elixir
**Primary Goals:**
- Learn how to run rigorous experiments
- Understand when results are significant
- Build good research habits early
- Complete thesis project successfully

**Pain Points:**
- "I don't know what I don't know"
- "Statistics is confusing - am I doing this right?"
- "My advisor asks about effect sizes and I panic"
- "I need examples to learn from"

**Daily Tasks:**
1. Figure out how to even start an experiment
2. Copy and modify existing examples
3. Understand why something failed
4. Ask "is this result good?"
5. Learn proper methodology through guardrails

**Success Metric:** Time from "confused" to "running first experiment" < 30 minutes

---

## 3. Top Pain Points: What Problems Does Integration Solve

### Pain Point 1: The "Assembly Required" Problem

**Current State:** To run a complete experiment, users must:
1. Load dataset with `crucible_datasets`
2. Configure ensemble with `crucible_ensemble`
3. Set up hedging with `crucible_hedging`
4. Validate inputs with `LlmGuard`
5. Check fairness with `ExFairness`
6. Run statistical tests with `crucible_bench`
7. Generate reports with `crucible_harness`
8. Instrument everything with `crucible_telemetry`

**User Experience:** "I need to read 8 READMEs, understand 8 different APIs, and wire them together myself."

**Integrated Solution:**
```elixir
# One import, one function, complete experiment
Crucible.run_experiment("mmlu_comparison",
  dataset: :mmlu,
  models: [:gpt4, :claude3, :llama3],
  metrics: [:accuracy, :latency, :fairness],
  output: :latex
)
```

**Impact:** 10x reduction in boilerplate for common workflows

---

### Pain Point 2: The "What Test Do I Use?" Problem

**Current State:** Researchers must know:
- When to use t-test vs Mann-Whitney
- When data violates normality assumptions
- How to handle multiple comparisons
- What effect size measure is appropriate

**User Experience:** "I always pick t-test because I know it exists, but I'm probably wrong."

**Integrated Solution:**
```elixir
# Automatic test selection with explanation
result = Crucible.compare(control_results, treatment_results)

# => %{
#   test_used: :mann_whitney_u,
#   why: "Non-normal distributions detected (Shapiro-Wilk p=0.02)",
#   p_value: 0.003,
#   effect_size: %{measure: :rank_biserial, value: 0.72, interpretation: "large"},
#   recommendation: "Statistically significant with large effect. Consider practical significance."
# }
```

**Impact:** Correct statistical tests without requiring statistical expertise

---

### Pain Point 3: The "It Worked Yesterday" Problem

**Current State:** Experiments fail to reproduce due to:
- Undocumented random seeds
- Changed dependencies
- Missing configuration
- Drifted datasets

**User Experience:** "My paper's reviewers can't reproduce my results and I don't know why."

**Integrated Solution:**
```elixir
# Automatic reproducibility manifest
experiment = Crucible.run_experiment(config)

# Generates experiment_manifest.json:
# - Exact dependency versions (mix.lock snapshot)
# - Random seed used
# - Dataset checksum
# - Full configuration
# - Git commit hash
# - System specifications

# Later, any user can:
Crucible.reproduce("experiment_manifest.json")
# => Verifies environment, restores seed, runs identical experiment
```

**Impact:** 100% reproducibility for publication artifact evaluation

---

### Pain Point 4: The "Is This Fair?" Problem

**Current State:** Fairness checking is:
- An afterthought
- Done separately from main evaluation
- Poorly understood
- Inconsistently applied

**User Experience:** "The ethics board asked about demographic parity and I had to scramble."

**Integrated Solution:**
```elixir
# Fairness is a first-class metric
Crucible.run_experiment(config,
  metrics: [:accuracy, :latency, :fairness],
  fairness: [
    sensitive_attrs: [:gender, :age_group],
    thresholds: %{demographic_parity: 0.8, equalized_odds: 0.9}
  ]
)

# Automatically included in report:
# - Demographic parity scores per group
# - Disparate impact ratio (EEOC 80% rule)
# - Clear warnings if thresholds violated
```

**Impact:** Responsible AI compliance built into workflow

---

### Pain Point 5: The "Why Is This Slow?" Problem

**Current State:** Diagnosing latency requires:
- Manual telemetry instrumentation
- Custom dashboards
- Log analysis
- Guessing at bottlenecks

**User Experience:** "P99 is 3 seconds but I don't know which model is slow or if it's network."

**Integrated Solution:**
```elixir
# Automatic latency breakdown
result = Crucible.predict_with_diagnostics(query, models: [:gpt4, :claude3, :llama3])

# => %{
#   prediction: "answer",
#   timing: %{
#     gpt4: %{queue: 5, network: 120, inference: 450, total: 575},
#     claude3: %{queue: 3, network: 95, inference: 320, total: 418},
#     llama3: %{queue: 2, network: 80, inference: 280, total: 362},
#     voting: 8,
#     total: 583
#   },
#   bottleneck: {:model, :gpt4, :inference},
#   recommendation: "Consider replacing gpt4 or enabling request hedging"
# }
```

**Impact:** Instant visibility into performance bottlenecks

---

### Pain Point 6: The "Did I Break Security?" Problem

**Current State:** Security testing is:
- Manual and sporadic
- Not integrated with development
- Done only before launch (if ever)
- Relies on external tools

**User Experience:** "We got prompt-injected in production because nobody tested for it."

**Integrated Solution:**
```elixir
# One-command security evaluation
Crucible.security_audit(my_pipeline,
  attacks: :comprehensive,
  report: :detailed
)

# => %{
#   defense_rate: 0.94,
#   bypassed: [
#     %{type: :prompt_injection_delimiter, payload: "...", recommendation: "Add delimiter stripping"}
#   ],
#   recommendations: ["Update LlmGuard patterns to v2.1", "Add rate limiting"],
#   risk_level: :low
# }
```

**Impact:** Security testing as easy as running tests

---

## 4. Workflow Improvements: How Daily Work Gets Better

### Workflow 1: Experiment Design to Results

**Before (Current):**
```
Day 1: Read docs for datasets, ensemble, harness (3 hours)
Day 1: Write boilerplate to wire components (2 hours)
Day 1: Debug configuration issues (2 hours)
Day 2: Run experiment (30 minutes)
Day 2: Figure out statistical analysis (2 hours)
Day 2: Manually create tables (1 hour)
Total: 10.5 hours across 2 days
```

**After (Integrated):**
```
Hour 1: Define experiment in DSL (30 minutes)
Hour 1: Run experiment (30 minutes)
Hour 2: Review auto-generated report (30 minutes)
Hour 2: Export to LaTeX (5 minutes)
Total: 1.5 hours in one session
```

**Improvement:** 7x faster, single session vs multi-day

---

### Workflow 2: Model Comparison

**Before (Current):**
```elixir
# Load dataset
{:ok, dataset} = CrucibleDatasets.load(:mmlu, sample_size: 500)

# Run each model
results_a = Enum.map(dataset.items, fn item ->
  CrucibleEnsemble.predict(item.input, models: [:gpt4])
end)

results_b = Enum.map(dataset.items, fn item ->
  CrucibleEnsemble.predict(item.input, models: [:claude3])
end)

# Extract scores
scores_a = Enum.map(results_a, & &1.metadata.accuracy)
scores_b = Enum.map(results_b, & &1.metadata.accuracy)

# Run statistical test
result = CrucibleBench.compare(scores_a, scores_b)

# Check assumptions manually...
# Generate report manually...
# Export manually...
```

**After (Integrated):**
```elixir
Crucible.compare_models(:gpt4, :claude3,
  dataset: {:mmlu, sample: 500},
  metrics: [:accuracy, :latency, :cost],
  output: [:console, :latex]
)
```

**Improvement:** 20 lines to 5 lines, no intermediate variables

---

### Workflow 3: Security Validation Pipeline

**Before (Current):**
```elixir
# Generate attacks
attacks = Enum.flat_map(attack_types, fn type ->
  {:ok, results} = CrucibleAdversary.attack_batch(inputs, type: type)
  results
end)

# Test each attack
defense_results = Enum.map(attacks, fn attack ->
  case LlmGuard.validate_input(attack.attacked, config) do
    {:ok, _} -> {:bypassed, attack}
    {:error, :detected, _} -> {:blocked, attack}
  end
end)

# Calculate metrics manually...
```

**After (Integrated):**
```elixir
Crucible.test_security(my_guard_config,
  attacks: [:prompt_injection, :jailbreak],
  samples: 100
)
# => %{defense_rate: 0.96, details: [...], recommendations: [...]}
```

**Improvement:** Automated red-team/blue-team in one call

---

### Workflow 4: Production Monitoring Setup

**Before (Current):**
- Manually attach telemetry handlers
- Build custom dashboards
- Write alert logic
- Parse logs for anomalies

**After (Integrated):**
```elixir
# In application.ex
Crucible.Monitoring.attach(
  components: [:ensemble, :hedging, :security],
  alerts: [
    {:accuracy_drop, threshold: 0.05, notify: :slack},
    {:latency_spike, threshold: 1000, notify: :pagerduty},
    {:security_threat, threshold: 5, window: :minute, notify: :both}
  ],
  dashboard: true  # Starts Phoenix dashboard on :4000/crucible
)
```

**Improvement:** Production monitoring in 10 lines

---

## 5. API Ergonomics: Make Common Cases Dead Simple

### Design Principle: The 80/20 API Surface

**Layer 1 (80% of users):** High-level convenience functions
**Layer 2 (15% of users):** Configurable mid-level APIs
**Layer 3 (5% of users):** Full access to underlying libraries

### High-Level API (Layer 1)

```elixir
defmodule Crucible do
  @moduledoc """
  The main entry point for 80% of Crucible usage.

  Start here. If you need more control, see CrucibleFramework.
  """

  # ============================================
  # EXPERIMENTS (most common task)
  # ============================================

  @doc """
  Run a complete experiment with one function call.

  ## Examples

      # Minimal - just dataset and models
      Crucible.experiment(:mmlu, [:gpt4, :claude3])

      # With metrics
      Crucible.experiment(:mmlu, [:gpt4, :claude3],
        metrics: [:accuracy, :latency, :cost, :fairness]
      )

      # With output formats
      Crucible.experiment(:mmlu, [:gpt4, :claude3],
        output: [:console, :latex, :html]
      )
  """
  def experiment(dataset, models, opts \\ [])

  @doc """
  Compare two configurations head-to-head.

  ## Examples

      Crucible.compare(:gpt4, :claude3, dataset: :mmlu)
      Crucible.compare(:single_model, :ensemble, dataset: :humaneval)
  """
  def compare(config_a, config_b, opts \\ [])

  @doc """
  Run ablation study removing one component at a time.

  ## Examples

      Crucible.ablation([:hedging, :ensemble, :security],
        baseline: full_config,
        dataset: :gsm8k
      )
  """
  def ablation(components, opts \\ [])

  # ============================================
  # PREDICTIONS (second most common)
  # ============================================

  @doc """
  Get a prediction with automatic ensemble and hedging.

  ## Examples

      # Simple
      Crucible.predict("What is 2+2?")

      # With model selection
      Crucible.predict("Complex question", models: [:gpt4, :claude3])

      # With all the bells and whistles
      Crucible.predict("Critical query",
        models: [:gpt4, :claude3, :llama3],
        hedging: true,
        validate: true,
        explain: true
      )
  """
  def predict(query, opts \\ [])

  @doc """
  Batch predictions with progress reporting.
  """
  def predict_batch(queries, opts \\ [])

  # ============================================
  # ANALYSIS (third most common)
  # ============================================

  @doc """
  Statistically compare two result sets with automatic test selection.

  ## Examples

      Crucible.analyze(control_results, treatment_results)
      # => Automatically picks t-test vs Mann-Whitney, calculates effect size
  """
  def analyze(group_a, group_b, opts \\ [])

  @doc """
  Generate publication-ready report from experiment results.
  """
  def report(experiment_id, format \\ :markdown)

  # ============================================
  # SECURITY (fourth most common)
  # ============================================

  @doc """
  Validate input for security threats.

  ## Examples

      case Crucible.validate("user input") do
        {:ok, safe_input} -> process(safe_input)
        {:blocked, reason} -> reject(reason)
      end
  """
  def validate(input, opts \\ [])

  @doc """
  Audit security configuration against attack corpus.
  """
  def security_audit(config, opts \\ [])

  # ============================================
  # FAIRNESS (fifth most common)
  # ============================================

  @doc """
  Check predictions for fairness issues.

  ## Examples

      Crucible.fairness_check(predictions, labels, :gender)
      # => %{demographic_parity: 0.92, equalized_odds: 0.88, ...}
  """
  def fairness_check(predictions, labels, sensitive_attr, opts \\ [])
end
```

### Mid-Level API (Layer 2)

```elixir
defmodule CrucibleFramework do
  @moduledoc """
  More control over experiment configuration.

  Use when Crucible.* functions don't provide enough customization.
  """

  # Ensemble with full configuration
  defdelegate configure_ensemble(opts), to: CrucibleEnsemble, as: :configure
  defdelegate predict_ensemble(query, config), to: CrucibleEnsemble, as: :predict

  # Hedging with strategy selection
  defdelegate configure_hedging(opts), to: CrucibleHedging, as: :configure
  defdelegate hedged_request(func, config), to: CrucibleHedging, as: :request

  # Statistical analysis with test selection
  defdelegate compare_groups(groups, opts), to: CrucibleBench, as: :compare_multiple
  defdelegate effect_size(group_a, group_b, opts), to: CrucibleBench

  # Full harness DSL
  defdelegate define_experiment(block), to: CrucibleHarness

  # Data validation
  defdelegate validate_data(data, expectations), to: ExDataCheck, as: :validate

  # Fairness metrics
  defdelegate fairness_report(predictions, labels, sensitive, opts), to: ExFairness
end
```

### Low-Level API (Layer 3)

Direct access to underlying libraries:
- `CrucibleEnsemble` - Full ensemble API
- `CrucibleHedging` - Full hedging API
- `CrucibleBench` - Full statistical API
- `CrucibleTelemetry` - Full telemetry API
- `LlmGuard` - Full security API
- `ExDataCheck` - Full validation API
- `ExFairness` - Full fairness API
- `CrucibleXai` - Full explainability API

---

## 6. Code Examples: User-Friendly APIs with Great DX

### Example 1: First Experiment (Jamie the Grad Student)

```elixir
# Jamie's first experiment - as simple as possible
# File: my_first_experiment.exs

# Step 1: Import Crucible
import Crucible

# Step 2: Run experiment (that's it!)
result = experiment(:mmlu, [:gpt4, :claude3],
  sample: 100,  # Start small
  output: :console
)

# Console shows:
#
# == Experiment Results ==
# Dataset: MMLU (100 samples)
#
# Model      | Accuracy | Latency (p50) | Cost
# -----------|----------|---------------|------
# gpt4       | 0.89     | 450ms         | $0.15
# claude3    | 0.91     | 380ms         | $0.12
#
# Statistical Analysis:
# - Test used: Mann-Whitney U (data non-normal)
# - p-value: 0.042 (significant at alpha=0.05)
# - Effect size: 0.28 (small)
#
# Interpretation: claude3 shows statistically significant
# improvement with small effect size. Consider larger sample
# for practical significance.
```

### Example 2: Publication-Ready Experiment (Dr. Sarah Chen)

```elixir
# Sarah's ICML submission experiment
# File: icml_2025_experiment.exs

defmodule ICML2025.EnsembleReliability do
  use Crucible.Experiment

  name "Ensemble Voting for LLM Reliability"

  hypothesis """
  H1: Majority voting ensemble of 3 models achieves higher
  accuracy than the best individual model while maintaining
  acceptable latency overhead.
  """

  # Dataset configuration
  dataset :mmlu
  sample_size 1000
  stratified_by :subject

  # Conditions to compare
  conditions do
    condition :baseline do
      models [:gpt4]
      description "Best single model baseline"
    end

    condition :ensemble do
      models [:gpt4, :claude3, :llama3]
      voting :majority
      hedging percentile: 95
      description "3-model ensemble with hedging"
    end
  end

  # Metrics to collect
  metrics do
    primary :accuracy
    secondary [:latency_p50, :latency_p99, :cost_per_query]

    # Fairness analysis
    fairness do
      sensitive_attributes [:subject_domain]  # STEM vs humanities
      threshold demographic_parity: 0.8
    end
  end

  # Statistical configuration
  analysis do
    alpha 0.05
    correction :bonferroni  # Multiple comparisons
    effect_size :cohens_d
    confidence_interval 0.95
  end

  # Output configuration
  output do
    formats [:latex, :html, :jupyter]

    latex do
      template :icml_2025
      table_style :booktabs
      figure_format :pgfplots
    end
  end

  # Reproducibility
  reproducibility do
    random_seed 42
    save_manifest true
    save_raw_data true
  end
end

# Run with:
# mix crucible.run ICML2025.EnsembleReliability
```

Generated LaTeX output:

```latex
\begin{table}[t]
\centering
\caption{Ensemble voting vs. single model baseline on MMLU (n=1000)}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Condition & Accuracy & Latency (p50) & Latency (p99) & Cost \\
\midrule
Baseline (GPT-4) & 0.891 & 450ms & 1.2s & \$0.15 \\
Ensemble (3-model) & \textbf{0.934} & 520ms & 890ms & \$0.42 \\
\midrule
Improvement & +4.3\% & +15\% & \textbf{-26\%} & +180\% \\
\bottomrule
\end{tabular}
\end{table}

% Statistical significance: Mann-Whitney U, p < 0.001
% Effect size: Cohen's d = 0.89 (large)
% Fairness: Demographic parity across subject domains = 0.94 (pass)
```

### Example 3: Production Pipeline (Marcus the ML Engineer)

```elixir
# Marcus's production configuration
# File: lib/my_app/crucible_setup.ex

defmodule MyApp.CrucibleSetup do
  def child_spec(_opts) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, []},
      type: :supervisor
    }
  end

  def start_link do
    Crucible.start_supervised(
      # Ensemble configuration
      ensemble: [
        models: [:gpt4, :claude3, :llama3],
        strategy: :weighted,
        weights: %{gpt4: 0.4, claude3: 0.35, llama3: 0.25},
        timeout: 5_000
      ],

      # Hedging for tail latency
      hedging: [
        strategy: :percentile,
        percentile: 95,
        max_hedges: 1
      ],

      # Security validation
      security: [
        prompt_injection: true,
        jailbreak: true,
        pii_redaction: true,
        confidence_threshold: 0.7
      ],

      # Monitoring
      monitoring: [
        enabled: true,
        dashboard_port: 4001,

        alerts: [
          accuracy_drop: [
            threshold: 0.05,
            window: :hour,
            notify: [:slack, :pagerduty]
          ],
          latency_spike: [
            metric: :p99,
            threshold: 2000,
            notify: [:pagerduty]
          ],
          security_events: [
            threshold: 10,
            window: :minute,
            notify: [:slack, :pagerduty]
          ]
        ]
      ]
    )
  end
end

# Usage in request handler:
defmodule MyApp.QueryHandler do
  def handle_query(query, user_context) do
    case Crucible.predict(query, context: user_context) do
      {:ok, result} ->
        {:ok, result.answer}

      {:blocked, :security_threat, details} ->
        Logger.warning("Security threat detected", details: details)
        {:error, :invalid_query}

      {:error, :timeout} ->
        {:error, :try_again}
    end
  end
end
```

### Example 4: Quick Security Audit

```elixir
# One-liner security audit for existing pipeline
iex> Crucible.security_audit(MyApp.guard_config())

Security Audit Report
====================
Total attacks tested: 500
Blocked: 478 (95.6%)
Bypassed: 22 (4.4%)

By Attack Type:
- prompt_injection_basic: 100% blocked
- prompt_injection_overflow: 98% blocked
- prompt_injection_delimiter: 84% blocked <-- ATTENTION
- jailbreak_roleplay: 96% blocked
- jailbreak_encode: 92% blocked

Recommendations:
1. [HIGH] Update delimiter handling - 16% bypass rate
   Action: Add pattern `~r/\[INST\].*\[\/INST\]/s` to blocked patterns

2. [MEDIUM] Strengthen encoding detection
   Action: Enable base64/rot13 decoding in preprocessing

3. [LOW] Consider jailbreak ML classifier
   Status: Available in LlmGuard v0.3.0
```

### Example 5: Data Quality Validation

```elixir
# Before running expensive experiment, validate data
iex> Crucible.validate_dataset("my_data.jsonl")

Dataset Validation Report
========================
File: my_data.jsonl
Records: 10,000

PASSED:
- Schema validation (all required fields present)
- No null values in critical columns
- Label distribution balanced (47% / 53%)

WARNINGS:
- Column 'response' has 3% outliers by length
- Possible PII detected in 12 records (email patterns)

FAILED:
- Column 'timestamp' has invalid format in 45 records
  Expected: ISO8601, Found: Unix timestamp
  Affected records: [1203, 1204, 1567, ...]

Recommendations:
1. Convert timestamps: `Crucible.transform(data, :timestamp, &DateTime.from_unix/1)`
2. Review PII: `Crucible.show_pii_samples(data, limit: 5)`
3. Handle outliers: Consider `trim: true` option in experiment config
```

---

## 7. Documentation Strategy: How to Make It Learnable

### Documentation Architecture

```
docs/
+-- getting-started/
|   +-- installation.md          # 5-minute quick start
|   +-- first-experiment.md      # 15-minute tutorial
|   +-- key-concepts.md          # Core mental model
|
+-- tutorials/
|   +-- model-comparison.md      # Compare two models
|   +-- ensemble-setup.md        # Configure ensemble
|   +-- statistical-analysis.md  # Understand output
|   +-- fairness-checking.md     # Check for bias
|   +-- security-validation.md   # Protect your pipeline
|   +-- production-deployment.md # Deploy with monitoring
|
+-- how-to/
|   +-- run-ablation-study.md
|   +-- generate-latex-tables.md
|   +-- set-up-alerts.md
|   +-- handle-errors.md
|   +-- optimize-latency.md
|   +-- ensure-reproducibility.md
|
+-- reference/
|   +-- api/                     # Generated from @doc
|   +-- configuration.md         # All config options
|   +-- telemetry-events.md      # All events
|   +-- error-codes.md           # All errors with solutions
|
+-- explanation/
|   +-- architecture.md          # Why it works this way
|   +-- statistical-tests.md     # When to use each test
|   +-- fairness-metrics.md      # What metrics mean
|   +-- security-threats.md      # Attack taxonomy
```

### Documentation Principles

**1. Copy-Pasteable Examples**
Every example should work when pasted into IEx:
```elixir
# This should work:
iex> Crucible.experiment(:mmlu, [:gpt4], sample: 10)
```

**2. Error Message as Documentation**
```elixir
** (Crucible.ConfigError) Invalid model name: :gpt5

Available models:
- :gpt4, :gpt4_turbo - OpenAI GPT-4 variants
- :claude3, :claude3_opus - Anthropic Claude 3
- :llama3, :llama3_70b - Meta LLaMA 3

Did you mean: :gpt4?

See: https://crucible.dev/docs/models for full list
```

**3. Progressive Complexity**
Each doc starts simple, then "But wait, there's more":
```markdown
# Running an Experiment

## Quick Start (most common)
```elixir
Crucible.experiment(:mmlu, [:gpt4, :claude3])
```

## With Options
[more detail]

## Full Control
[DSL example]

## Under the Hood
[link to architecture explanation]
```

**4. Troubleshooting Integrated**
Every page has "Common Issues" section at bottom:
```markdown
## Common Issues

### "Timeout waiting for model response"
**Cause:** Model API latency exceeded default 5s timeout
**Solution:**
```elixir
Crucible.predict(query, timeout: 30_000)  # 30 seconds
```
**See also:** [Handling Slow Models](/how-to/handle-slow-models)
```

### Learning Paths

**Path 1: Researcher in a Hurry (30 minutes)**
1. Getting Started (5 min)
2. First Experiment tutorial (15 min)
3. Generate LaTeX tables how-to (10 min)

**Path 2: Production Engineer (2 hours)**
1. Getting Started (5 min)
2. Ensemble Setup tutorial (20 min)
3. Security Validation tutorial (20 min)
4. Production Deployment tutorial (30 min)
5. Set Up Alerts how-to (15 min)

**Path 3: Understanding the System (4 hours)**
1. Key Concepts explanation (30 min)
2. Architecture explanation (30 min)
3. All tutorials (2.5 hours)
4. Statistical Tests explanation (30 min)

---

## 8. Adoption Path: How Users Migrate to Integrated System

### Migration Strategy: Gradual, Not Big Bang

**Principle:** Users can adopt one feature at a time, never forced to rewrite everything.

### Phase 1: Try It Out (Day 1)

**Existing users:** Keep current code, add Crucible for new features
```elixir
# Old code continues to work
result = CrucibleEnsemble.predict(query, config)

# New convenience for quick experiments
Crucible.experiment(:mmlu, [:gpt4], sample: 50)
```

**New users:** Start with high-level API
```elixir
# Just this. Nothing else needed.
Crucible.experiment(:mmlu, [:gpt4, :claude3])
```

### Phase 2: Adopt Monitoring (Week 1)

```elixir
# Add to application.ex - works with existing code
Crucible.Monitoring.attach(
  components: [:ensemble],  # Only what you use
  dashboard: true
)
```

### Phase 3: Use Unified Telemetry (Week 2)

**Before:** Multiple telemetry handlers
```elixir
:telemetry.attach("ensemble", [:crucible_ensemble, :predict, :stop], &handle/4, nil)
:telemetry.attach("hedging", [:crucible_hedging, :request, :stop], &handle/4, nil)
```

**After:** Single unified handler
```elixir
Crucible.Telemetry.attach("my_handler", &handle_all/4, nil)
# Receives standardized events from all components
```

### Phase 4: Migrate to DSL (Month 1)

**Before:** Manual orchestration
```elixir
# 50 lines of orchestration code
{:ok, dataset} = CrucibleDatasets.load(:mmlu)
# ... wire everything together ...
```

**After:** DSL definition
```elixir
defmodule MyExperiment do
  use Crucible.Experiment
  # ... 20 lines of clear configuration ...
end
```

### Phase 5: Full Integration (Month 2+)

- Replace all direct library calls with Crucible facade
- Use unified configuration
- Leverage all automated features

### Migration Guides

**Guide 1: From standalone crucible_ensemble**
```markdown
# Migration: CrucibleEnsemble to Crucible

## Why Migrate?
- Automatic telemetry integration
- Built-in hedging support
- Unified monitoring

## Step-by-Step
1. Keep existing `CrucibleEnsemble.predict/2` calls
2. Add `Crucible.Monitoring.attach/1` for visibility
3. Gradually replace with `Crucible.predict/2`
4. Remove direct `crucible_ensemble` dependency

## Compatibility
Your existing ensemble config maps directly:
```elixir
# Old
CrucibleEnsemble.predict(query, models: [:gpt4], strategy: :majority)

# New
Crucible.predict(query, models: [:gpt4], strategy: :majority)
```
```

**Guide 2: From manual experiment orchestration**
```markdown
# Migration: Manual to DSL

## Your Current Setup
If you have code that:
- Loads datasets with CrucibleDatasets
- Runs models with CrucibleEnsemble
- Analyzes with CrucibleBench
- Reports with manual formatting

## Migration Path
[detailed steps with before/after code]
```

### Deprecation Policy

**Timeline:**
- **v1.0:** Old APIs work, new APIs available
- **v1.1:** Old APIs emit deprecation warnings
- **v2.0:** Old APIs removed (18 months later)

**Deprecation warnings are helpful:**
```elixir
warning: CrucibleEnsemble.predict/2 is deprecated, use Crucible.predict/2

Migration:
  # Before
  CrucibleEnsemble.predict(query, opts)

  # After
  Crucible.predict(query, opts)

See: https://crucible.dev/migration/ensemble
```

---

## 9. Success Metrics

### User Experience Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Time to first experiment | Unknown | < 30 minutes | User study |
| Time to publication-ready results | ~2 days | < 4 hours | User feedback |
| Lines of code for common experiment | ~100 | < 20 | Code analysis |
| Support questions per user | - | < 3 | Issue tracker |

### Adoption Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Hex downloads (month 1) | 500 | Hex.pm |
| GitHub stars (month 3) | 200 | GitHub |
| Example notebooks created | 20 | Repo count |
| Community plugins | 5 | Registry |

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Documentation coverage | 100% public APIs | ExDoc |
| Example test coverage | 100% | CI |
| Error message helpfulness | > 4/5 rating | User survey |

---

## 10. Conclusion

The Crucible ecosystem has exceptional technical depth. This design ensures that depth is **accessible**.

**Key commitments:**
1. **Simple things are one-liners** - No boilerplate for common tasks
2. **Errors teach** - Every error message includes solution
3. **Defaults are smart** - 80% of users never configure
4. **Migration is gradual** - Adopt at your own pace
5. **Documentation is example-first** - Copy, paste, run

**The test of this design:** A grad student can go from "what is Crucible?" to "running first experiment" in under 30 minutes. A researcher can go from hypothesis to publication-ready tables in an afternoon.

The power is there. Let's make it **usable**.

---

*End of Useful Design Document*
