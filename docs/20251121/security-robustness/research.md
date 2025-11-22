# Security and Robustness Libraries Research Report

**Date:** November 21, 2025
**Researcher:** Claude Code Agent
**Scope:** Integration analysis of security/robustness components into crucible_framework

---

## Executive Summary

This report analyzes four North-Shore-AI repositories focused on security, robustness, and explainability for integration into the crucible_framework. These libraries provide complementary capabilities for comprehensive LLM evaluation and protection.

### Repository Overview

| Repository | Version | Tests | Coverage | Primary Function |
|------------|---------|-------|----------|------------------|
| crucible_adversary | 0.2.0 | 203 | 88.5% | Adversarial attack generation & robustness evaluation |
| crucible_xai | 0.2.1 | 277 | 94.1% | Explainable AI (LIME, SHAP, feature attribution) |
| LlmGuard | 0.2.0 | 222 | 97.4% | AI firewall & prompt injection detection |
| crucible_examples | 0.1.0 | 85 | N/A | Interactive Phoenix LiveView demos |

### Key Finding

**crucible_adversary and LlmGuard have significant overlap in prompt injection/jailbreak detection**, but approach from opposite directions:
- **LlmGuard**: Defensive (block malicious inputs before reaching LLM)
- **crucible_adversary**: Offensive (generate attacks to test model robustness)

This creates an ideal red-team/blue-team integration opportunity.

---

## 1. CrucibleAdversary

### Purpose
Adversarial testing framework for evaluating AI/ML model robustness against 21 attack types.

### Architecture

```
lib/crucible_adversary/
├── crucible_adversary.ex          # Main API: attack/2, attack_batch/2, evaluate/3
├── config.ex                       # Configuration management
├── attack_result.ex               # Result struct
├── evaluation_result.ex           # Evaluation result struct
├── perturbations/
│   ├── character.ex               # 5 attacks: swap, delete, insert, homoglyph, keyboard
│   ├── word.ex                    # 4 attacks: deletion, insertion, synonym, shuffle
│   └── semantic.ex                # 4 attacks: paraphrase, back-translate, reorder, formality
├── attacks/
│   ├── injection.ex               # 4 attacks: basic, overflow, delimiter, template
│   └── jailbreak.ex               # 4 attacks: roleplay, context_switch, encode, hypothetical
├── defenses/
│   ├── detection.ex               # Multi-pattern attack detection
│   ├── filtering.ex               # Input filtering (strict/permissive)
│   └── sanitization.ex            # Input cleaning strategies
├── metrics/
│   ├── accuracy.ex                # Accuracy drop calculation
│   ├── asr.ex                     # Attack Success Rate
│   └── consistency.ex             # Semantic similarity & output consistency
└── evaluation/
    └── robustness.ex              # Model robustness evaluation framework
```

### Key APIs

```elixir
# Single attack
{:ok, result} = CrucibleAdversary.attack(input, type: :character_swap, rate: 0.2)

# Batch attacks
{:ok, results} = CrucibleAdversary.attack_batch(inputs, types: [:character_swap, :word_deletion])

# Model evaluation
{:ok, eval} = CrucibleAdversary.evaluate(model, test_set,
  attacks: [:character_swap, :prompt_injection_basic],
  metrics: [:accuracy_drop, :asr]
)

# Defense detection
result = CrucibleAdversary.Defenses.Detection.detect_attack(input)
# => %{is_adversarial: true, confidence: 0.8, detected_patterns: [...], risk_level: :high}
```

### Attack Categories

1. **Character-level (5)**: swap, delete, insert, homoglyph, keyboard_typo
2. **Word-level (4)**: deletion, insertion, synonym_replacement, shuffle
3. **Semantic-level (4)**: paraphrase, back_translate, sentence_reorder, formality_change
4. **Prompt Injection (4)**: basic, overflow, delimiter, template
5. **Jailbreak (4)**: roleplay, context_switch, encode, hypothetical

### Metrics

- **accuracy_drop**: Measures degradation with severity classification (low/moderate/high/critical)
- **asr (Attack Success Rate)**: Per-type success tracking
- **consistency**: Semantic similarity via Jaccard and output consistency stats

### Dependencies

```elixir
# Only development dependency
{:ex_doc, "~> 0.31", only: :dev, runtime: false}
```

**Minimal dependencies** - entirely self-contained with no runtime dependencies.

### Integration Points with crucible_framework

1. **ResearchHarness integration**: Use as attack generator for robustness experiments
2. **Telemetry integration**: Emit attack/evaluation events to crucible_telemetry
3. **Ensemble testing**: Test ensemble voting robustness against adversarial inputs
4. **Bench integration**: Statistical comparison of model robustness metrics

---

## 2. CrucibleXAI

### Purpose
Explainable AI library providing model interpretability through LIME, SHAP, and feature attribution methods.

### Architecture

```
lib/crucible_xai/
├── crucible_xai.ex                # Main API: explain/3, explain_batch/3, explain_shap/4
├── explanation.ex                  # Explanation struct & utilities
├── lime.ex                        # LIME implementation with parallel batch
├── lime/
│   ├── sampling.ex                # Gaussian, Uniform, Categorical, Combined
│   ├── kernels.ex                 # Exponential, Cosine kernels
│   ├── interpretable_models.ex    # Linear Regression, Ridge
│   └── feature_selection.ex       # Highest weights, Forward selection, Lasso
├── shap.ex                        # SHAP API
├── shap/
│   ├── kernel_shap.ex             # Model-agnostic (~1s)
│   ├── linear_shap.ex             # Exact for linear (<2ms)
│   └── sampling_shap.ex           # Monte Carlo (~100ms)
├── gradient_attribution.ex        # Gradient x Input, Integrated Gradients, SmoothGrad
├── occlusion_attribution.ex       # Feature occlusion, Sliding window, Sensitivity
├── global/
│   ├── pdp.ex                     # Partial Dependence Plots (1D & 2D)
│   ├── ice.ex                     # Individual Conditional Expectation
│   ├── ale.ex                     # Accumulated Local Effects
│   └── interaction.ex             # H-Statistic for feature interactions
├── feature_attribution.ex         # Permutation importance API
├── feature_attribution/
│   └── permutation.ex             # Permutation importance implementation
└── visualization.ex               # HTML visualizations (Chart.js)
```

### Key APIs

```elixir
# LIME explanation
explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 5000)

# Batch explanations (with parallel processing)
explanations = CrucibleXai.explain_batch(instances, predict_fn, parallel: true)

# SHAP values
shap_values = CrucibleXai.explain_shap(instance, background, predict_fn)

# Feature importance
importance = CrucibleXai.feature_importance(predict_fn, validation_data)

# Global interpretability
pdp = CrucibleXAI.Global.PDP.partial_dependence(predict_fn, data, feature_idx)
ice = CrucibleXAI.Global.ICE.ice_curves(predict_fn, data, feature_idx)
ale = CrucibleXAI.Global.ALE.accumulated_local_effects(predict_fn, data, feature_idx)
h_stat = CrucibleXAI.Global.Interaction.h_statistic(predict_fn, data, {idx1, idx2})

# Gradient-based (for Nx models)
attrs = CrucibleXAI.GradientAttribution.integrated_gradients(model_fn, instance, baseline)
```

### Methods Implemented

**Local Attribution (10 methods):**
- LIME, KernelSHAP, LinearSHAP, SamplingSHAP
- Gradient x Input, Integrated Gradients, SmoothGrad
- Feature Occlusion, Sliding Window Occlusion, Occlusion Sensitivity

**Global Interpretability (7 methods):**
- Permutation Importance, PDP 1D, PDP 2D
- ICE, Centered ICE, ALE, H-Statistic

### Dependencies

```elixir
{:nx, "~> 0.7"},                    # Numerical computing
{:jason, "~> 1.4"},                 # JSON
{:stream_data, "~> 1.1", only: :test},
{:credo, "~> 1.7", only: [:dev, :test]},
{:dialyxir, "~> 1.4", only: :dev},
{:excoveralls, "~> 0.18", only: :test}
```

**Key dependency: Nx** - requires Nx for tensor operations, gradient computation.

### Integration Points with crucible_framework

1. **CrucibleTrace integration**: Combine explanations with causal traces for comprehensive decision transparency
2. **Ensemble explanation**: Explain why ensemble voting reached a particular decision
3. **Reporter integration**: Generate explanation visualizations in reports
4. **Datasets integration**: Apply XAI to standard benchmark results
5. **Model comparison**: Compare feature attributions across different models

---

## 3. LlmGuard

### Purpose
AI Firewall providing security protection for LLM applications with prompt injection detection, PII redaction, and jailbreak prevention.

### Architecture

```
lib/llm_guard/
├── llm_guard.ex                   # Main API: validate_input/2, validate_output/2, validate_batch/2
├── config.ex                      # Configuration with validation
├── pipeline.ex                    # Security pipeline architecture
├── detector.ex                    # Detector behaviour
├── utils/
│   └── patterns.ex                # Pattern matching utilities
└── detectors/
    ├── prompt_injection.ex        # 34+ patterns, 5 categories
    ├── jailbreak.ex               # Roleplay, hypothetical, encoding attacks
    └── data_leakage/
        ├── pii_scanner.ex         # PII detection
        └── pii_redactor.ex        # PII redaction
```

### Key APIs

```elixir
# Configuration
config = LlmGuard.Config.new(
  prompt_injection_detection: true,
  confidence_threshold: 0.7
)

# Input validation
case LlmGuard.validate_input(user_input, config) do
  {:ok, safe_input} -> process(safe_input)
  {:error, :detected, details} -> handle_threat(details)
end

# Output validation
case LlmGuard.validate_output(llm_response, config) do
  {:ok, safe_output} -> return(safe_output)
  {:error, :detected, details} -> block_output(details)
end

# Batch validation
results = LlmGuard.validate_batch(inputs, config)
```

### Detection Capabilities

**Prompt Injection (34 patterns in 5 categories):**
1. **instruction_override**: "ignore previous instructions", "bypass safety"
2. **system_extraction**: "show system prompt", "output base prompt"
3. **delimiter_injection**: Special tokens, code block roles
4. **mode_switching**: "enable debug mode", "disable filters"
5. **role_manipulation**: DAN jailbreak, "act as unrestricted"

**PII Detection:**
- Email addresses (95% confidence)
- Phone numbers (80-90%)
- SSN (95%)
- Credit cards (98% with Luhn)
- IP addresses (85-90%)
- URLs (90%)

### Multi-Layer Detection Strategy

```
Layer 1: Pattern Matching    (~1ms)   - Fast regex detection
Layer 2: Heuristic Analysis  (~10ms)  - Statistical analysis (coming soon)
Layer 3: ML Classification   (~50ms)  - Advanced threat detection (coming soon)
```

### Dependencies

```elixir
{:telemetry, "~> 1.2"},            # Telemetry events
{:stream_data, "~> 1.0", only: [:test, :dev]},
{:mox, "~> 1.0", only: :test},
{:dialyxir, "~> 1.4", only: :dev},
{:credo, "~> 1.7", only: :dev},
{:benchee, "~> 1.1", only: :dev}
```

**Key dependency: telemetry** - emits telemetry events for monitoring.

### Integration Points with crucible_framework

1. **Pipeline integration**: Use as pre-processor before LLM calls in experiments
2. **Telemetry integration**: Already emits telemetry events
3. **Adversary testing**: Test LlmGuard against crucible_adversary attacks
4. **Rate limiting**: Add to ResearchHarness for experiment throttling
5. **Audit logging**: Integrate with crucible_telemetry for security event storage

---

## 4. CrucibleExamples

### Purpose
Interactive Phoenix LiveView demonstrations showcasing Crucible Framework components with mock LLM scenarios.

### Architecture

```
lib/crucible_examples/
├── application.ex                 # OTP Application
├── mock/
│   ├── models.ex                  # Simulated model responses
│   ├── latency.ex                 # Realistic latency distributions
│   ├── datasets.ex                # Mock benchmark questions
│   └── pricing.ex                 # Cost tracking
├── scenarios/
│   ├── ensemble_demo.ex           # Ensemble voting demo
│   ├── hedging_demo.ex            # Request hedging demo
│   ├── stats_demo.ex              # Statistical comparison demo
│   ├── trace_demo.ex              # Causal trace demo
│   ├── monitoring_demo.ex         # Production monitoring demo
│   └── optimization_demo.ex       # Optimization demo
└── telemetry/                     # Event collection
```

### Crucible Dependencies

```elixir
{:crucible_bench, path: "../crucible_bench"},
{:crucible_ensemble, path: "../crucible_ensemble"},
{:crucible_hedging, path: "../crucible_hedging"},
{:crucible_telemetry, path: "../crucible_telemetry"},
{:crucible_trace, path: "../crucible_trace"},
{:crucible_harness, path: "../crucible_harness"},
{:crucible_datasets, path: "../crucible_datasets"}
```

### Interactive Demos

1. **Ensemble Reliability Dashboard**: Medical diagnosis with 5-model voting
2. **Request Hedging Simulator**: Customer support with tail latency reduction
3. **Statistical Comparison Lab**: A/B testing with t-tests and effect sizes
4. **Causal Trace Explorer**: Multi-step reasoning timeline
5. **Production Monitoring Dashboard**: 30-day model health tracking
6. **Optimization Playground**: Systematic parameter search

### Mock System

Simulates realistic LLM behavior without API keys:
- **GPT-4**: 94% accuracy, $0.005/query, medium latency
- **Claude-3**: 93% accuracy, $0.003/query, fast
- **Gemini-Pro**: 90% accuracy, $0.001/query, medium
- **Llama-3**: 87% accuracy, $0.0002/query, fast
- **Mixtral**: 89% accuracy, $0.0008/query, medium

### Integration Value

- **Reference implementation**: Shows how all Crucible components work together
- **Testing ground**: Validate new integrations with visual feedback
- **Documentation**: Living examples of framework capabilities

---

## Integration Recommendations

### High-Priority Integrations

#### 1. crucible_adversary + LlmGuard: Red-Team/Blue-Team Pipeline

```elixir
defmodule CrucibleFramework.Security.RedBlueTeam do
  @moduledoc "Test security defenses against adversarial attacks"

  def evaluate_defenses(config) do
    # Generate attacks with crucible_adversary
    attacks = CrucibleAdversary.attack_batch(inputs,
      types: [:prompt_injection_basic, :jailbreak_roleplay, :prompt_injection_delimiter]
    )

    # Test defenses with LlmGuard
    results = Enum.map(attacks, fn attack ->
      case LlmGuard.validate_input(attack.attacked, config) do
        {:ok, _} -> {:bypassed, attack}
        {:error, :detected, details} -> {:blocked, attack, details}
      end
    end)

    # Calculate defense effectiveness
    %{
      total_attacks: length(attacks),
      blocked: Enum.count(results, &match?({:blocked, _, _}, &1)),
      bypassed: Enum.count(results, &match?({:bypassed, _}, &1)),
      defense_rate: blocked / length(attacks)
    }
  end
end
```

#### 2. crucible_xai + CrucibleTrace: Comprehensive Decision Transparency

```elixir
defmodule CrucibleFramework.Transparency.FullExplanation do
  @moduledoc "Combine XAI explanations with causal traces"

  def explain_decision(model, instance, trace_ctx) do
    # Get local explanation
    explanation = CrucibleXai.explain(instance, model)

    # Get global importance
    importance = CrucibleXai.feature_importance(model, validation_data)

    # Combine with causal trace
    CrucibleTrace.with_trace(trace_ctx, fn ctx ->
      CrucibleTrace.log_event(ctx, :explanation_generated, %{
        local_weights: explanation.feature_weights,
        global_importance: importance,
        r_squared: explanation.score
      })
    end)

    %{
      local: explanation,
      global: importance,
      trace: trace_ctx
    }
  end
end
```

#### 3. Adversarial Robustness + Ensemble Voting

```elixir
defmodule CrucibleFramework.Robustness.EnsembleTest do
  @moduledoc "Test ensemble robustness against adversarial perturbations"

  def evaluate_ensemble_robustness(ensemble_config, test_set) do
    # Standard evaluation
    {:ok, baseline} = CrucibleEnsemble.vote(clean_inputs, ensemble_config)

    # Adversarial evaluation
    {:ok, attacked} = CrucibleAdversary.evaluate(
      &CrucibleEnsemble.vote(&1, ensemble_config),
      test_set,
      attacks: [:character_swap, :semantic_paraphrase, :word_deletion],
      metrics: [:accuracy_drop, :asr, :consistency]
    )

    %{
      baseline_accuracy: baseline.accuracy,
      adversarial_accuracy: attacked.metrics.accuracy_drop.attacked_accuracy,
      robustness_score: 1 - attacked.metrics.accuracy_drop.relative_drop
    }
  end
end
```

### Medium-Priority Integrations

#### 4. LlmGuard in ResearchHarness Pipeline

Add security validation as a pre-processing step:

```elixir
defmodule CrucibleHarness.Hooks.SecurityValidation do
  def before_llm_call(input, config) do
    case LlmGuard.validate_input(input, config.security) do
      {:ok, sanitized} -> {:ok, sanitized}
      {:error, reason, details} -> {:halt, {:security_blocked, details}}
    end
  end
end
```

#### 5. XAI Visualizations in Reporter

Generate HTML explanation visualizations alongside statistical reports:

```elixir
defmodule CrucibleReporter.Formats.HTMLWithXAI do
  def generate(experiment_results, options) do
    # Generate standard report
    base_html = CrucibleReporter.Formats.HTML.generate(experiment_results, options)

    # Add XAI visualizations
    explanations = options[:explanations] || []
    xai_html = Enum.map(explanations, &CrucibleXAI.Visualization.to_html/1)

    # Combine
    inject_xai_section(base_html, xai_html)
  end
end
```

#### 6. crucible_examples as Integration Test Suite

Use examples scenarios for integration testing:

```elixir
defmodule CrucibleFramework.IntegrationTest do
  use ExUnit.Case

  test "ensemble demo produces valid results" do
    result = CrucibleExamples.Scenarios.EnsembleDemo.run()

    assert result.consensus_reached
    assert result.confidence >= 0.8
    assert length(result.model_responses) == 5
  end
end
```

### Architectural Considerations

#### Shared Dependencies

| Library | Nx | Telemetry | Jason |
|---------|-----|-----------|-------|
| crucible_adversary | No | No | No |
| crucible_xai | **Yes** | No | **Yes** |
| LlmGuard | No | **Yes** | No |
| crucible_examples | No | Yes | Yes |

**Recommendation**: Add optional Nx and telemetry support to crucible_adversary for consistency.

#### Common Patterns

All libraries follow:
- Behaviour-based detectors/evaluators
- Result structs with metadata
- Configuration structs with validation
- Comprehensive test suites (>85% coverage)

#### Suggested Unified Interface

```elixir
defmodule CrucibleFramework.Security do
  @moduledoc "Unified security interface"

  # Defense (LlmGuard)
  defdelegate validate_input(input, config), to: LlmGuard
  defdelegate validate_output(output, config), to: LlmGuard

  # Attack generation (CrucibleAdversary)
  defdelegate generate_attack(input, opts), to: CrucibleAdversary, as: :attack
  defdelegate evaluate_robustness(model, test_set, opts), to: CrucibleAdversary, as: :evaluate

  # Defense detection (CrucibleAdversary.Defenses)
  defdelegate detect_attack(input, opts), to: CrucibleAdversary.Defenses.Detection

  # Combined red-team/blue-team
  def test_defenses(inputs, attack_types, guard_config) do
    # Implementation as shown above
  end
end

defmodule CrucibleFramework.Explainability do
  @moduledoc "Unified XAI interface"

  defdelegate explain(instance, predict_fn, opts \\ []), to: CrucibleXai
  defdelegate explain_shap(instance, background, predict_fn, opts \\ []), to: CrucibleXai
  defdelegate feature_importance(predict_fn, data, opts \\ []), to: CrucibleXai

  # Global methods
  defdelegate pdp(predict_fn, data, feature_idx, opts \\ []), to: CrucibleXAI.Global.PDP, as: :partial_dependence
  defdelegate ice(predict_fn, data, feature_idx, opts \\ []), to: CrucibleXAI.Global.ICE, as: :ice_curves
end
```

---

## Future Roadmap Alignment

### crucible_adversary Roadmap
- Data extraction attacks
- Bias exploitation techniques
- CrucibleBench integration
- Real-time monitoring

### LlmGuard Roadmap
- v0.3.0: Jailbreak detection
- v0.4.0: Content moderation
- v0.5.0: Rate limiting & audit logging
- v0.6.0: Heuristic analysis (Layer 2)
- v1.0.0: ML classification (Layer 3)

### crucible_xai Roadmap
- TreeSHAP for tree-based models
- Advanced visualizations
- CrucibleTrace integration

### Alignment Opportunities

1. **Shared ML layer**: Both LlmGuard and crucible_adversary could benefit from shared ML-based detection
2. **Unified telemetry**: Standardize telemetry events across all security components
3. **Benchmark datasets**: Create shared adversarial benchmarks for testing both attack and defense

---

## Conclusion

The four repositories provide complementary capabilities that together form a comprehensive security and explainability layer for crucible_framework:

- **crucible_adversary**: Attack generation and robustness testing
- **LlmGuard**: Production security defenses
- **crucible_xai**: Model interpretability and explanation
- **crucible_examples**: Reference implementations and visual demos

The highest-value integration is the **red-team/blue-team pipeline** combining crucible_adversary attack generation with LlmGuard defense testing. This enables:

1. Automated security testing of LLM applications
2. Quantified defense effectiveness metrics
3. Continuous improvement of detection patterns
4. Research-grade documentation of vulnerabilities

**Recommended next steps:**

1. Create `CrucibleFramework.Security` unified interface
2. Add telemetry support to crucible_adversary
3. Build red-team/blue-team integration module
4. Add XAI visualization support to CrucibleReporter
5. Create integration test suite using crucible_examples scenarios
