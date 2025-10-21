# Production Evaluation Systems - A Novel Use Case for Crucible
## Date: 2025-10-20

## Executive Summary

Crucible Framework's novel use case: **The scientific evaluation backbone for production ML systems**. While most evaluation frameworks focus on offline research OR basic production metrics, Crucible bridges both worlds - enabling production systems to make ML decisions with statistical rigor.

## The Gap in Production ML

### What Production Systems Have Today

**Basic Monitoring:**
- Accuracy metrics (overall numbers)
- Latency percentiles (P50, P95, P99)
- Error rates and counts
- Cost tracking

**What's Missing:**
- Statistical significance testing
- Confidence intervals
- Effect size analysis
- Proper A/B test design
- Causal attribution
- Scientific reproducibility

### The Result

Production ML teams make critical decisions based on:
- "Accuracy went from 92% to 94%" ← **Is this significant or noise?**
- "New model seems faster" ← **By how much? With what confidence?**
- "Variant B is better" ← **Based on proper A/B test or eyeballing metrics?**
- "Let's deploy the new prompt" ← **Did we optimize systematically or try 3 things?**

## Crucible's Novel Proposition

### **Scientific Rigor for Production Decisions**

Transform production ML workflows from "gut feel" to "statistical confidence":

```elixir
# Current production ML workflow:
def should_deploy_new_model?() do
  new_accuracy = measure_accuracy(new_model)
  old_accuracy = measure_accuracy(old_model)

  # Decision based on point estimates (WRONG!)
  new_accuracy > old_accuracy
end

# With Crucible:
def should_deploy_new_model?() do
  # Proper experimental design
  {:ok, comparison} = Crucible.ProductionEval.compare_models(
    control: old_model,
    treatment: new_model,
    dataset: validation_set,
    metrics: [:accuracy, :latency_p95, :cost_per_query],
    statistical_tests: true,
    alpha: 0.05,
    min_effect_size: 0.02,
    power: 0.8
  )

  # Decision based on statistical significance + practical significance
  comparison.significant? and
    comparison.effect_size >= 0.02 and
    comparison.guardrails_passed?
end
```

## Novel Use Case: Production Evaluation Pipeline

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Production ML System (Real-time)                     │
│                                                                    │
│  • Serves predictions to users                                    │
│  • Collects telemetry continuously                                │
│  • A/B tests new features                                         │
│  • Deploys models regularly                                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ Telemetry Stream
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│         Crucible Framework (Scientific Evaluation Layer)          │
│                                                                    │
│  Continuous Validation:                                           │
│  → Import production telemetry                                    │
│  → Statistical analysis with confidence intervals                 │
│  → Degradation detection with significance tests                  │
│  → Automated alerts when CI bounds violated                       │
│                                                                    │
│  Pre-Deployment Gates:                                            │
│  → Candidate model vs current model (proper RCT)                  │
│  → Multi-metric validation (accuracy, latency, cost, fairness)    │
│  → Effect size requirements + statistical power                   │
│  → Generate stakeholder reports (Markdown/LaTeX)                  │
│                                                                    │
│  Systematic Optimization:                                         │
│  → Define parameter search spaces                                 │
│  → Grid/Bayesian/evolutionary search                              │
│  → Track optimization history                                     │
│  → Reproduce best configurations                                  │
│                                                                    │
│  A/B Test Analysis:                                               │
│  → Proper experimental design (sample size, power analysis)       │
│  → Multiple comparison correction (Bonferroni, FDR)               │
│  → Sequential testing (early stopping)                            │
│  → Guardrail metrics (prevent regressions)                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ Reports & Decisions
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Stakeholders                                │
│                                                                    │
│  • Engineering: Deploy with confidence                            │
│  • Product: Understand tradeoffs (accuracy vs cost vs latency)    │
│  • Business: ROI validated statistically                          │
│  • Legal/Compliance: Fairness/bias documented                     │
└──────────────────────────────────────────────────────────────────┘
```

## Use Case 1: Pre-Deployment Validation Gates

### Problem

ML teams deploy models based on "looks good" without rigorous validation:
- No statistical tests
- No confidence intervals
- No effect size analysis
- No guardrail checks

### Crucible Solution

**Scientific Deployment Gate:**

```elixir
defmodule ProductionEval.DeploymentGate do
  @moduledoc """
  Scientifically rigorous pre-deployment validation.

  No model reaches production without passing this gate.
  """

  def validate_deployment(candidate_model, current_model, opts \\ []) do
    # 1. Experimental design
    experiment = design_deployment_experiment(candidate_model, current_model, opts)

    # 2. Execute with Crucible harness
    {:ok, results} = ResearchHarness.run(experiment,
      output_dir: "/var/ml_validation/#{experiment.id}"
    )

    # 3. Statistical decision framework
    decision = make_deployment_decision(results)

    # 4. Generate stakeholder report
    {:ok, report} = Reporter.generate(results,
      format: :markdown,
      include: [:statistical_tests, :effect_sizes, :confidence_intervals, :recommendations]
    )

    %{
      decision: decision,  # :deploy, :reject, :need_more_data
      results: results,
      report: report,
      metadata: %{
        timestamp: DateTime.utc_now(),
        validated_by: "Crucible Framework v#{Application.spec(:crucible_framework, :vsn)}",
        experiment_id: experiment.id
      }
    }
  end

  defp design_deployment_experiment(candidate, current, opts) do
    %{
      name: "Deployment Validation: #{candidate.name} vs #{current.name}",
      type: :pre_deployment_gate,

      # Conditions
      conditions: [
        %{name: "current_production", model: current},
        %{name: "candidate", model: candidate}
      ],

      # Metrics with requirements
      metrics: [
        %{name: :accuracy, requirement: %{min: current.baseline_accuracy}},
        %{name: :latency_p95, requirement: %{max: current.baseline_latency * 1.1}},
        %{name: :cost_per_query, requirement: %{max: current.baseline_cost * 1.2}},
        %{name: :error_rate, requirement: %{max: 0.01}}
      ],

      # Statistical requirements
      statistical_config: %{
        alpha: opts[:alpha] || 0.05,
        power: opts[:power] || 0.8,
        min_effect_size: opts[:min_effect_size] || 0.02,
        multiple_testing_correction: :bonferroni
      },

      # Sample size (calculated from power analysis)
      sample_size: calculate_required_sample_size(opts),

      # Repetitions for stability
      repetitions: 3
    }
  end

  defp make_deployment_decision(results) do
    checks = [
      accuracy_improved: check_accuracy_improvement(results),
      no_latency_regression: check_latency_regression(results),
      cost_acceptable: check_cost_acceptable(results),
      guardrails_passed: check_all_guardrails(results),
      statistically_significant: check_statistical_significance(results)
    ]

    all_passed = Enum.all?(checks, fn {_check, result} -> result.passed? end)

    cond do
      all_passed ->
        {:deploy, "All validation checks passed with statistical confidence"}

      checks.statistically_significant.passed? == false ->
        {:need_more_data, "Difference not statistically significant - need larger sample"}

      true ->
        failed_checks = Enum.filter(checks, fn {_k, v} -> not v.passed? end)
        {:reject, "Failed checks: #{inspect(failed_checks)}"}
    end
  end
end
```

## Use Case 2: Continuous Production Monitoring

### Problem

Production models degrade over time (data drift, concept drift, performance decay) but teams only notice when customers complain or metrics crash.

### Crucible Solution

**Statistical Production Monitoring:**

```elixir
defmodule ProductionEval.ContinuousMonitoring do
  @moduledoc """
  Statistical monitoring of production ML with automated alerting.
  """

  def setup_monitoring(model_config) do
    # 1. Establish baseline from historical data
    {:ok, baseline} = establish_baseline(
      model_config.model_id,
      lookback_days: 30
    )

    # 2. Configure monitoring
    monitoring_config = %{
      model_id: model_config.model_id,
      baseline: baseline,

      # Check windows
      check_interval: :hourly,
      comparison_window: :daily,

      # Statistical config
      alpha: 0.01,  # Stricter for production
      consecutive_violations: 3,  # Require 3 violations before alert

      # Metrics to monitor
      metrics: [
        %{name: :accuracy, bounds: :two_sided, threshold: 0.03},
        %{name: :latency_p95, bounds: :upper, threshold: 1.2},
        %{name: :error_rate, bounds: :upper, threshold: 0.01}
      ]
    }

    # 3. Start continuous monitoring
    {:ok, monitor_pid} = ContinuousMonitor.start_link(monitoring_config)

    # 4. Schedule periodic analysis
    schedule_analysis(monitor_pid)

    {:ok, monitor_pid}
  end

  def analyze_recent_performance(model_id, window \\ :last_24_hours) do
    # Import production telemetry
    {:ok, events} = import_production_telemetry(model_id, window)

    # Convert to Crucible experiment
    {:ok, exp} = TelemetryResearch.create_from_events(events,
      name: "Production Monitoring - #{model_id}",
      condition: "production",
      tags: ["monitoring", "automated"]
    )

    # Statistical analysis
    current_metrics = TelemetryResearch.analyze(exp)

    # Load baseline
    baseline_metrics = load_baseline_metrics(model_id)

    # Compare with statistical rigor
    comparison = Bench.compare(
      baseline_metrics.samples,
      current_metrics.samples,
      paired: false,
      alpha: 0.01
    )

    # Detect degradation
    degradation_detected = comparison.significant? and
                          comparison.effect_size < 0 and  # Negative effect
                          abs(comparison.effect_size) > 0.03  # Practical significance

    if degradation_detected do
      alert_team(%{
        severity: :high,
        model_id: model_id,
        message: "Statistically significant performance degradation detected",
        details: %{
          baseline_mean: baseline_metrics.mean,
          current_mean: current_metrics.mean,
          difference: comparison.effect_size,
          p_value: comparison.p_value,
          confidence_interval: comparison.confidence_interval
        },
        recommended_action: :trigger_retraining
      })
    end

    %{
      current_metrics: current_metrics,
      baseline_metrics: baseline_metrics,
      comparison: comparison,
      degradation_detected: degradation_detected,
      timestamp: DateTime.utc_now()
    }
  end

  defp establish_baseline(model_id, opts) do
    lookback_days = Keyword.get(opts, :lookback_days, 30)

    # Get historical telemetry
    start_time = DateTime.add(DateTime.utc_now(), -lookback_days, :day)
    end_time = DateTime.utc_now()

    {:ok, historical_events} = import_production_telemetry(
      model_id,
      {start_time, end_time}
    )

    # Statistical analysis
    {:ok, exp} = TelemetryResearch.create_from_events(historical_events)
    analysis = TelemetryResearch.analyze(exp)

    # Create baseline with confidence intervals
    %{
      model_id: model_id,
      period: {start_time, end_time},
      metrics: analysis.metrics,
      samples: analysis.raw_samples,
      mean: analysis.metrics.mean,
      std_dev: analysis.metrics.std_dev,
      confidence_interval_95: Bench.confidence_interval(analysis.raw_samples, 0.95),
      established_at: DateTime.utc_now()
    }
  end
end
```

## Use Case 3: Systematic LLM Prompt Optimization

### Problem

Production systems using LLMs optimize prompts through manual trial-and-error:
- Try a few variations
- Pick one that "seems better"
- No systematic search
- No statistical validation
- Can't reproduce process

### Crucible Solution

**Scientific Prompt Optimization Pipeline:**

```elixir
defmodule ProductionEval.PromptOptimization do
  @moduledoc """
  Systematically optimize LLM prompts for production use.

  Transforms ad-hoc prompt engineering into scientific optimization.
  """

  def optimize_prompt(task_spec, opts \\ []) do
    # 1. Define variable space for prompt parameters
    variable_space = define_prompt_space(task_spec)

    # 2. Prepare evaluation data
    {:ok, eval_data} = prepare_evaluation_data(task_spec, opts)

    # 3. Define objective function
    objective_fn = create_objective_function(task_spec, eval_data)

    # 4. Run systematic optimization
    {:ok, optimization_result} = Variable.Optimizer.bayesian_optimize(
      variable_space,
      objective_fn,
      n_trials: opts[:n_trials] || 50,
      n_initial_random: 10
    )

    # 5. Validate best configuration
    {:ok, validation_result} = validate_optimal_config(
      optimization_result.best_config,
      eval_data.validation_set
    )

    # 6. Generate deployment report
    {:ok, report} = generate_optimization_report(
      optimization_result,
      validation_result,
      task_spec
    )

    %{
      optimal_config: optimization_result.best_config,
      optimization_history: optimization_result.trials,
      validation: validation_result,
      report: report,
      deployment_ready: validation_result.passed_all_checks?
    }
  end

  defp define_prompt_space(task_spec) do
    Variable.Space.new("prompt_optimization_#{task_spec.name}", [
      # LLM parameters
      Variable.new(:temperature,
        type: :float,
        range: {0.0, 2.0},
        default: 0.7
      ),

      Variable.new(:max_tokens,
        type: :integer,
        range: {100, 4000},
        default: 1000
      ),

      # Prompt engineering parameters
      Variable.new(:system_prompt_variant,
        type: :choice,
        range: task_spec.system_prompt_variants,
        default: :v1
      ),

      Variable.new(:num_examples,
        type: :integer,
        range: {0, 10},
        default: 3
      ),

      Variable.new(:response_format,
        type: :choice,
        range: [:json, :markdown, :plain],
        default: :json
      ),

      # Advanced parameters
      Variable.new(:chain_of_thought,
        type: :boolean,
        default: true
      ),

      Variable.new(:include_confidence,
        type: :boolean,
        default: true
      )
    ])
  end

  defp create_objective_function(task_spec, eval_data) do
    fn config ->
      # Execute LLM with this configuration
      results = Enum.map(eval_data.test_cases, fn test_case ->
        prompt = build_prompt(test_case, config)

        {:ok, response} = call_llm(prompt, config)

        # Evaluate response
        score = task_spec.eval_fn.(response, test_case.expected)
        cost = estimate_cost(response, config)
        latency = response.metadata.latency_ms

        %{score: score, cost: cost, latency: latency}
      end)

      # Composite objective: balance accuracy, cost, latency
      avg_score = Enum.map(results, & &1.score) |> Enum.mean()
      avg_cost = Enum.map(results, & &1.cost) |> Enum.mean()
      avg_latency = Enum.map(results, & &1.latency) |> Enum.mean()

      # Weighted objective (customizable)
      weights = task_spec.objective_weights || %{score: 1.0, cost: -0.1, latency: -0.01}

      (avg_score * weights.score) +
      (avg_cost * weights.cost) +
      (avg_latency * weights.latency)
    end
  end

  defp validate_optimal_config(config, validation_set) do
    # Run optimal config on fresh validation data
    validation_results = Enum.map(validation_set, fn test_case ->
      prompt = build_prompt(test_case, config)
      {:ok, response} = call_llm(prompt, config)

      %{
        expected: test_case.expected,
        actual: response.answer,
        correct: test_case.eval_fn.(response.answer, test_case.expected)
      }
    end)

    # Calculate confidence intervals
    accuracy_samples = Enum.map(validation_results, & if &1.correct, do: 1.0, else: 0.0)

    {:ok, ci} = Bench.confidence_interval(accuracy_samples, 0.95)

    %{
      accuracy: Enum.mean(accuracy_samples),
      confidence_interval_95: ci,
      sample_size: length(validation_results),
      passed_all_checks?: Enum.mean(accuracy_samples) > 0.85 and ci.lower > 0.80,
      detailed_results: validation_results
    }
  end
end
```

## Use Case 4: Ensemble Reliability in Production

### Problem

Production systems use multiple models (ensemble, fallback, A/B test) but don't systematically evaluate reliability improvements.

### Crucible Solution

**Ensemble Strategy Evaluation:**

```elixir
defmodule ProductionEval.EnsembleOptimization do
  @moduledoc """
  Optimize ensemble configurations for production reliability.
  """

  def optimize_ensemble_strategy(production_models, eval_dataset) do
    # Define ensemble configuration space
    ensemble_space = Variable.Space.new("ensemble_optimization", [
      Variable.new(:voting_strategy,
        type: :choice,
        range: [:majority, :weighted, :unanimous, :best_confidence]
      ),

      Variable.new(:execution_strategy,
        type: :choice,
        range: [:parallel, :sequential, :hedged, :cascade]
      ),

      Variable.new(:model_selection,
        type: :choice,
        range: generate_model_combinations(production_models)
      ),

      Variable.new(:confidence_threshold,
        type: :float,
        range: {0.5, 0.95},
        default: 0.8
      ),

      Variable.new(:timeout_ms,
        type: :integer,
        range: {1000, 10000},
        default: 5000
      )
    ])

    # Objective: maximize reliability, minimize cost
    objective_fn = fn config ->
      # Test ensemble with this configuration
      {:ok, ensemble} = Ensemble.create(
        models: select_models(production_models, config.model_selection),
        strategy: config.voting_strategy,
        execution: config.execution_strategy,
        confidence_threshold: config.confidence_threshold,
        timeout: config.timeout_ms
      )

      # Evaluate on dataset
      results = Enum.map(eval_dataset, fn item ->
        case Ensemble.predict(ensemble, item.input) do
          {:ok, result} ->
            %{
              correct: result.answer == item.expected,
              consensus: result.metadata.consensus,
              cost: result.metadata.cost_usd,
              latency: result.metadata.latency_ms
            }
          {:error, _} ->
            %{correct: false, consensus: 0, cost: 0, latency: 0}
        end
      end)

      # Composite score
      accuracy = Enum.count(results, & &1.correct) / length(results)
      avg_consensus = Enum.map(results, & &1.consensus) |> Enum.mean()
      avg_cost = Enum.map(results, & &1.cost) |> Enum.mean()
      avg_latency = Enum.map(results, & &1.latency) |> Enum.mean()

      # Weighted objective
      (accuracy * 100) +           # Accuracy is primary
      (avg_consensus * 10) -       # Higher consensus is good
      (avg_cost * 1000) -          # Minimize cost
      (avg_latency / 100)          # Minimize latency
    end

    # Optimize with Crucible
    {:ok, optimal} = Variable.Optimizer.bayesian_optimize(
      ensemble_space,
      objective_fn,
      n_trials: 100
    )

    # Validate optimal configuration
    {:ok, validation} = validate_ensemble_config(optimal.best_config, production_models)

    %{
      optimal_config: optimal.best_config,
      expected_accuracy: validation.accuracy,
      expected_cost: validation.cost_per_query,
      expected_latency: validation.latency_p95,
      confidence_interval: validation.confidence_interval,
      deployment_ready: validation.meets_production_requirements?
    }
  end

  def monitor_ensemble_health(ensemble_id) do
    # Collect recent production data
    recent_data = get_recent_ensemble_telemetry(ensemble_id, hours: 24)

    # Analyze per-model contribution
    model_analysis = analyze_model_contributions(recent_data)

    # Detect issues
    issues = [
      # Model consistently wrong
      check_model_accuracy_degradation(model_analysis),

      # Model too slow
      check_model_latency_regression(model_analysis),

      # Model too expensive
      check_model_cost_increase(model_analysis),

      # Consensus dropping
      check_consensus_degradation(recent_data),

      # Voting strategy suboptimal
      check_voting_strategy_effectiveness(recent_data)
    ] |> List.flatten()

    if length(issues) > 0 do
      generate_health_report(ensemble_id, issues, model_analysis)
    else
      {:ok, :healthy}
    end
  end
end
```

## Use Case 5: Multi-Variant Experiment Analysis

### Problem

Production teams run A/B/C/D/... tests with multiple variants but:
- Use wrong statistical tests (treat as multiple t-tests)
- No multiple comparison correction
- No power analysis beforehand
- Can't handle early stopping correctly

### Crucible Solution

**Rigorous Multi-Variant Testing:**

```elixir
defmodule ProductionEval.MultiVariantTest do
  @moduledoc """
  Statistically rigorous multi-variant testing for production ML.
  """

  def design_multi_variant_test(variants, opts) do
    # Power analysis for multi-variant test
    sample_size_per_variant = Bench.power_analysis(
      test_type: :anova,
      effect_size: opts[:expected_effect_size] || 0.02,
      power: opts[:power] || 0.8,
      alpha: opts[:alpha] || 0.05,
      num_groups: length(variants)
    )

    %{
      name: opts[:name] || "Multi-Variant Test",
      variants: variants,

      # Balanced allocation
      traffic_allocation: equal_allocation(variants),

      # Sample size per variant
      sample_size_per_variant: sample_size_per_variant,
      total_sample_size: sample_size_per_variant * length(variants),

      # Statistical configuration
      statistical_config: %{
        test_type: :anova,  # For multiple groups
        alpha: opts[:alpha] || 0.05,
        multiple_comparison_correction: :bonferroni,
        post_hoc_tests: true
      },

      # Metrics
      primary_metric: opts[:primary_metric],
      secondary_metrics: opts[:secondary_metrics] || [],
      guardrail_metrics: opts[:guardrail_metrics] || [],

      # Early stopping
      early_stopping: opts[:early_stopping] || false,
      interim_analyses: opts[:interim_analyses] || []
    }
  end

  def analyze_multi_variant_results(test_design, production_data) do
    # Group data by variant
    variant_groups = group_by_variant(production_data, test_design.variants)

    # 1. Overall ANOVA test
    metric_samples = Enum.map(variant_groups, fn {variant, data} ->
      extract_metric(data, test_design.primary_metric)
    end)

    {:ok, anova_result} = Bench.anova(metric_samples,
      labels: Enum.map(variant_groups, fn {variant, _} -> variant.name end)
    )

    # 2. Post-hoc pairwise comparisons (if ANOVA significant)
    pairwise_comparisons = if anova_result.significant? do
      perform_pairwise_comparisons(
        variant_groups,
        test_design.primary_metric,
        correction: :bonferroni
      )
    else
      []
    end

    # 3. Effect size analysis
    effect_sizes = calculate_effect_sizes(variant_groups, test_design.primary_metric)

    # 4. Practical significance
    practical_significance = assess_practical_significance(
      variant_groups,
      test_design.primary_metric,
      min_improvement: test_design.min_effect_size || 0.02
    )

    # 5. Guardrail checks
    guardrail_results = check_all_guardrails(variant_groups, test_design.guardrail_metrics)

    # 6. Winner selection
    winner = select_winner(
      anova_result,
      pairwise_comparisons,
      effect_sizes,
      practical_significance,
      guardrail_results
    )

    %{
      anova: anova_result,
      pairwise_comparisons: pairwise_comparisons,
      effect_sizes: effect_sizes,
      practical_significance: practical_significance,
      guardrail_results: guardrail_results,
      winner: winner,
      confidence: calculate_decision_confidence(anova_result, effect_sizes),
      recommendation: generate_deployment_recommendation(winner, guardrail_results)
    }
  end

  defp perform_pairwise_comparisons(variant_groups, metric, opts) do
    correction = Keyword.get(opts, :correction, :bonferroni)

    # All pairwise comparisons
    pairs = for {v1, d1} <- variant_groups,
                {v2, d2} <- variant_groups,
                v1.name < v2.name do
      {v1.name, v2.name, d1, d2}
    end

    # Run comparisons
    comparisons = Enum.map(pairs, fn {v1_name, v2_name, d1, d2} ->
      samples1 = extract_metric(d1, metric)
      samples2 = extract_metric(d2, metric)

      {:ok, comparison} = Bench.compare(samples1, samples2)

      %{
        pair: "#{v1_name} vs #{v2_name}",
        comparison: comparison
      }
    end)

    # Apply multiple comparison correction
    p_values = Enum.map(comparisons, & &1.comparison.p_value)
    corrected_alpha = apply_correction(correction, p_values, 0.05)

    # Mark significant after correction
    Enum.map(comparisons, fn comp ->
      %{comp |
        significant_after_correction: comp.comparison.p_value < corrected_alpha
      }
    end)
  end
end
```

## Use Case 6: Automated Retraining Triggers

### Problem

Production systems don't know WHEN to retrain models:
- Time-based (every week? month?) is arbitrary
- Manual detection of degradation is slow
- No statistical confidence in retraining decision

### Crucible Solution

**Statistical Retraining Decision System:**

```elixir
defmodule ProductionEval.RetrainingTrigger do
  @moduledoc """
  Statistically-driven automated retraining triggers.
  """

  def setup_triggers(model_config) do
    %{
      model_id: model_config.model_id,

      triggers: [
        # Trigger 1: Statistical degradation
        %{
          type: :performance_degradation,
          enabled: true,
          config: %{
            metric: :accuracy,
            alpha: 0.01,  # Strong evidence required
            min_effect_size: 0.03,  # Practical significance
            consecutive_windows: 3,  # Must persist
            window_size_hours: 24
          }
        },

        # Trigger 2: Data drift
        %{
          type: :data_drift,
          enabled: true,
          config: %{
            test: :kolmogorov_smirnov,
            alpha: 0.01,
            feature_subset: :top_10_important
          }
        },

        # Trigger 3: Concept drift
        %{
          type: :concept_drift,
          enabled: true,
          config: %{
            test: :page_hinkley,
            threshold: 50,
            min_instances: 1000
          }
        },

        # Trigger 4: Cost efficiency
        %{
          type: :cost_efficiency,
          enabled: false,  # Optional
          config: %{
            cost_increase_threshold: 1.5,
            accuracy_decrease_threshold: 0.02
          }
        }
      ],

      # Retraining configuration
      retraining_config: %{
        min_new_data_samples: 1000,
        validation_split: 0.2,
        require_improvement: true,
        deployment_gate: :statistical_validation
      }
    }
  end

  def evaluate_triggers(trigger_config, current_telemetry) do
    # Evaluate each trigger
    trigger_evaluations = Enum.map(trigger_config.triggers, fn trigger ->
      evaluate_single_trigger(trigger, current_telemetry, trigger_config)
    end)

    # Aggregate decision
    any_triggered = Enum.any?(trigger_evaluations, & &1.triggered?)

    # Generate recommendation
    recommendation = if any_triggered do
      triggered_reasons = Enum.filter(trigger_evaluations, & &1.triggered?)
                         |> Enum.map(& &1.reason)

      %{
        action: :initiate_retraining,
        reasons: triggered_reasons,
        confidence: calculate_trigger_confidence(trigger_evaluations),
        estimated_improvement: estimate_retraining_benefit(trigger_evaluations)
      }
    else
      %{
        action: :continue_monitoring,
        next_check: schedule_next_check(trigger_config)
      }
    end

    %{
      trigger_evaluations: trigger_evaluations,
      recommendation: recommendation,
      timestamp: DateTime.utc_now()
    }
  end

  defp evaluate_single_trigger(%{type: :performance_degradation} = trigger, telemetry, config) do
    # Get baseline performance
    baseline = load_baseline_metrics(config.model_id)

    # Get recent performance (sliding windows)
    recent_windows = get_recent_windows(
      telemetry,
      trigger.config.window_size_hours,
      trigger.config.consecutive_windows
    )

    # Statistical test for each window
    window_results = Enum.map(recent_windows, fn window_data ->
      Bench.compare(
        baseline.samples,
        extract_metric_samples(window_data, trigger.config.metric),
        alpha: trigger.config.alpha
      )
    end)

    # Check if degradation is consistent
    consistent_degradation = Enum.all?(window_results, fn result ->
      result.significant? and
      result.effect_size < 0 and  # Performance worse
      abs(result.effect_size) >= trigger.config.min_effect_size
    end)

    %{
      trigger_type: :performance_degradation,
      triggered?: consistent_degradation,
      reason: if consistent_degradation do
        "Performance degraded significantly for #{trigger.config.consecutive_windows} consecutive windows"
      else
        nil
      end,
      details: %{
        baseline_mean: baseline.mean,
        recent_means: Enum.map(recent_windows, &calculate_mean/1),
        statistical_tests: window_results
      }
    }
  end

  defp evaluate_single_trigger(%{type: :data_drift} = trigger, telemetry, config) do
    # Get baseline data distribution
    baseline_dist = load_baseline_distribution(config.model_id)

    # Get recent data distribution
    recent_dist = extract_recent_distribution(telemetry, trigger.config.feature_subset)

    # Kolmogorov-Smirnov test for distribution shift
    {:ok, ks_result} = Bench.kolmogorov_smirnov_test(
      baseline_dist.samples,
      recent_dist.samples
    )

    drift_detected = ks_result.p_value < trigger.config.alpha

    %{
      trigger_type: :data_drift,
      triggered?: drift_detected,
      reason: if drift_detected do
        "Significant data distribution shift detected (p=#{ks_result.p_value})"
      else
        nil
      end,
      details: %{
        ks_statistic: ks_result.statistic,
        p_value: ks_result.p_value,
        baseline_summary: baseline_dist.summary,
        recent_summary: recent_dist.summary
      }
    }
  end
end
```

## Use Case 7: DSPy Program Lifecycle Management

### Problem

Teams using DSPy in production face:
- No systematic way to evaluate DSPy program performance
- Manual prompt optimization
- No version control for prompt configurations
- Can't A/B test different DSPy strategies

### Crucible Solution

**Complete DSPy Lifecycle Management:**

```elixir
defmodule ProductionEval.DSPyLifecycle do
  @moduledoc """
  Complete lifecycle management for production DSPy programs.

  Covers: Development → Optimization → Validation → Deployment → Monitoring
  """

  # Stage 1: Development - Evaluate initial DSPy program
  def evaluate_dspy_program(program_fn, eval_dataset, opts \\ []) do
    program = Crucible.DSPy.Program.new(
      name: opts[:name] || "DSPy Program",
      execute_fn: program_fn
    )

    {:ok, results} = Crucible.DSPy.Evaluation.evaluate(
      program,
      dataset: eval_dataset,
      config: opts[:config] || %{},
      metrics: [:accuracy, :cost_per_query, :latency_p95]
    )

    %{
      baseline_performance: results,
      ready_for_optimization: results.accuracy > 0.7,
      next_step: if results.accuracy > 0.7, do: :optimize, else: :revise_program
    }
  end

  # Stage 2: Optimization - Systematic parameter search
  def optimize_dspy_program(program_fn, training_data, validation_data, opts \\ []) do
    # Define search space
    variable_space = Variable.Space.new("dspy_optimization", [
      Variable.new(:temperature, type: :float, range: {0.0, 2.0}),
      Variable.new(:max_tokens, type: :integer, range: {100, 2000}),
      Variable.new(:num_examples, type: :integer, range: {0, 10}),
      Variable.new(:include_reasoning, type: :boolean),
      Variable.new(:response_format, type: :choice, range: [:json, :text])
    ])

    # Run optimization
    {:ok, optimal} = Crucible.DSPy.Optimizer.optimize(
      variable_space,
      program: program_fn,
      dataset: training_data,
      validation: validation_data,
      n_trials: opts[:n_trials] || 50
    )

    %{
      optimal_config: optimal.best_config,
      improvement: optimal.best_score - optimal.initial_score,
      optimization_history: optimal.trials,
      ready_for_validation: true
    }
  end

  # Stage 3: Validation - Statistical validation before production
  def validate_for_production(program_fn, optimal_config, validation_dataset) do
    # Run with optimal configuration (multiple repetitions)
    repetitions = 5

    results = Enum.map(1..repetitions, fn _i ->
      evaluate_program(program_fn, validation_dataset, optimal_config)
    end)

    # Statistical analysis
    accuracy_samples = Enum.map(results, & &1.accuracy)
    latency_samples = Enum.map(results, & &1.latency_p95)
    cost_samples = Enum.map(results, & &1.cost_per_query)

    accuracy_ci = Bench.confidence_interval(accuracy_samples, 0.95)

    # Production requirements check
    production_ready = accuracy_ci.lower > 0.85 and
                      Enum.mean(latency_samples) < 1000 and
                      Enum.mean(cost_samples) < 0.01

    %{
      accuracy: %{
        mean: Enum.mean(accuracy_samples),
        ci_95: accuracy_ci,
        meets_requirement: accuracy_ci.lower > 0.85
      },
      latency: %{
        mean: Enum.mean(latency_samples),
        p95: Statistics.percentile(latency_samples, 0.95),
        meets_requirement: Enum.mean(latency_samples) < 1000
      },
      cost: %{
        mean: Enum.mean(cost_samples),
        meets_requirement: Enum.mean(cost_samples) < 0.01
      },
      production_ready: production_ready,
      recommendation: if production_ready, do: :deploy, else: :further_optimization
    }
  end

  # Stage 4: A/B Test in Production
  def production_ab_test(current_program, new_program, opts \\ []) do
    # Design A/B test
    test_design = Crucible.DSPy.ProductionTest.design_ab_test(
      name: "DSPy Program A/B Test",
      variants: [
        %{name: "control", program: current_program},
        %{name: "treatment", program: new_program}
      ],
      traffic_split: %{control: 0.5, treatment: 0.5},
      duration_days: opts[:duration_days] || 7,
      primary_metric: :accuracy,
      guardrail_metrics: [:error_rate, :latency_p99, :cost_per_query]
    )

    # Monitor and analyze (called after test runs)
    fn production_data ->
      Crucible.DSPy.ProductionTest.analyze_results(test_design, production_data)
    end
  end

  # Stage 5: Continuous Monitoring
  def monitor_dspy_performance(program_id, opts \\ []) do
    # Setup continuous monitoring
    monitoring = ProductionEval.ContinuousMonitoring.setup_monitoring(%{
      model_id: program_id,
      check_interval: :hourly,

      metrics: [
        %{name: :accuracy, alert_threshold: 0.03},
        %{name: :latency_p95, alert_threshold: 1.2},
        %{name: :cost_per_query, alert_threshold: 1.5}
      ],

      degradation_detection: %{
        alpha: 0.01,
        consecutive_violations: 3,
        lookback_days: 7
      }
    })

    {:ok, monitoring}
  end
end
```

## Implementation: New Package - `crucible_production_eval`

### Package Structure

```
crucible_production_eval/
├── lib/
│   └── crucible_production_eval/
│       ├── deployment_gate.ex         # Pre-deployment validation
│       ├── continuous_monitoring.ex   # Production monitoring
│       ├── prompt_optimization.ex     # LLM prompt optimization
│       ├── ensemble_optimization.ex   # Ensemble strategy optimization
│       ├── multi_variant_test.ex      # Multi-variant A/B testing
│       ├── retraining_trigger.ex      # Automated retraining decisions
│       ├── dspy_lifecycle.ex          # DSPy program lifecycle
│       └── telemetry_import.ex        # Production telemetry bridge
├── test/
│   └── crucible_production_eval/
│       ├── deployment_gate_test.exs
│       ├── continuous_monitoring_test.exs
│       ├── prompt_optimization_test.exs
│       └── ...
├── examples/
│   ├── 01_deployment_validation.exs
│   ├── 02_continuous_monitoring.exs
│   ├── 03_prompt_optimization.exs
│   ├── 04_ab_test_analysis.exs
│   └── 05_full_lifecycle.exs
├── README.md
└── mix.exs
```

### Dependencies

```elixir
defp deps do
  [
    # Core Crucible libraries
    {:crucible_bench, "~> 1.0"},
    {:crucible_ensemble, "~> 1.0"},
    {:crucible_hedging, "~> 1.0"},
    {:crucible_telemetry, "~> 1.0"},
    {:crucible_datasets, "~> 1.0"},
    {:crucible_harness, "~> 1.0"},

    # New dependencies
    {:crucible_variables, "~> 1.0"},  # For optimization

    # Utilities
    {:jason, "~> 1.4"},
    {:req, "~> 0.4"},  # For telemetry import
    {:ex_doc, "~> 0.31", only: :dev}
  ]
end
```

## Novel Value Proposition

### What Makes This Different

**Existing Tools:**
- MLflow: Experiment tracking, not statistical rigor
- Weights & Biases: Visualization, not rigorous testing
- Evidently AI: Drift detection, not comprehensive evaluation
- Great Expectations: Data validation, not ML-specific

**Crucible Production Eval:**
- ✅ **Statistical rigor** for ALL production decisions
- ✅ **Pre-deployment gates** with confidence intervals
- ✅ **Continuous monitoring** with significance testing
- ✅ **Systematic optimization** not trial-and-error
- ✅ **Proper A/B testing** with power analysis and corrections
- ✅ **Automated triggers** with statistical confidence
- ✅ **Production-ready** on BEAM/OTP for reliability

### The Killer Feature

**Make production ML decisions with scientific confidence:**

```elixir
# Every production ML decision backed by statistics
deployment_decision = """
Recommendation: DEPLOY

Statistical Evidence:
• Accuracy improvement: +3.2% (CI: [+2.1%, +4.3%])
• Effect size: d=0.54 (medium effect)
• P-value: p=0.0012 (highly significant)
• Power: 0.94 (well-powered study)
• Sample size: n=1,247 per group

Guardrails:
✅ Latency P95: 423ms vs 445ms (within threshold)
✅ Cost: $0.0023 vs $0.0021 (+9%, acceptable)
✅ Error rate: 0.8% vs 0.9% (no increase)

Confidence: 95% confidence that improvement is real and will persist in production.

Generated by Crucible Framework #{timestamp}
"""
```

## Integration Pattern for Any Production System

### Generic Integration Interface

```elixir
defmodule MyProductionSystem.CrucibleIntegration do
  @moduledoc """
  Template for integrating any production ML system with Crucible.
  """

  # 1. Provide telemetry export
  def export_telemetry(time_range) do
    # Query your production metrics store
    events = MyProductionSystem.Metrics.query(time_range)

    # Convert to Crucible format
    Enum.map(events, fn event ->
      %{
        event: normalize_event_name(event.type),
        measurements: %{
          duration: event.latency_ms,
          value: event.metric_value
        },
        metadata: %{
          model_id: event.model,
          version: event.version,
          # ... sanitize any sensitive data
        },
        timestamp: event.timestamp
      }
    end)
  end

  # 2. Wrap models for evaluation
  def wrap_model_for_evaluation(model_id) do
    fn input, config ->
      # Call your production model
      result = MyProductionSystem.Models.predict(model_id, input, config)

      # Return in standard format
      %{
        prediction: result.output,
        confidence: result.confidence,
        metadata: %{
          latency_ms: result.latency,
          cost_usd: result.cost
        }
      }
    end
  end

  # 3. Define evaluation metrics
  def evaluation_metrics do
    [
      accuracy: &MyProductionSystem.Metrics.calculate_accuracy/2,
      precision: &MyProductionSystem.Metrics.calculate_precision/2,
      recall: &MyProductionSystem.Metrics.calculate_recall/2,
      f1_score: &MyProductionSystem.Metrics.calculate_f1/2,
      domain_specific_metric: &MyProductionSystem.Metrics.custom_metric/2
    ]
  end

  # 4. Run validation experiment
  def validate_new_model(candidate_model_id) do
    ProductionEval.DeploymentGate.validate_deployment(
      candidate: wrap_model_for_evaluation(candidate_model_id),
      current: wrap_model_for_evaluation(MyProductionSystem.Models.current_production_id()),
      dataset: MyProductionSystem.TestData.load_validation_set(),
      metrics: evaluation_metrics(),
      requirements: MyProductionSystem.Requirements.production_standards()
    )
  end

  # 5. Setup continuous monitoring
  def setup_monitoring do
    ProductionEval.ContinuousMonitoring.setup_monitoring(%{
      model_id: MyProductionSystem.Models.current_production_id(),
      telemetry_export_fn: &export_telemetry/1,
      baseline_fn: &establish_baseline/0,
      alert_fn: &MyProductionSystem.Alerts.send_alert/1
    })
  end
end
```

## Timeline for Production Evaluation Package

### Week 1-2: Core Infrastructure
- [ ] Telemetry import (generic interface)
- [ ] Deployment gate (pre-deployment validation)
- [ ] Basic monitoring (statistical degradation detection)
- [ ] Integration examples

### Week 3-4: Optimization Tools
- [ ] Prompt optimization framework
- [ ] Ensemble configuration optimizer
- [ ] Variable system integration
- [ ] Optimization examples

### Week 5-6: Advanced Testing
- [ ] Multi-variant test framework
- [ ] Sequential testing (early stopping)
- [ ] Multiple comparison corrections
- [ ] A/B test examples

### Week 7-8: Lifecycle Management
- [ ] Retraining trigger system
- [ ] DSPy lifecycle tools
- [ ] Automated reporting
- [ ] Full lifecycle examples

## Success Criteria

### Technical
- [ ] Works with any production system (generic interface)
- [ ] Statistical tests are rigorous and correct
- [ ] Performance overhead <100ms per evaluation
- [ ] Handles 10k+ events per analysis

### Production Systems
- [ ] Can make deployment decisions with confidence
- [ ] Continuous monitoring catches regressions
- [ ] Optimization improves metrics measurably
- [ ] Reports convince stakeholders

### Adoption
- [ ] 3+ production systems using Crucible
- [ ] 20+ deployment decisions validated
- [ ] 0 false positive degradation alerts
- [ ] Documented cost savings from optimization

---

**Status**: Novel Use Case Proposal
**Timeline**: 8 weeks for complete package
**Value Proposition**: Scientific rigor for production ML decisions
**Target Audience**: ML teams running production systems
**Unique Selling Point**: Statistical confidence in every production decision
