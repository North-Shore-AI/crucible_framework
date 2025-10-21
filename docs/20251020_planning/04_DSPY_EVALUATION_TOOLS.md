# DSPy Evaluation Tools for Crucible Framework
## Date: 2025-10-20

## Vision

Enable scientific, statistically rigorous evaluation of DSPy programs. Transform DSPy development from "try things and see" to "systematic optimization with confidence intervals."

## Core Concept

**Crucible treats DSPy programs as black-box functions** and applies rigorous scientific methodology:

- Statistical comparison of program variants
- Systematic hyperparameter optimization
- Teleprompter strategy evaluation
- Production A/B testing with proper experimental design

## What Crucible Does NOT Need to Do

Crucible is NOT reimplementing DSPy. We assume:
- ✅ Users have DSPy programs (implemented elsewhere)
- ✅ Users have LLM provider access (configured elsewhere)
- ✅ Users have training/validation data (stored elsewhere)

Crucible ONLY provides:
- Scientific evaluation methodology
- Statistical rigor
- Systematic optimization
- Reproducible experimentation

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     User's DSPy Programs                          │
│                                                                    │
│  • Implemented in Python or Elixir                                │
│  • Connected to LLM providers                                     │
│  • Production or research code                                    │
│                                                                    │
│  Provides to Crucible:                                            │
│  → Callable function: input → output                              │
│  → Configuration parameters (temperature, examples, etc.)         │
│  → Evaluation dataset                                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              Crucible DSPy Evaluation (NEW)                       │
│                                                                    │
│  Black-Box Evaluation:                                            │
│  → Execute program with different configs                         │
│  → Collect results (accuracy, cost, latency)                      │
│  → Statistical comparison (t-tests, effect sizes, CIs)            │
│                                                                    │
│  Systematic Optimization:                                         │
│  → Define parameter search space (Variables)                      │
│  → Grid/random/Bayesian search                                    │
│  → Track optimization history                                     │
│  → Reproduce best configuration                                   │
│                                                                    │
│  Production Integration:                                          │
│  → A/B test different DSPy variants                               │
│  → Monitor production performance                                 │
│  → Trigger optimization when degraded                             │
│                                                                    │
│  Uses Existing Crucible:                                          │
│  → Bench for statistical tests                                    │
│  → ResearchHarness for orchestration                              │
│  → TelemetryResearch for tracking                                 │
│  → Variable system for optimization                               │
└──────────────────────────────────────────────────────────────────┘
```

## Core Modules

### Module 1: DSPy Program Wrapper

```elixir
defmodule Crucible.DSPy.Program do
  @moduledoc """
  Wrapper for any DSPy program as a black-box evaluable function.

  ## Example

      # Wrap a Python DSPy program
      program = DSPy.Program.new(
        name: "QAProgram",
        execute_fn: fn input, config ->
          # Call Python DSPy via Port or HTTP
          call_python_dspy(input, config)
        end,
        variables: [
          Variable.new(:temperature, type: :float, range: {0.0, 2.0}),
          Variable.new(:num_examples, type: :integer, range: {0, 10})
        ]
      )

      # Or wrap an Elixir DSPy implementation
      program = DSPy.Program.new(
        name: "ChainOfThought",
        execute_fn: &MyElixirDSPy.ChainOfThought.run/2,
        variables: [...]
      )
  """

  @type config :: %{
    optional(:temperature) => float(),
    optional(:max_tokens) => integer(),
    optional(:num_examples) => integer(),
    optional(:model) => String.t(),
    optional(:provider) => atom(),
    optional(atom()) => any()
  }

  @type result :: %{
    answer: any(),
    optional(:confidence) => float(),
    optional(:reasoning) => String.t(),
    optional(:metadata) => map()
  }

  @type execute_fn :: (input :: any(), config()) :: {:ok, result()} | {:error, term()}

  defstruct [
    :name,
    :execute_fn,
    :variables,
    :description,
    :metadata
  ]

  @doc "Create a DSPy program wrapper"
  def new(opts) do
    %__MODULE__{
      name: Keyword.fetch!(opts, :name),
      execute_fn: Keyword.fetch!(opts, :execute_fn),
      variables: Keyword.get(opts, :variables, []),
      description: Keyword.get(opts, :description),
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end

  @doc "Execute the program with given configuration"
  def execute(%__MODULE__{} = program, input, config \\ %{}) do
    start_time = System.monotonic_time(:millisecond)

    result = try do
      program.execute_fn.(input, config)
    catch
      kind, error ->
        {:error, {kind, error, __STACKTRACE__}}
    end

    end_time = System.monotonic_time(:millisecond)
    latency = end_time - start_time

    # Enrich result with metadata
    case result do
      {:ok, output} ->
        {:ok, Map.put(output, :_crucible_metadata, %{
          latency_ms: latency,
          timestamp: DateTime.utc_now(),
          config: config
        })}

      error ->
        error
    end
  end

  @doc "Extract configurable variables from program"
  def variables(%__MODULE__{variables: vars}), do: vars
end
```

### Module 2: DSPy Evaluation Framework

```elixir
defmodule Crucible.DSPy.Evaluation do
  @moduledoc """
  Evaluate DSPy programs with statistical rigor.
  """

  @doc """
  Compare multiple DSPy program configurations.

  Returns statistical comparison with confidence intervals.

  ## Example

      configs = [
        %{name: "low_temp", config: %{temperature: 0.3, num_examples: 3}},
        %{name: "mid_temp", config: %{temperature: 0.7, num_examples: 5}},
        %{name: "high_temp", config: %{temperature: 1.2, num_examples: 3}}
      ]

      {:ok, comparison} = DSPy.Evaluation.compare_configurations(
        program: my_dspy_program,
        configurations: configs,
        dataset: test_dataset,
        metrics: [:accuracy, :cost_per_query],
        repetitions: 5  # For statistical stability
      )

      # comparison includes:
      # - Mean and CI for each config
      # - Statistical tests (ANOVA + post-hoc)
      # - Effect sizes
      # - Best configuration with confidence
  """
  def compare_configurations(opts) do
    program = Keyword.fetch!(opts, :program)
    configurations = Keyword.fetch!(opts, :configurations)
    dataset = Keyword.fetch!(opts, :dataset)
    metrics = Keyword.get(opts, :metrics, [:accuracy])
    repetitions = Keyword.get(opts, :repetitions, 3)

    # Run each configuration multiple times
    results = Enum.map(configurations, fn config_spec ->
      run_configuration_with_repetitions(
        program,
        config_spec,
        dataset,
        metrics,
        repetitions
      )
    end)

    # Statistical analysis
    statistical_analysis = analyze_configurations(results, metrics)

    # Generate report
    report = generate_comparison_report(results, statistical_analysis)

    {:ok, %{
      results: results,
      statistical_analysis: statistical_analysis,
      best_configuration: select_best_configuration(results, statistical_analysis),
      report: report
    }}
  end

  defp run_configuration_with_repetitions(program, config_spec, dataset, metrics, reps) do
    # Multiple repetitions for statistical stability
    repetition_results = Enum.map(1..reps, fn rep_num ->
      # Execute program on entire dataset with this config
      predictions = Enum.map(dataset, fn item ->
        case DSPy.Program.execute(program, item.input, config_spec.config) do
          {:ok, result} ->
            %{
              input: item.input,
              expected: item.expected,
              predicted: result.answer,
              metadata: result._crucible_metadata
            }

          {:error, reason} ->
            %{input: item.input, expected: item.expected, error: reason}
        end
      end)

      # Calculate metrics for this repetition
      metric_values = Enum.map(metrics, fn metric ->
        {metric, calculate_metric(metric, predictions)}
      end) |> Map.new()

      %{repetition: rep_num, metrics: metric_values, predictions: predictions}
    end)

    # Aggregate across repetitions
    %{
      configuration: config_spec,
      repetitions: repetition_results,
      aggregated_metrics: aggregate_repetitions(repetition_results, metrics)
    }
  end

  defp analyze_configurations(results, metrics) do
    # For each metric, perform ANOVA + post-hoc tests
    Enum.map(metrics, fn metric ->
      # Extract samples for each configuration
      samples_per_config = Enum.map(results, fn result ->
        result.repetitions
        |> Enum.map(& &1.metrics[metric])
      end)

      # ANOVA test
      {:ok, anova} = Bench.anova(samples_per_config,
        labels: Enum.map(results, & &1.configuration.name)
      )

      # Post-hoc pairwise comparisons (if ANOVA significant)
      pairwise = if anova.significant? do
        perform_pairwise_comparisons(samples_per_config, results)
      else
        []
      end

      # Effect sizes
      effect_sizes = calculate_all_effect_sizes(samples_per_config, results)

      {metric, %{
        anova: anova,
        pairwise_comparisons: pairwise,
        effect_sizes: effect_sizes
      }}
    end) |> Map.new()
  end
end
```

### Module 3: DSPy Optimizer

```elixir
defmodule Crucible.DSPy.Optimizer do
  @moduledoc """
  Systematically optimize DSPy program parameters.
  """

  @doc """
  Optimize a DSPy program's configuration.

  ## Example

      # Define search space
      space = Variable.Space.new("dspy_optimization", [
        Variable.new(:temperature, type: :float, range: {0.0, 2.0}, default: 0.7),
        Variable.new(:max_tokens, type: :integer, range: {100, 2000}, default: 500),
        Variable.new(:num_examples, type: :integer, range: {0, 10}, default: 3),
        Variable.new(:include_cot, type: :boolean, default: true)
      ])

      # Optimize
      {:ok, result} = DSPy.Optimizer.optimize(
        program: my_dspy_program,
        variable_space: space,
        training_data: train_set,
        validation_data: val_set,
        objective: :accuracy,  # or custom objective fn
        strategy: :bayesian,   # or :grid, :random, :evolutionary
        n_trials: 50
      )

      # result.best_config is statistically validated optimal configuration
  """
  def optimize(opts) do
    program = Keyword.fetch!(opts, :program)
    variable_space = Keyword.fetch!(opts, :variable_space)
    training_data = Keyword.fetch!(opts, :training_data)
    validation_data = Keyword.fetch!(opts, :validation_data)
    objective = Keyword.get(opts, :objective, :accuracy)
    strategy = Keyword.get(opts, :strategy, :bayesian)
    n_trials = Keyword.get(opts, :n_trials, 50)

    # Define objective function
    objective_fn = create_objective_function(
      program,
      training_data,
      validation_data,
      objective
    )

    # Run optimization strategy
    {:ok, optimization_result} = case strategy do
      :grid ->
        Variable.Optimizer.grid_search(variable_space, objective_fn,
          resolution: opts[:grid_resolution] || 3)

      :random ->
        Variable.Optimizer.random_search(variable_space, objective_fn,
          n_trials: n_trials)

      :bayesian ->
        Variable.Optimizer.bayesian_optimize(variable_space, objective_fn,
          n_trials: n_trials,
          n_initial_random: div(n_trials, 5))

      :evolutionary ->
        Variable.Optimizer.evolutionary_optimize(variable_space, objective_fn,
          population_size: 20,
          generations: div(n_trials, 20))
    end

    # Validate best configuration on fresh data
    {:ok, validation} = validate_optimal_configuration(
      program,
      optimization_result.best_config,
      validation_data
    )

    {:ok, %{
      best_config: optimization_result.best_config,
      best_score: optimization_result.best_score,
      optimization_history: optimization_result.trials,
      validation: validation,
      strategy_used: strategy,
      metadata: %{
        total_trials: length(optimization_result.trials),
        convergence_iteration: find_convergence_point(optimization_result.trials),
        improvement_over_default: calculate_improvement(optimization_result)
      }
    }}
  end

  defp create_objective_function(program, training_data, validation_data, objective_spec) do
    fn config ->
      # Evaluate on validation data (not training - avoid overfitting)
      predictions = Enum.map(validation_data, fn item ->
        case DSPy.Program.execute(program, item.input, config) do
          {:ok, result} ->
            %{
              correct: evaluate_correctness(result.answer, item.expected),
              cost: estimate_cost(result, config),
              latency: result._crucible_metadata.latency_ms
            }
          {:error, _} ->
            %{correct: false, cost: 0, latency: 0}
        end
      end)

      # Calculate objective based on spec
      case objective_spec do
        :accuracy ->
          Enum.count(predictions, & &1.correct) / length(predictions)

        :cost_efficiency ->
          accuracy = Enum.count(predictions, & &1.correct) / length(predictions)
          avg_cost = Enum.map(predictions, & &1.cost) |> Enum.mean()
          # Maximize accuracy per dollar
          accuracy / (avg_cost + 0.0001)

        :latency_optimized ->
          accuracy = Enum.count(predictions, & &1.correct) / length(predictions)
          avg_latency = Enum.map(predictions, & &1.latency) |> Enum.mean()
          # Maximize accuracy per millisecond
          accuracy / (avg_latency / 1000)

        custom_fn when is_function(custom_fn) ->
          custom_fn.(predictions, config)
      end
    end
  end

  @doc """
  Compare different teleprompter strategies scientifically.

  ## Example

      # User provides compilation functions for each teleprompter
      teleprompter_compilers = %{
        bootstrap: &MyDSPy.Bootstrap.compile/2,
        copro: &MyDSPy.COPRO.compile/2,
        mipro: &MyDSPy.MIPRO.compile/2
      }

      {:ok, comparison} = DSPy.Optimizer.compare_teleprompters(
        base_program: my_program,
        training_data: train_set,
        validation_data: val_set,
        teleprompters: teleprompter_compilers,
        metrics: [:accuracy, :optimization_time, :final_cost]
      )

      # comparison.best_teleprompter selected with statistical confidence
  """
  def compare_teleprompters(opts) do
    base_program = Keyword.fetch!(opts, :base_program)
    training_data = Keyword.fetch!(opts, :training_data)
    validation_data = Keyword.fetch!(opts, :validation_data)
    teleprompters = Keyword.fetch!(opts, :teleprompters)
    metrics = Keyword.get(opts, :metrics, [:accuracy])

    # Compile program with each teleprompter (multiple runs for stability)
    repetitions = Keyword.get(opts, :repetitions, 3)

    results = Enum.map(teleprompters, fn {teleprompter_name, compile_fn} ->
      repetition_results = Enum.map(1..repetitions, fn _rep ->
        # Time the compilation
        {compile_time, {:ok, compiled_program}} = :timer.tc(fn ->
          compile_fn.(base_program, training_data)
        end)

        # Evaluate compiled program
        evaluation = evaluate_compiled_program(compiled_program, validation_data, metrics)

        %{
          compile_time_ms: div(compile_time, 1000),
          evaluation: evaluation
        }
      end)

      %{
        teleprompter: teleprompter_name,
        repetitions: repetition_results,
        aggregated: aggregate_teleprompter_results(repetition_results, metrics)
      }
    end)

    # Statistical comparison
    statistical_analysis = Enum.map(metrics, fn metric ->
      samples_per_teleprompter = Enum.map(results, fn result ->
        result.repetitions |> Enum.map(& &1.evaluation.metrics[metric])
      end)

      {:ok, comparison} = Bench.anova(samples_per_teleprompter,
        labels: Enum.map(results, & &1.teleprompter)
      )

      {metric, comparison}
    end) |> Map.new()

    {:ok, %{
      results: results,
      statistical_analysis: statistical_analysis,
      best_teleprompter: select_best_teleprompter(results, statistical_analysis),
      recommendation: generate_teleprompter_recommendation(results, statistical_analysis)
    }}
  end
end
```

### Module 3: Production Testing

```elixir
defmodule Crucible.DSPy.ProductionTest do
  @moduledoc """
  Rigorous A/B testing for DSPy programs in production.
  """

  @doc """
  Design a statistically sound A/B test for DSPy variants.

  ## Example

      test = DSPy.ProductionTest.design_test(
        name: "Prompt Template Comparison",
        control_variant: %{
          name: "current_prompt",
          config: current_config
        },
        treatment_variants: [
          %{name: "cot_prompt", config: cot_config},
          %{name: "react_prompt", config: react_config}
        ],
        primary_metric: :accuracy,
        secondary_metrics: [:latency_p95, :cost_per_query],
        guardrails: [
          %{metric: :error_rate, max: 0.01},
          %{metric: :latency_p99, max: 2000}
        ],
        statistical_config: %{
          alpha: 0.05,
          power: 0.8,
          min_effect_size: 0.02
        },
        traffic_split: :balanced  # or custom %{control: 0.4, treat1: 0.3, treat2: 0.3}
      )

      # test includes required sample size from power analysis
  """
  def design_test(opts) do
    # Calculate required sample size
    sample_size = calculate_required_sample_size(opts)

    # Design test
    %{
      name: Keyword.fetch!(opts, :name),
      control: Keyword.fetch!(opts, :control_variant),
      treatments: Keyword.fetch!(opts, :treatment_variants),

      # Metrics
      primary_metric: Keyword.fetch!(opts, :primary_metric),
      secondary_metrics: Keyword.get(opts, :secondary_metrics, []),
      guardrail_metrics: Keyword.get(opts, :guardrails, []),

      # Statistical design
      sample_size_per_variant: sample_size,
      total_sample_size: sample_size * (1 + length(Keyword.fetch!(opts, :treatment_variants))),
      alpha: opts[:statistical_config][:alpha] || 0.05,
      power: opts[:statistical_config][:power] || 0.8,
      min_effect_size: opts[:statistical_config][:min_effect_size] || 0.02,

      # Traffic allocation
      traffic_split: calculate_traffic_split(opts),

      # Advanced options
      early_stopping: Keyword.get(opts, :early_stopping, false),
      sequential_testing: Keyword.get(opts, :sequential_testing, false),
      multiple_comparison_correction: Keyword.get(opts, :correction, :bonferroni)
    }
  end

  @doc """
  Analyze A/B test results with proper statistical methodology.
  """
  def analyze_test_results(test_design, collected_data) do
    # Validate sample sizes
    sample_sizes = validate_sample_sizes(test_design, collected_data)

    unless sample_sizes.adequate? do
      return {:error, :insufficient_data, sample_sizes.details}
    end

    # Group by variant
    control_data = extract_variant_data(collected_data, test_design.control.name)
    treatment_data = Enum.map(test_design.treatments, fn treatment ->
      {treatment.name, extract_variant_data(collected_data, treatment.name)}
    end) |> Map.new()

    # Primary metric analysis
    primary_analysis = analyze_primary_metric(
      test_design.primary_metric,
      control_data,
      treatment_data,
      test_design
    )

    # Secondary metrics (for context, not decision)
    secondary_analyses = Enum.map(test_design.secondary_metrics, fn metric ->
      {metric, analyze_secondary_metric(metric, control_data, treatment_data)}
    end) |> Map.new()

    # Guardrail checks (must pass for deployment)
    guardrail_results = check_guardrails(
      test_design.guardrail_metrics,
      treatment_data
    )

    # Make recommendation
    recommendation = make_ab_test_recommendation(
      primary_analysis,
      guardrail_results,
      test_design
    )

    {:ok, %{
      primary_analysis: primary_analysis,
      secondary_analyses: secondary_analyses,
      guardrail_results: guardrail_results,
      recommendation: recommendation,
      confidence: calculate_recommendation_confidence(primary_analysis, guardrail_results)
    }}
  end

  defp analyze_primary_metric(metric, control_data, treatment_data, test_design) do
    control_samples = extract_metric_samples(control_data, metric)

    # Compare each treatment to control
    treatment_comparisons = Enum.map(treatment_data, fn {treatment_name, data} ->
      treatment_samples = extract_metric_samples(data, metric)

      # Statistical test
      {:ok, test_result} = Bench.compare(control_samples, treatment_samples,
        alpha: test_design.alpha
      )

      # Effect size
      effect_size = Bench.effect_size(control_samples, treatment_samples)

      # Practical significance
      practically_significant = abs(effect_size.cohens_d) >= test_design.min_effect_size

      %{
        treatment: treatment_name,
        statistical_test: test_result,
        effect_size: effect_size,
        statistically_significant: test_result.significant?,
        practically_significant: practically_significant,
        both_significant: test_result.significant? and practically_significant
      }
    end)

    # Apply multiple comparison correction if multiple treatments
    if length(treatment_comparisons) > 1 do
      apply_multiple_comparison_correction(treatment_comparisons, test_design)
    else
      %{
        metric: metric,
        control_mean: Enum.mean(control_samples),
        control_ci: Bench.confidence_interval(control_samples, 1 - test_design.alpha),
        comparisons: treatment_comparisons
      }
    end
  end

  defp make_ab_test_recommendation(primary_analysis, guardrail_results, test_design) do
    # Find winning variant (if any)
    winners = primary_analysis.comparisons
    |> Enum.filter(& &1.both_significant)
    |> Enum.sort_by(& &1.effect_size.cohens_d, :desc)

    # Check guardrails
    all_guardrails_passed = Enum.all?(guardrail_results, & &1.passed?)

    cond do
      # Clear winner with guardrails passed
      length(winners) > 0 and all_guardrails_passed ->
        winner = List.first(winners)
        %{
          action: :deploy_winner,
          winner: winner.treatment,
          confidence: :high,
          rationale: """
          #{winner.treatment} shows statistically significant improvement
          (p=#{winner.statistical_test.p_value}, d=#{winner.effect_size.cohens_d})
          with all guardrails passing.
          """
        }

      # Winners but guardrails failed
      length(winners) > 0 and not all_guardrails_passed ->
        %{
          action: :reject_all,
          confidence: :high,
          rationale: """
          Treatment variants show improvement on primary metric but failed
          guardrail checks: #{summarize_failed_guardrails(guardrail_results)}
          """
        }

      # No statistically significant winners
      true ->
        %{
          action: :keep_control,
          confidence: :medium,
          rationale: """
          No treatment variant showed statistically significant improvement
          over control. Consider: 1) larger sample size, 2) different variants,
          or 3) control is already optimal.
          """
        }
    end
  end
end
```

### Module 4: Lifecycle Integration

```elixir
defmodule Crucible.DSPy.Lifecycle do
  @moduledoc """
  Complete DSPy program lifecycle: develop → optimize → validate → deploy → monitor.
  """

  @doc """
  Execute complete DSPy program lifecycle with scientific rigor.

  ## Example

      lifecycle = DSPy.Lifecycle.new(
        program: MyDSPyProgram,
        development_data: dev_set,
        validation_data: val_set,
        production_config: prod_config
      )

      # Guided workflow
      {:ok, result} = DSPy.Lifecycle.execute(lifecycle, [
        :initial_evaluation,
        :optimization,
        :validation,
        :ab_test_design,
        :deployment_recommendation
      ])
  """
  def execute(lifecycle_spec, stages) do
    initial_state = %{
      program: lifecycle_spec.program,
      current_config: nil,
      optimization_result: nil,
      validation_result: nil,
      ab_test_design: nil,
      recommendation: nil
    }

    # Execute each stage in sequence
    final_state = Enum.reduce(stages, {:ok, initial_state}, fn stage, {:ok, state} ->
      execute_stage(stage, state, lifecycle_spec)
    end)

    case final_state do
      {:ok, state} ->
        {:ok, generate_lifecycle_report(state, lifecycle_spec)}
      error ->
        error
    end
  end

  defp execute_stage(:initial_evaluation, state, lifecycle) do
    # Evaluate with default configuration
    default_config = extract_default_config(state.program)

    {:ok, eval} = DSPy.Evaluation.evaluate(
      state.program,
      dataset: lifecycle.development_data,
      config: default_config,
      repetitions: 3
    )

    {:ok, %{state |
      current_config: default_config,
      initial_evaluation: eval
    }}
  end

  defp execute_stage(:optimization, state, lifecycle) do
    # Extract variable space from program
    variable_space = DSPy.Program.variables(state.program)
    |> Variable.Space.new("optimization")

    {:ok, optimization} = DSPy.Optimizer.optimize(
      program: state.program,
      variable_space: variable_space,
      training_data: lifecycle.development_data,
      validation_data: lifecycle.validation_data,
      strategy: lifecycle.optimization_strategy || :bayesian,
      n_trials: lifecycle.n_optimization_trials || 50
    )

    {:ok, %{state |
      current_config: optimization.best_config,
      optimization_result: optimization
    }}
  end

  defp execute_stage(:validation, state, lifecycle) do
    # Statistical validation on fresh data
    {:ok, validation} = DSPy.Optimizer.validate_optimal_configuration(
      state.program,
      state.current_config,
      lifecycle.validation_data
    )

    {:ok, %{state | validation_result: validation}}
  end

  defp execute_stage(:ab_test_design, state, lifecycle) do
    # Design production A/B test
    ab_test = DSPy.ProductionTest.design_test(
      name: "Production Deployment Test",
      control_variant: %{
        name: "current_production",
        config: lifecycle.production_config.current
      },
      treatment_variants: [
        %{name: "optimized", config: state.current_config}
      ],
      primary_metric: :accuracy,
      statistical_config: lifecycle.production_config.statistical_requirements || %{},
      guardrails: lifecycle.production_config.guardrails || []
    )

    {:ok, %{state | ab_test_design: ab_test}}
  end

  defp execute_stage(:deployment_recommendation, state, _lifecycle) do
    # Generate final recommendation
    recommendation = %{
      deploy: state.validation_result.production_ready,
      confidence: state.validation_result.confidence,
      optimal_config: state.current_config,
      expected_improvement: calculate_expected_improvement(state),
      ab_test_plan: state.ab_test_design,
      next_steps: generate_next_steps(state)
    }

    {:ok, %{state | recommendation: recommendation}}
  end
end
```

## Implementation Timeline

### Week 1: Core DSPy Evaluation
**Deliverables:**
- [ ] `Crucible.DSPy.Program` wrapper module (black-box interface)
- [ ] `Crucible.DSPy.Evaluation` comparison framework
- [ ] Integration with existing Bench for statistics
- [ ] 10+ tests (program wrapper, evaluation correctness)
- [ ] Example: comparing 3 DSPy configurations on GSM8K

### Week 2: Optimization Framework
**Deliverables:**
- [ ] `Crucible.DSPy.Optimizer` with grid/random/Bayesian search
- [ ] Integration with Variable system
- [ ] Teleprompter comparison framework
- [ ] 10+ tests (optimization convergence, teleprompter comparison)
- [ ] Example: optimizing prompt parameters on MMLU

### Week 3: Production Integration
**Deliverables:**
- [ ] `Crucible.DSPy.ProductionTest` A/B testing framework
- [ ] Multiple comparison corrections
- [ ] Guardrail checking
- [ ] 10+ tests (A/B test design, analysis correctness)
- [ ] Example: production A/B test analysis

### Week 4: Lifecycle Management
**Deliverables:**
- [ ] `Crucible.DSPy.Lifecycle` complete workflow
- [ ] Integration with all previous modules
- [ ] Comprehensive documentation
- [ ] 5+ end-to-end tests
- [ ] Example: complete DSPy lifecycle from development to production

## Success Criteria

### Technical
- [ ] Can evaluate any DSPy program (black-box)
- [ ] Statistical tests are rigorous (peer-review quality)
- [ ] Optimization finds measurably better configurations
- [ ] A/B test analysis provides clear deployment recommendations
- [ ] All functionality works with Python or Elixir DSPy

### User Experience
- [ ] Simple API for common use cases
- [ ] Clear documentation with examples
- [ ] Helpful error messages
- [ ] <5 minutes from install to first evaluation

### Production Validation
- [ ] Used by at least 1 production system
- [ ] At least 10 real DSPy programs evaluated
- [ ] At least 3 production deployments validated
- [ ] Zero false positive degradation alerts

## Example: Complete Workflow

```elixir
defmodule CompleteWorkflowExample do
  @doc """
  Shows complete DSPy program lifecycle with Crucible.
  """

  def run_complete_workflow do
    # Stage 1: Wrap your DSPy program
    program = Crucible.DSPy.Program.new(
      name: "QuestionAnswering",
      execute_fn: fn input, config ->
        # Call your DSPy implementation (Python or Elixir)
        YourDSPy.QA.run(input, config)
      end,
      variables: [
        Variable.new(:temperature, type: :float, range: {0.0, 2.0}),
        Variable.new(:max_tokens, type: :integer, range: {100, 2000}),
        Variable.new(:num_examples, type: :integer, range: {0, 10})
      ]
    )

    # Stage 2: Initial evaluation
    {:ok, initial_eval} = Crucible.DSPy.Evaluation.compare_configurations(
      program: program,
      configurations: [
        %{name: "default", config: %{temperature: 0.7, max_tokens: 500, num_examples: 3}}
      ],
      dataset: load_dev_dataset(),
      metrics: [:accuracy, :cost_per_query]
    )

    IO.puts("Initial accuracy: #{initial_eval.results |> List.first() |> Map.get(:aggregated_metrics) |> Map.get(:accuracy) |> Map.get(:mean)}")

    # Stage 3: Systematic optimization
    {:ok, optimization} = Crucible.DSPy.Optimizer.optimize(
      program: program,
      variable_space: Variable.Space.new(program.variables),
      training_data: load_train_dataset(),
      validation_data: load_validation_dataset(),
      objective: :accuracy,
      strategy: :bayesian,
      n_trials: 30
    )

    IO.puts("Optimized accuracy: #{optimization.best_score}")
    IO.puts("Improvement: +#{(optimization.best_score - initial_eval.results |> List.first() |> Map.get(:aggregated_metrics) |> Map.get(:accuracy) |> Map.get(:mean)) * 100}%")

    # Stage 4: Statistical validation
    {:ok, validation} = Crucible.DSPy.Optimizer.validate_optimal_configuration(
      program,
      optimization.best_config,
      load_test_dataset()
    )

    IO.puts("Validation accuracy: #{validation.accuracy_mean} (95% CI: #{inspect(validation.accuracy_ci)})")

    if validation.production_ready do
      # Stage 5: Design A/B test for production
      ab_test = Crucible.DSPy.ProductionTest.design_test(
        name: "Optimized QA Program",
        control_variant: %{name: "current", config: current_production_config()},
        treatment_variants: [%{name: "optimized", config: optimization.best_config}],
        primary_metric: :accuracy,
        guardrails: [
          %{metric: :error_rate, max: 0.01},
          %{metric: :latency_p99, max: 1000}
        ]
      )

      IO.puts("A/B test designed. Required sample size: #{ab_test.sample_size_per_variant} per variant")
      IO.puts("Run test for #{ab_test.duration_days} days, then call analyze_test_results/2")

      {:ok, %{
        optimized_config: optimization.best_config,
        expected_improvement: optimization.best_score - validation.baseline_accuracy,
        validation: validation,
        ab_test_plan: ab_test,
        ready_for_production: true
      }}
    else
      {:error, :validation_failed, validation}
    end
  end
end
```

---

**Status**: Implementation-Ready Design
**Timeline**: 4 weeks
**Dependencies**: Variable system, Complete harness
**Value**: Scientific DSPy evaluation for production systems
**Next Steps**: Approve design → Implement → Validate with real DSPy programs
