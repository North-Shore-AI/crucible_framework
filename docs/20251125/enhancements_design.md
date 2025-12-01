# CrucibleFramework Enhancement Design Document

**Date:** 2025-11-25
**Version:** 0.3.1 → 0.4.0
**Author:** AI Research Infrastructure Team

---

## Executive Summary

This document outlines a comprehensive set of enhancements to the CrucibleFramework that improve developer ergonomics, experiment validation, error handling, and extensibility while maintaining the framework's core principles of scientific rigor and reproducibility.

### Enhancement Goals

1. **Improved Developer Experience** - Better APIs and helper functions
2. **Enhanced Validation** - Pre-flight validation of experiments
3. **Middleware Architecture** - Cross-cutting concerns (logging, metrics, retries)
4. **Better Error Handling** - Recovery strategies and detailed diagnostics
5. **Observability** - Enhanced tracing and debugging capabilities

### Impact Assessment

- **Breaking Changes:** None (fully backward compatible)
- **New Features:** 5 major capabilities
- **Code Addition:** ~1,500 lines (tests included)
- **Documentation:** Comprehensive inline docs + examples

---

## 1. Analysis: Current Architecture

### Strengths Identified

1. **Clean IR Design** - Experiment IR is backend-agnostic and serializable
2. **Stage-Based Pipeline** - Flexible, composable stage architecture
3. **Comprehensive Tracing** - Excellent decision provenance via CrucibleTrace
4. **Robust Persistence** - Ecto-based persistence with sanitization
5. **Statistical Rigor** - Deep integration with CrucibleBench
6. **Ensemble & Hedging** - Production-ready reliability features

### Gaps and Enhancement Opportunities

#### Gap 1: Pre-Flight Validation
**Problem:** Experiments can fail mid-execution due to configuration errors that could be caught earlier.

**Current Behavior:**
```elixir
# This fails at runtime during backend_call stage
experiment = %Experiment{
  id: "test",
  backend: %BackendRef{id: :nonexistent},  # Invalid backend
  pipeline: [%StageDef{name: :data_load}, %StageDef{name: :backend_call}]
}

{:ok, ctx} = CrucibleFramework.run(experiment)
# Fails during pipeline execution - time and resources wasted
```

**Desired Behavior:**
```elixir
# Validate before execution
case CrucibleFramework.validate(experiment) do
  {:ok, _validation_result} ->
    CrucibleFramework.run(experiment)
  {:error, errors} ->
    IO.inspect(errors, label: "Validation failed")
    # %{
    #   backend: ["Backend :nonexistent not registered"],
    #   pipeline: ["Stage :custom_stage module not found"]
    # }
end
```

#### Gap 2: Context Ergonomics
**Problem:** Common context operations require verbose code.

**Current Approach:**
```elixir
# Adding metrics
new_ctx = %Context{
  ctx | metrics: Map.put(ctx.metrics, :my_metric, value)
}

# Checking if data loaded
has_data = ctx.dataset != nil && ctx.examples != nil

# Getting stage result
case Map.fetch(ctx.metrics, :backend) do
  {:ok, backend_metrics} -> backend_metrics
  :error -> nil
end
```

**Desired Approach:**
```elixir
# Adding metrics
new_ctx = Context.put_metric(ctx, :my_metric, value)

# Checking if data loaded
has_data = Context.has_data?(ctx)

# Getting stage result
backend_metrics = Context.get_metric(ctx, :backend)
```

#### Gap 3: Cross-Cutting Concerns
**Problem:** Logging, timing, retries, and other concerns are scattered across stages.

**Current:** Each stage implements its own logging, timing, error handling:
```elixir
defmodule MyStage do
  def run(ctx, opts) do
    Logger.info("Starting MyStage")
    start_time = System.monotonic_time()

    result = do_work(ctx, opts)

    duration = System.monotonic_time() - start_time
    Logger.info("MyStage completed in #{duration}ms")
    result
  end
end
```

**Desired:** Middleware handles cross-cutting concerns declaratively:
```elixir
# In config or experiment
middleware: [
  Crucible.Pipeline.Middleware.Timing,
  Crucible.Pipeline.Middleware.Logging,
  {Crucible.Pipeline.Middleware.Retry, max_attempts: 3}
]

# Stages focus on business logic only
defmodule MyStage do
  def run(ctx, opts) do
    do_work(ctx, opts)  # No boilerplate
  end
end
```

#### Gap 4: Error Recovery Strategies
**Problem:** Pipeline halts on first error with limited recovery options.

**Current:** Pipeline stops on first error:
```elixir
pipeline: [
  %StageDef{name: :data_load},
  %StageDef{name: :backend_call},  # Fails here
  %StageDef{name: :analysis},      # Never runs
  %StageDef{name: :report}         # Never runs
]
```

**Desired:** Configurable error handling per stage:
```elixir
pipeline: [
  %StageDef{name: :data_load},
  %StageDef{
    name: :backend_call,
    error_strategy: {:retry, max_attempts: 3, backoff: :exponential}
  },
  %StageDef{
    name: :analysis,
    error_strategy: {:continue, default: %{}}  # Use default on error
  },
  %StageDef{name: :report, required: true}  # Must succeed
]
```

#### Gap 5: Enhanced Observability
**Problem:** Difficult to debug complex pipelines with many stages.

**Missing Capabilities:**
- Stage-level performance profiling
- Resource usage tracking
- Visual pipeline execution timeline
- Stage dependency graph visualization

---

## 2. Proposed Enhancements

### Enhancement 1: Experiment Validation Stage

**Module:** `Crucible.Stage.Validate`

**Purpose:** Pre-flight validation of experiment configuration before execution.

**Features:**
1. Backend registration check
2. Stage module resolution
3. Dataset provider verification
4. Pipeline dependency validation
5. Configuration completeness check

**API:**
```elixir
# As a standalone stage
pipeline: [
  %StageDef{name: :validate},  # First stage validates everything
  %StageDef{name: :data_load},
  # ... rest of pipeline
]

# As a standalone function
case Crucible.Stage.Validate.validate_experiment(experiment) do
  {:ok, validation_report} ->
    # %{
    #   backend: :ok,
    #   stages: :ok,
    #   dataset: :ok,
    #   warnings: ["Stage :analysis has no registered adapter"]
    # }
  {:error, errors} ->
    # %{
    #   backend: ["Backend :foo not found"],
    #   stages: ["Stage :bar module not found"]
    # }
end
```

**Validation Rules:**
- Backend ID must be registered in application config
- All stage names must resolve to modules
- Dataset provider must exist if dataset specified
- Required reliability configs must be present
- Output sinks must be valid

**Benefits:**
- Fail fast with clear error messages
- Reduce wasted computation on invalid experiments
- Better developer feedback
- Easier debugging of configuration issues

---

### Enhancement 2: Context Helper Functions

**Module:** `Crucible.Context` (enhanced)

**Purpose:** Provide ergonomic helper functions for common context operations.

**New Functions:**

```elixir
# Metrics management
@spec put_metric(t(), atom(), term()) :: t()
def put_metric(%Context{} = ctx, key, value)

@spec get_metric(t(), atom(), term()) :: term()
def get_metric(%Context{} = ctx, key, default \\ nil)

@spec update_metric(t(), atom(), (term() -> term())) :: t()
def update_metric(%Context{} = ctx, key, update_fn)

@spec merge_metrics(t(), map()) :: t()
def merge_metrics(%Context{} = ctx, metrics)

# Output management
@spec add_output(t(), term()) :: t()
def add_output(%Context{} = ctx, output)

@spec add_outputs(t(), list()) :: t()
def add_outputs(%Context{} = ctx, outputs)

# Artifact management
@spec put_artifact(t(), atom(), term()) :: t()
def put_artifact(%Context{} = ctx, key, artifact)

@spec get_artifact(t(), atom(), term()) :: term()
def get_artifact(%Context{} = ctx, key, default \\ nil)

# Assigns management (Phoenix-style)
@spec assign(t(), atom(), term()) :: t()
def assign(%Context{} = ctx, key, value)

@spec assign(t(), keyword() | map()) :: t()
def assign(%Context{} = ctx, assigns)

# Query functions
@spec has_data?(t()) :: boolean()
def has_data?(%Context{} = ctx)

@spec has_backend_session?(t(), atom()) :: boolean()
def has_backend_session?(%Context{} = ctx, backend_id)

@spec get_backend_session(t(), atom()) :: term() | nil
def get_backend_session(%Context{} = ctx, backend_id)

# Stage completion tracking
@spec mark_stage_complete(t(), atom()) :: t()
def mark_stage_complete(%Context{} = ctx, stage_name)

@spec stage_completed?(t(), atom()) :: boolean()
def stage_completed?(%Context{} = ctx, stage_name)

@spec completed_stages(t()) :: [atom()]
def completed_stages(%Context{} = ctx)
```

**Implementation Strategy:**
- Add helper functions to existing `Crucible.Context` module
- Maintain backward compatibility (no struct changes)
- Comprehensive doctests for each function
- Integration tests with real pipeline execution

---

### Enhancement 3: Pipeline Middleware

**Module:** `Crucible.Pipeline.Middleware`

**Purpose:** Implement middleware pattern for cross-cutting concerns.

**Architecture:**

```elixir
defmodule Crucible.Pipeline.Middleware do
  @moduledoc """
  Behaviour for pipeline middleware.

  Middleware wraps stage execution to handle cross-cutting concerns
  like logging, timing, retries, and observability.
  """

  @callback before_stage(Context.t(), StageDef.t()) :: Context.t()
  @callback after_stage(Context.t(), StageDef.t(), result :: term()) ::
              {:ok, Context.t()} | {:error, term()}
  @callback on_error(Context.t(), StageDef.t(), error :: term()) ::
              {:retry, Context.t()} | {:continue, Context.t()} | {:halt, term()}
end
```

**Built-in Middleware:**

#### 1. `Crucible.Pipeline.Middleware.Timing`
Tracks execution time for each stage.

```elixir
# Adds to context.metrics
%{
  stage_timings: %{
    data_load: %{start: ~U[...], end: ~U[...], duration_ms: 125},
    backend_call: %{start: ~U[...], end: ~U[...], duration_ms: 3421}
  }
}
```

#### 2. `Crucible.Pipeline.Middleware.Logging`
Structured logging for stage execution.

```elixir
# Logs:
# [info] Pipeline stage starting: data_load
# [info] Pipeline stage completed: data_load (125ms)
# [error] Pipeline stage failed: backend_call - :timeout after 3421ms
```

#### 3. `Crucible.Pipeline.Middleware.Retry`
Automatic retry logic with backoff.

```elixir
# Configuration
%{
  max_attempts: 3,
  backoff: :exponential,  # or :linear, :constant
  backoff_ms: 1000,
  retryable_errors: [:timeout, :rate_limited]
}
```

#### 4. `Crucible.Pipeline.Middleware.Telemetry`
Enhanced telemetry events.

```elixir
# Emits events:
[:crucible, :pipeline, :stage, :start]
[:crucible, :pipeline, :stage, :stop]
[:crucible, :pipeline, :stage, :exception]
```

#### 5. `Crucible.Pipeline.Middleware.Checkpoint`
Automatic state checkpointing for long pipelines.

```elixir
# Configuration
%{
  checkpoint_after: [:data_load, :backend_call],
  storage: :ets  # or :file, :database
}
```

**Usage:**

```elixir
# Global middleware (via config)
config :crucible_framework,
  pipeline_middleware: [
    Crucible.Pipeline.Middleware.Timing,
    Crucible.Pipeline.Middleware.Logging,
    {Crucible.Pipeline.Middleware.Retry, max_attempts: 3}
  ]

# Per-experiment middleware
experiment = %Experiment{
  # ...
  metadata: %{
    middleware: [
      Crucible.Pipeline.Middleware.Checkpoint,
      {Crucible.Pipeline.Middleware.Retry, retryable_errors: [:timeout]}
    ]
  }
}
```

**Implementation:**
- Middleware wraps each stage execution
- Executed in order: `before_stage -> stage.run -> after_stage`
- Error handling: `on_error` can retry, continue, or halt
- Composable: middleware can wrap other middleware

---

### Enhancement 4: Enhanced Error Handling

**Module:** `Crucible.Pipeline.ErrorHandler`

**Purpose:** Provide sophisticated error handling and recovery strategies.

**Error Strategies:**

#### 1. Halt (Default)
Stop pipeline immediately on error.

```elixir
%StageDef{
  name: :critical_stage,
  error_strategy: :halt  # Default behavior
}
```

#### 2. Retry
Retry stage execution with backoff.

```elixir
%StageDef{
  name: :flaky_api_call,
  error_strategy: {
    :retry,
    max_attempts: 3,
    backoff: :exponential,
    backoff_ms: 1000,
    retryable: fn error ->
      error in [:timeout, :rate_limited, :service_unavailable]
    end
  }
}
```

#### 3. Continue with Default
Continue pipeline with default value on error.

```elixir
%StageDef{
  name: :optional_enrichment,
  error_strategy: {
    :continue,
    default: %{enriched: false, reason: :unavailable}
  }
}
```

#### 4. Skip
Skip stage and continue pipeline.

```elixir
%StageDef{
  name: :optional_analysis,
  error_strategy: :skip
}
```

#### 5. Fallback
Try alternative implementation on error.

```elixir
%StageDef{
  name: :backend_call,
  error_strategy: {
    :fallback,
    alternative: %StageDef{
      module: Crucible.Stage.BackendCall.Fallback
    }
  }
}
```

**Enhanced Error Types:**

```elixir
defmodule Crucible.Error do
  @type t :: %__MODULE__{
    type: atom(),
    message: String.t(),
    stage: atom(),
    context: map(),
    recoverable?: boolean(),
    retry_after: non_neg_integer() | nil,
    details: map()
  }
end
```

**Error Context Tracking:**

```elixir
# Errors stored with full context
ctx.assigns.errors = [
  %{
    stage: :backend_call,
    attempt: 1,
    error: %Crucible.Error{type: :timeout, ...},
    timestamp: ~U[2025-11-25 20:00:00Z]
  },
  %{
    stage: :backend_call,
    attempt: 2,
    error: %Crucible.Error{type: :rate_limited, ...},
    timestamp: ~U[2025-11-25 20:00:05Z]
  }
]
```

---

### Enhancement 5: Enhanced Observability

**Module:** `Crucible.Pipeline.Profiler`

**Purpose:** Provide detailed performance and resource profiling.

**Features:**

#### 1. Stage Performance Profiling
```elixir
# Automatic profiling via middleware
ctx.metrics.profiling = %{
  data_load: %{
    cpu_time_ms: 45,
    wall_time_ms: 125,
    memory_mb: 12.3,
    reductions: 45_231
  },
  backend_call: %{
    cpu_time_ms: 234,
    wall_time_ms: 3421,
    memory_mb: 45.7,
    reductions: 523_412,
    api_calls: 10,
    avg_api_latency_ms: 342
  }
}
```

#### 2. Pipeline Timeline Export
```elixir
# Generate visual timeline
Crucible.Pipeline.Profiler.export_timeline(ctx, "timeline.html")

# Generates interactive HTML with:
# - Stage execution timeline
# - Concurrent operations visualization
# - Error markers
# - Retry indicators
# - Resource usage graphs
```

#### 3. Dependency Graph
```elixir
# Visualize stage dependencies
Crucible.Pipeline.Analyzer.dependency_graph(experiment)

# Output: Mermaid/DOT format showing:
# - Stage execution order
# - Data dependencies
# - Conditional branches
# - Potential parallelization opportunities
```

#### 4. Bottleneck Detection
```elixir
# Analyze pipeline performance
Crucible.Pipeline.Analyzer.find_bottlenecks(ctx)

# Returns:
%{
  slowest_stages: [
    {:backend_call, 3421, percentage: 85.3},
    {:analysis, 412, percentage: 10.2}
  ],
  memory_peaks: [
    {:backend_call, 145.7},
    {:data_load, 23.4}
  ],
  optimization_suggestions: [
    "Consider batching API calls in backend_call stage",
    "data_load could be streamed to reduce memory"
  ]
}
```

---

## 3. Implementation Plan

### Phase 1: Foundation (Days 1-2)

**Tasks:**
1. Implement Context helper functions
2. Write comprehensive tests for Context helpers
3. Update documentation with examples

**Deliverables:**
- `Crucible.Context` with 20+ helper functions
- 50+ tests covering all helpers
- Inline documentation with doctests

### Phase 2: Validation (Days 3-4)

**Tasks:**
1. Implement `Crucible.Stage.Validate`
2. Add validation rules for all IR components
3. Integrate with pipeline runner
4. Write validation tests

**Deliverables:**
- Working validation stage
- Comprehensive validation rules
- 30+ validation test cases
- Examples in documentation

### Phase 3: Middleware Infrastructure (Days 5-7)

**Tasks:**
1. Design middleware behaviour
2. Update pipeline runner to support middleware
3. Implement Timing middleware
4. Implement Logging middleware
5. Write middleware tests

**Deliverables:**
- Middleware behaviour definition
- Updated runner with middleware support
- 2 core middleware implementations
- 40+ middleware tests

### Phase 4: Error Handling (Days 8-9)

**Tasks:**
1. Implement error strategy system
2. Add retry logic with backoff
3. Add fallback mechanism
4. Write error handling tests

**Deliverables:**
- 5 error strategies implemented
- Enhanced error types
- 35+ error handling tests

### Phase 5: Observability (Days 10-11)

**Tasks:**
1. Implement profiling middleware
2. Add timeline export functionality
3. Add bottleneck detection
4. Write profiling tests

**Deliverables:**
- Profiling infrastructure
- Timeline HTML export
- Analysis tools
- 25+ profiling tests

### Phase 6: Integration & Documentation (Days 12-14)

**Tasks:**
1. Integration testing of all enhancements
2. Performance testing
3. Documentation updates
4. Example experiments
5. CHANGELOG update

**Deliverables:**
- Fully integrated system
- Updated documentation
- 5+ example experiments
- Performance benchmarks

---

## 4. Testing Strategy

### Unit Tests

**Coverage Goals:**
- Context helpers: 100% coverage
- Validation rules: 100% coverage
- Middleware: 95%+ coverage
- Error handling: 95%+ coverage

**Test Categories:**
1. **Happy path tests** - Normal operation
2. **Edge case tests** - Boundary conditions
3. **Error tests** - Failure modes
4. **Property tests** - Invariants (using StreamData)

### Integration Tests

**Scenarios:**
1. Complete pipeline with all middleware
2. Pipeline with validation errors
3. Pipeline with retry scenarios
4. Pipeline with fallback strategies
5. Long-running pipeline with checkpointing

### Performance Tests

**Benchmarks:**
1. Middleware overhead (< 1% per middleware)
2. Context helper performance
3. Validation speed (< 10ms for typical experiment)
4. Memory usage with profiling enabled

---

## 5. Backward Compatibility

**Guarantees:**
- All existing experiments work unchanged
- No breaking changes to public APIs
- Optional features (opt-in)
- Graceful degradation if middleware not configured

**Migration Path:**
- Existing code: zero changes required
- New features: explicitly opt-in via config or experiment metadata
- Documentation: clear upgrade guide with examples

---

## 6. Success Metrics

### Developer Experience
- Reduce boilerplate code by 40%
- Reduce context manipulation verbosity by 60%
- Reduce debugging time by 50% (via better errors and profiling)

### Reliability
- Reduce configuration errors by 80% (via validation)
- Improve error recovery success rate by 70%
- Reduce failed experiments due to transient errors by 60%

### Observability
- 100% of pipeline execution observable
- Performance bottlenecks identifiable within 1 minute
- Complete audit trail for all experiments

---

## 7. Future Enhancements

### Post-0.4.0 Roadmap

1. **Stage Parallelization** - Automatic parallel execution of independent stages
2. **Distributed Pipeline** - Multi-node execution via distributed Elixir
3. **Visual Pipeline Builder** - LiveView-based UI for building experiments
4. **A/B Testing Framework** - Built-in support for comparing configurations
5. **Cost Optimization** - Automatic selection of cost-optimal backends
6. **Fairness Validation** - Integrate ExFairness into validation stage

---

## 8. Code Examples

### Example 1: Complete Pipeline with Enhancements

```elixir
alias Crucible.IR.{Experiment, DatasetRef, BackendRef, StageDef, ReliabilityConfig}

experiment = %Experiment{
  id: "enhanced_pipeline_demo",
  description: "Demonstration of v0.4.0 enhancements",

  dataset: %DatasetRef{
    provider: :crucible_datasets,
    name: "gsm8k",
    split: :test,
    options: %{limit: 100}
  },

  pipeline: [
    # Validate experiment before execution
    %StageDef{name: :validate},

    # Load data
    %StageDef{name: :data_load},

    # Backend call with retry on transient errors
    %StageDef{
      name: :backend_call,
      options: %{mode: :sample, prompts: ["test"]},
      error_strategy: {
        :retry,
        max_attempts: 3,
        backoff: :exponential,
        retryable: [:timeout, :rate_limited]
      }
    },

    # Optional analysis - continue on error
    %StageDef{
      name: :analysis_metrics,
      error_strategy: {:continue, default: %{analysis: :unavailable}}
    },

    # Statistical benchmarking
    %StageDef{name: :bench},

    # Generate report
    %StageDef{name: :report}
  ],

  backend: %BackendRef{id: :tinkex, profile: :inference},
  reliability: %ReliabilityConfig{},

  metadata: %{
    # Enable middleware
    middleware: [
      Crucible.Pipeline.Middleware.Timing,
      Crucible.Pipeline.Middleware.Logging,
      Crucible.Pipeline.Middleware.Checkpoint,
      {Crucible.Pipeline.Middleware.Telemetry, detailed: true}
    ],

    # Enable profiling
    profiling: true
  }
}

# Run with enhanced features
{:ok, ctx} = CrucibleFramework.run(experiment)

# Access results with helpers
backend_metrics = Crucible.Context.get_metric(ctx, :backend)
timings = Crucible.Context.get_metric(ctx, :stage_timings)
completed = Crucible.Context.completed_stages(ctx)

# Export timeline
Crucible.Pipeline.Profiler.export_timeline(ctx, "timeline.html")

# Analyze bottlenecks
bottlenecks = Crucible.Pipeline.Analyzer.find_bottlenecks(ctx)
IO.inspect(bottlenecks, label: "Performance Bottlenecks")
```

### Example 2: Custom Middleware

```elixir
defmodule MyApp.Middleware.CustomMetrics do
  @behaviour Crucible.Pipeline.Middleware

  @impl true
  def before_stage(ctx, stage_def) do
    # Initialize custom metrics
    Crucible.Context.put_metric(
      ctx,
      :"#{stage_def.name}_custom",
      %{start_memory: :erlang.memory(:total)}
    )
  end

  @impl true
  def after_stage(ctx, stage_def, {:ok, new_ctx}) do
    # Calculate memory delta
    metrics = Crucible.Context.get_metric(new_ctx, :"#{stage_def.name}_custom")
    end_memory = :erlang.memory(:total)
    delta = end_memory - metrics.start_memory

    updated_metrics = Map.put(metrics, :memory_delta_bytes, delta)

    new_ctx = Crucible.Context.put_metric(
      new_ctx,
      :"#{stage_def.name}_custom",
      updated_metrics
    )

    {:ok, new_ctx}
  end

  @impl true
  def on_error(ctx, _stage_def, error) do
    # Log error and continue
    Logger.error("Stage error: #{inspect(error)}")
    {:continue, ctx}
  end
end
```

---

## 9. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Middleware performance overhead | Medium | Low | Benchmark each middleware, make all optional |
| Context helper bugs | Low | Medium | Comprehensive tests, property-based testing |
| Backward compatibility break | Low | High | Extensive integration tests, careful review |
| Validation false positives | Medium | Low | Configurable validation levels, override option |

### Mitigation Strategies

1. **Performance:** Benchmark before/after, set performance budgets
2. **Quality:** Require 95%+ test coverage, code review for all changes
3. **Compatibility:** Test against existing experiments, gradual rollout
4. **Documentation:** Update docs alongside code, add migration guide

---

## 10. Conclusion

This enhancement proposal adds significant value to CrucibleFramework while maintaining its core principles:

**Key Benefits:**
1. **Developer Experience:** 40-60% reduction in boilerplate
2. **Reliability:** 60-80% reduction in configuration errors
3. **Debuggability:** Complete observability into pipeline execution
4. **Extensibility:** Middleware enables custom functionality

**Effort:** ~14 days implementation + testing

**Version:** 0.3.0 → 0.4.0 (minor version bump - new features, no breaking changes)

**Recommendation:** Approve and proceed with phased implementation starting with Context helpers and validation (highest ROI, lowest risk).

---

## Appendix A: API Reference

See inline documentation in:
- `lib/crucible/context.ex` - Context helpers
- `lib/crucible/stage/validate.ex` - Validation stage
- `lib/crucible/pipeline/middleware.ex` - Middleware behaviour
- `lib/crucible/pipeline/error_handler.ex` - Error handling
- `lib/crucible/pipeline/profiler.ex` - Profiling tools

## Appendix B: Test Plan

See `test/crucible/enhancements/` for comprehensive test suite covering:
- 50+ Context helper tests
- 30+ Validation tests
- 40+ Middleware tests
- 35+ Error handling tests
- 25+ Profiling tests

**Total:** 180+ new tests

---

**Document Status:** ✅ Approved for Implementation
**Next Review:** After Phase 3 completion
**Questions/Feedback:** Open GitHub issue with tag `enhancement-0.4.0`
