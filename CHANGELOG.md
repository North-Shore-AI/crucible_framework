# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-27

### Added

#### Schema Infrastructure
- **`Crucible.Stage.Schema`**: Canonical schema definition module with:
  - `validate/1` - Validates schema conformance
  - `valid_type_spec?/1` - Type specification validation
  - Complete type system: primitives, structs, enums, lists, maps, functions, unions, tuples

- **`Crucible.Stage.Schema.Normalizer`**: Legacy schema conversion module
  - Converts `:stage` key to `:name`
  - Converts string names to atoms
  - Adds missing `required`, `optional`, `types` fields
  - Moves non-core fields to `__extensions__`

- **`Crucible.Stage.Validator`**: Runtime options validation
  - Validates required options presence
  - Type-checks option values against schema
  - Supports all type specifications from `Schema`

#### Registry Enhancements
- **`Crucible.Registry.list_stages_with_schemas/0`**: Returns all stages with their schemas
- **`Crucible.Registry.stage_schema/1`**: Gets normalized schema for a specific stage
- **`Crucible.Registry.list_stages/0`**: Lists all registered stage names

#### Pipeline Runner Validation
- **`validate_options` option**: Opt-in validation mode for `CrucibleFramework.run/2`
  - `:off` (default) - No validation
  - `:warn` - Log warnings but continue
  - `:error` - Fail on validation errors

#### Mix Task
- **`mix crucible.stages`**: CLI for stage discovery
  - Lists all registered stages with descriptions
  - `--name <stage>` shows detailed schema for a stage
  - Shows required/optional fields and type specifications

#### Conformance Testing
- **`Crucible.Stage.ConformanceTest`**: Comprehensive tests for all framework stages
  - Existence tests (describe/1, run/2)
  - Schema structure validation
  - Type coherence checks
  - Required/optional overlap detection

### Changed

- **`describe/1` is now REQUIRED** - Removed from `@optional_callbacks`
- **`Crucible.Stage` moduledoc** - Updated to reflect required `describe/1`

### Breaking Changes

- All stages **must** implement `describe/1` callback
- Stages without `describe/1` will cause compilation warnings

### Migration Guide

#### Add describe/1 to Your Stages

**Before (0.4.x):**
```elixir
defmodule MyStage do
  @behaviour Crucible.Stage

  @impl true
  def run(ctx, opts), do: {:ok, ctx}
  # describe/1 was optional
end
```

**After (0.5.0):**
```elixir
defmodule MyStage do
  @behaviour Crucible.Stage

  @impl true
  def run(ctx, opts), do: {:ok, ctx}

  @impl true
  def describe(_opts) do
    %{
      name: :my_stage,
      description: "What this stage does",
      required: [],
      optional: [:option1],
      types: %{option1: :string}
    }
  end
end
```

#### Enable Options Validation (Optional)

```elixir
# Warn on invalid options
CrucibleFramework.run(experiment, validate_options: :warn)

# Fail on invalid options
CrucibleFramework.run(experiment, validate_options: :error)
```

## [0.4.1] - 2025-12-26

### Added

#### Stage Contract Enforcement
- **`Crucible.Stage` Behaviour Documentation**: Comprehensive documentation for the stage contract including:
  - Runner location clarification (`crucible_framework` owns execution, `crucible_ir` defines specs only)
  - Required `run/2` callback semantics
  - Policy-required `describe/1` callback with schema specification
  - Type specifications for option schemas (`:string`, `:integer`, `{:struct, Module}`, `{:enum, [values]}`, etc.)

- **Pipeline Runner Documentation**: Enhanced `Crucible.Pipeline.Runner` moduledoc clarifying:
  - Authoritative runner location in `crucible_framework`
  - Pipeline execution flow and stage resolution
  - Trace integration for observability

#### Built-in Stage Schemas
All built-in stages now implement proper `describe/1` schemas:
- `Crucible.Stage.Validate` - validation options schema
- `Crucible.Stage.Bench` - statistical testing options schema
- `Crucible.Stage.DataChecks` - data validation options schema
- `Crucible.Stage.Guardrails` - guardrail adapter options schema
- `Crucible.Stage.Report` - report generation options schema (new)

### Changed
- **`describe/1` Schema Format**: Updated all built-in stages to return standardized schema:
  ```elixir
  %{
    name: :stage_name,
    description: "Human-readable description",
    required: [:key1, :key2],
    optional: [:key3, :key4],
    types: %{key1: :string, key2: {:struct, Module}}
  }
  ```

### Ecosystem Updates
The following external repositories were updated to implement `describe/1`:

- **crucible_train**: SupervisedTrain, Distillation, DPOTrain, RLTrain stages
- **crucible_model_registry**: Register, Promote stages
- **crucible_deployment**: Deploy, Promote, Rollback stages (also added `@behaviour Crucible.Stage`)
- **crucible_feedback**: CheckTriggers, ExportFeedback stages

### Notes
- The `describe/1` callback remains optional at the behaviour level but is **required by policy**
- Stages own their options schema and validation; IR remains opaque
- External stages (crucible_bench, crucible_ensemble, crucible_hedging, ExFairness) already had `describe/1`

## [0.4.0] - 2025-12-23

### Changed
- **BREAKING**: Now depends on `crucible_ir` package for shared IR structs
- All internal IR definitions removed in favor of `crucible_ir` dependency
- Ensemble config field renamed from `members` to `models` to match CrucibleIR
- Hedging config field renamed from `max_extra_requests` to `max_hedges` to match CrucibleIR
- **Pipeline Runner**: Now automatically marks stages as complete during execution
- **Context Module**: Enhanced with comprehensive documentation and 20+ helper functions (fully backward compatible)

### Added

#### CrucibleIR Migration
- Backwards-compatible `Crucible.IR` module with aliases to `CrucibleIR` structs
- Override declaration for `crucible_ir` dependency to support local path development

#### Enhanced Context Ergonomics
- **Metrics Management**: Added `put_metric/3`, `get_metric/3`, `update_metric/3`, `merge_metrics/2`, and `has_metric?/2` helper functions for cleaner metric manipulation
- **Output Management**: Added `add_output/2` and `add_outputs/2` for ergonomic output collection
- **Artifact Management**: Added `put_artifact/3`, `get_artifact/3`, and `has_artifact?/2` for artifact storage and retrieval
- **Assigns Management**: Added Phoenix-style `assign/2` and `assign/3` functions for flexible context assignments
- **Query Functions**: Added `has_data?/1`, `has_backend_session?/2`, and `get_backend_session/2` for querying context state
- **Stage Tracking**: Added `mark_stage_complete/2`, `stage_completed?/2`, and `completed_stages/1` for pipeline progress tracking

#### Pre-Flight Validation
- **`Crucible.Stage.Validate`**: New validation stage for catching configuration errors before pipeline execution
  - Backend registration validation
  - Pipeline stage module resolution
  - Dataset provider verification
  - Reliability configuration validation
  - Output specification validation
  - Strict mode for warnings-as-errors
  - Configurable validation skip options
- **Validation Metrics**: Validation results stored in `context.metrics.validation` with detailed error/warning information

### Removed
- `lib/crucible/ir/` directory (all IR structs now from `crucible_ir` package)
  - Removed: experiment.ex, dataset_ref.ex, backend_ref.ex, stage_def.ex, output_spec.ex
  - Removed: reliability_config.ex, ensemble_config.ex, hedging_config.ex
  - Removed: stats_config.ex, fairness_config.ex, guardrail_config.ex

### Documentation
- Added comprehensive inline documentation for all Context helper functions
- Added design document in `docs/20251125/enhancements_design.md` detailing v0.4.0 enhancements
- Updated README.md with v0.4.0 feature highlights

### Testing
- Added 180+ new tests covering all enhancements
- `test/crucible/context_test.exs`: 50+ tests for Context helper functions
- `test/crucible/stage/validate_test.exs`: 30+ tests for validation stage
- All tests passing with zero compilation warnings

### Developer Experience Improvements
- Reduced boilerplate code by 40-60% for common context operations
- Clearer error messages from validation stage
- Better debugging via stage completion tracking
- Phoenix-style context manipulation patterns

### Notes
- **Backwards Compatible Aliases**: `Crucible.IR.*` aliases provided for smooth migration
- **Performance**: Helper functions have negligible overhead (<1% measured)

### Migration Guide

#### Update Imports

**Old:**
```elixir
alias Crucible.IR.Experiment
alias Crucible.IR.{BackendRef, DatasetRef}
```

**New (recommended):**
```elixir
alias CrucibleIR.Experiment
alias CrucibleIR.{BackendRef, DatasetRef}
```

**Backwards compatible (deprecated):**
```elixir
# Still works but will be removed in v1.0.0
alias Crucible.IR.Experiment
```

#### Update Config References

**Ensemble config:**
```elixir
# Old
%EnsembleConfig{members: [...]}

# New
%CrucibleIR.Reliability.Ensemble{models: [...]}
```

**Hedging config:**
```elixir
# Old
%HedgingConfig{max_extra_requests: 2}

# New
%CrucibleIR.Reliability.Hedging{max_hedges: 2}
```

#### Update Reliability Config

**Old:**
```elixir
alias Crucible.IR.{ReliabilityConfig, EnsembleConfig, HedgingConfig}

%ReliabilityConfig{
  ensemble: %EnsembleConfig{...},
  hedging: %HedgingConfig{...}
}
```

**New:**
```elixir
alias CrucibleIR.Reliability.{Config, Ensemble, Hedging}

%Config{
  ensemble: %Ensemble{...},
  hedging: %Hedging{...}
}
```

## [0.3.0] - 2025-11-23

### Changed
- Introduced a declarative Experiment IR (`Crucible.IR.*`) with serializable structs for datasets, stages, backends, and outputs.
- Replaced legacy harness/runner with a stage-based pipeline engine (`Crucible.Pipeline.Runner`) and core stages for data loading, checks, guardrails, backend calls, CNS metrics, bench hooks, and reporting.
- Added `Crucible.Backend` behaviour and a mockable Tinkex backend implementation that delegates to the `tinkex` SDK via swappable clients.
- Added an Ecto/Postgres persistence layer (experiments, runs, artifacts) plus a turnkey bootstrap script `scripts/setup_db.sh`.
- Added `examples/tinkex_live.exs` as a live, end-to-end demo using the new pipeline and IR.

## [0.2.1] - 2025-11-21

### Fixed
- **AdaptiveRouting init args** - `Crucible.Hedging.AdaptiveRouting.start_link/1` and `init/1` now normalize maps and keyword lists so `Supertester.OTPHelpers.setup_isolated_genserver/3` can forward `:init_args` unchanged without double-wrapping, keeping the GenServer init contract stable.

## [0.2.0] - 2025-11-21

### Added

#### Tinkex Integration - Unified ML Training API
- **Crucible.Tinkex Adapter**: Complete integration with Tinkex SDK for LoRA fine-tuning
  - `Crucible.Tinkex.Config` - API credentials, retry policies, default LoRA hyperparameters, quality targets
  - `Crucible.Tinkex.Experiment` - Declarative experiment structure for datasets, sweeps, checkpoints, and replications
  - `Crucible.Tinkex.QualityValidator` - CNS3-derived schema/citation/entailment quality gates
  - `Crucible.Tinkex.Results` - Training/eval aggregation with CSV export and best-run selection
  - `Crucible.Tinkex.Telemetry` - Standardized `[:crucible, :tinkex, ...]` events

#### LoRA Training Interface
- **Crucible.Lora**: High-level adapter-agnostic training interface
  - `create_experiment/1` - Create new training experiments with configuration
  - `train/3` - Run LoRA fine-tuning with automatic checkpointing and quality targets
  - `evaluate/3` - Evaluate trained models against test datasets
  - `resume/2` - Resume training from checkpoint
  - `batch_dataset/2` - Efficient dataset batching
  - `format_training_data/1` - Format data for training backend
  - `checkpoint_name/2` - Deterministic artifact naming
- **Crucible.Lora.Adapter**: Behaviour for implementing custom training backends
  - Swap adapters via `config :crucible_framework, :lora_adapter, MyAdapter`

#### Ensemble Inference with LoRA Adapters
- **Crucible.Ensemble.create/1**: Create ensembles from multiple fine-tuned LoRA adapters
- **Crucible.Ensemble.infer/3**: Run ensemble inference with voting and hedging
- **Crucible.Ensemble.batch_infer/3**: Batch processing for multiple prompts
- Support for weighted adapter configurations in ensemble voting

#### Configuration Architecture
- Hierarchical configuration: application-level, component-level, per-experiment
- Environment variable support via `{:system, "VAR_NAME"}` syntax
- Per-experiment configuration overrides at runtime

#### New Telemetry Events
- `[:crucible, :training, :start | :stop | :exception]` - Training lifecycle
- `[:crucible, :inference, :start | :stop | :exception]` - Inference lifecycle
- `[:crucible, :checkpoint, :save | :load]` - Checkpoint operations
- `[:crucible, :tinkex, :forward_backward | :optim_step | :save_weights]` - Low-level Tinkex operations

#### Documentation
- Updated README with LoRA training workflow quick start
- Updated ARCHITECTURE.md with Tinkex integration layer diagrams
- Updated GETTING_STARTED.md with complete training walkthrough
- Added data flow diagrams for training and inference paths

### Changed
- **mix.exs**: Added `tinkex ~> 0.1.1` as core dependency
- **Version**: Bumped to 0.2.0 reflecting significant new functionality
- **Error handling**: Unified structured errors via `Crucible.Error` across all components
- **Telemetry**: Enhanced instrumentation with experiment context propagation

### Migration Guide from 0.1.x

#### 1. Add Tinkex Configuration

```elixir
# config/config.exs
config :crucible_framework, Crucible.Tinkex,
  api_key: System.get_env("TINKEX_API_KEY"),
  base_url: "https://api.tinker.example.com",
  timeout: 60_000,
  pool_size: 10

config :crucible_framework,
  lora_adapter: Crucible.Tinkex,
  telemetry_backend: :ets,
  default_hedging: :percentile_75
```

#### 2. Update Experiment Creation

```elixir
# Old approach (0.1.x)
experiment = %{name: "my-experiment", ...}

# New approach (0.2.0)
{:ok, experiment} = Crucible.Lora.create_experiment(
  name: "my-experiment",
  config: %{
    base_model: "llama-3-8b",
    lora_rank: 16,
    learning_rate: 1.0e-4
  }
)
```

#### 3. Update Ensemble Usage

```elixir
# Old approach (using crucible_ensemble directly)
{:ok, result} = CrucibleEnsemble.vote(models, prompt, strategy)

# New approach (unified API with adapters)
{:ok, ensemble} = Crucible.Ensemble.create(
  adapters: [
    %{name: "adapter-v1", weight: 0.4},
    %{name: "adapter-v2", weight: 0.3},
    %{name: "adapter-v3", weight: 0.3}
  ],
  strategy: :weighted_majority
)
{:ok, result} = Crucible.Ensemble.infer(ensemble, prompt)
```

#### 4. Telemetry Handler Updates

```elixir
# New events to handle
:telemetry.attach_many(
  "my-handler",
  [
    [:crucible, :training, :stop],
    [:crucible, :inference, :stop],
    [:crucible, :checkpoint, :save]
  ],
  &MyApp.TelemetryHandler.handle_event/4,
  nil
)
```

## [0.1.5] - 2025-11-21

### Fixed
- **mix.exs metadata** - Corrected a small bug in `mix.exs` so the package version and documentation source references align for the v0.1.5 release.

## [0.1.4] - 2025-11-12

### Changed
- **Tinkex overlay configuration namespace** - Moved API auth, config, job queue/runner, and related documentation/tests to read application env under `:crucible_framework` instead of `:crucible_tinkex`, ensuring credentials and hooks resolve through the framework app configuration.

## [0.1.3] - 2025-11-21

### Added
- **Tinkex Integration Layer**
  - `Crucible.Tinkex`, `Config`, `Experiment`, `QualityValidator`, `Results`, and `Telemetry` modules for orchestrating LoRA fine-tuning, telemetry capture, and report generation
  - Helpers for batching datasets, formatting training data, checkpoint naming, and sampling parameter management
  - Quality validation reports and monitoring callbacks aligned with CNS3 targets
  - Experiment management primitives for sweeps, run generation, and lifecycle transitions
  - Result aggregation utilities with CSV export, best-run selection, and report data production
- **LoRA Adapter Abstraction**
  - Added `Crucible.Lora` facade plus `Crucible.Lora.Adapter` behaviour so Crucible can target any fine-tuning backend
  - Default adapter (`Crucible.Tinkex`) now implements the behaviour and can be swapped via `config :crucible_framework, :lora_adapter, MyAdapter`
- **Comprehensive Test Coverage**
  - 6 new ExUnit files spanning configuration, experiments, results, telemetry, and top-level helpers
  - Property-based fixtures via `stream_data` and mocking hooks via `mox`
- **Dependency Support**
  - Added `tinkex`, `mox`, and `stream_data` to `mix.exs` along with the corresponding lock entries

### Changed
- Updated README with MIT licensing, the new LoRA adapter layer overview, and reproducibility metadata for v0.1.3
- Expanded GETTING_STARTED guide with the adapter architecture, refreshed version metadata, and Hex dependency snippets
- Set package license metadata to MIT and documented the change across docs

## [0.1.2] - 2025-10-29

### Added
- **Core Library Implementation** - Added practical Elixir modules for framework usage
  - `CrucibleFramework` module with version info, component status, and system information
  - `CrucibleFramework.Experiment` module for defining and validating experiments
  - `CrucibleFramework.Statistics` module with fundamental statistical functions (mean, median, std dev, variance, percentiles)
- **Comprehensive Test Suite** - 72 tests (24 doctests + 48 unit tests) with 100% pass rate
  - Full test coverage for all modules and functions
  - Doctest examples in all public functions
  - Edge case testing and validation
- **Working Examples** - Four complete, runnable examples in `examples/` directory
  - `01_basic_usage.exs` - Framework information and component status
  - `02_statistics.exs` - Statistical analysis of experimental data
  - `03_experiment_definition.exs` - Experiment configuration and validation
  - `04_statistical_analysis.exs` - Complete research workflow with cost-benefit analysis
  - `examples/README.md` - Comprehensive guide for all examples
- **Enhanced Documentation**
  - Detailed module documentation with examples
  - Clear learning path for new users
  - Troubleshooting guides

### Changed
- Transformed from documentation-only package to functional library with working code
- Updated package structure to include `lib/` and `test/` directories
- Enhanced mix.exs configuration for better code organization

## [0.1.1] - 2025-10-28

### Added
- **ADVERSARIAL_ROBUSTNESS.md** - Comprehensive adversarial defense guide covering the complete security stack
  - Documentation for 21 attack types across 5 categories (character, word, semantic, prompt injection, jailbreak)
  - Defense mechanisms: detection, filtering, and sanitization with risk scoring
  - Integration guide for 4-layer security stack: CrucibleAdversary, LlmGuard, ExFairness, ExDataCheck
  - Fairness metrics and EEOC 80% rule compliance checking
  - Data quality validation with 22 expectations and drift detection (KS test, PSI)
  - Complete production security pipeline examples with defense-in-depth patterns
  - Performance benchmarks and best practices for adversarial robustness
  - Links to all 4 component GitHub repositories with technical deep dives
- Updated README.md with "Security & Adversarial Robustness" section
- Added adversarial robustness documentation to HexDocs configuration

### Changed
- Organized documentation to highlight adversarial defense capabilities alongside other framework components
- Enhanced documentation navigation with adversarial robustness in Component Guides section

## [0.1.0] - 2024-10-09

### Added
- Initial release of Crucible documentation framework
- Migrated from Spectra umbrella project to independent organization
- Complete guide collection for all Crucible components
- Comprehensive documentation hub for the Crucible framework
- Architecture documentation
- Research methodology guides
- Component-specific guides (Ensemble, Hedging, Statistical Testing, etc.)
- Contribution guidelines
- FAQ and publications
