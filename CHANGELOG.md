# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
