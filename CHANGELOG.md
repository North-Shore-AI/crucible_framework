# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
