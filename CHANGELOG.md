# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
