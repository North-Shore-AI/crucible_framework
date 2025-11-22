# North-Shore-AI Libraries Integration Analysis for crucible_framework

**Date:** November 21, 2025
**Purpose:** Deep investigation of ExDataCheck, ExFairness, coalas-lab, and tinkerer for integration into crucible_framework
**Author:** Research Agent

---

## Executive Summary

This document provides a comprehensive analysis of four North-Shore-AI repositories for integration into the crucible_framework ecosystem. The analysis reveals strong opportunities for enhancing the framework's data validation, fairness assessment, and research capabilities.

### Key Findings

| Repository | Type | Maturity | Integration Priority |
|------------|------|----------|---------------------|
| **ExDataCheck** | Elixir Library | Production-Ready (v0.2.0) | **HIGH** - Direct dependency candidate |
| **ExFairness** | Elixir Library | Production-Ready (v0.2.0) | **HIGH** - Direct dependency candidate |
| **coalas-lab** | Research Docs | Reference | **MEDIUM** - Research guidance |
| **tinkerer** | Development Workspace | Active Development | **LOW** - Future integration pathways |

---

## 1. ExDataCheck - Data Validation & Quality Library

### Overview

ExDataCheck is a production-ready data validation library bringing Great Expectations-style validation to the Elixir ecosystem. It provides 22 built-in expectations, advanced profiling, drift detection, and comprehensive statistical analysis specifically designed for ML workflows.

**Version:** 0.2.0
**License:** MIT
**Tests:** 273 passing (4 doctests, 25 property-based, 244 unit)
**Coverage:** >90%

### Key APIs and Modules

#### Core Module: `ExDataCheck`

```elixir
# Main API Functions
ExDataCheck.validate(dataset, expectations)       # Validate dataset against expectations
ExDataCheck.validate!(dataset, expectations)      # Validate and raise on failure
ExDataCheck.profile(dataset, opts \\ [])          # Generate dataset profile
ExDataCheck.create_baseline(dataset)              # Create drift detection baseline
ExDataCheck.detect_drift(dataset, baseline, opts) # Detect distribution drift
```

#### Expectation Modules

**Schema Expectations (`ExDataCheck.Expectations.Schema`)**
- `expect_column_to_exist/1` - Column must exist
- `expect_column_to_be_of_type/2` - Type validation (integer, float, string, boolean, etc.)
- `expect_column_count_to_equal/1` - Dataset column count

**Value Expectations (`ExDataCheck.Expectations.Value`)**
- `expect_column_values_to_be_between/3` - Range validation
- `expect_column_values_to_be_in_set/2` - Allowed values
- `expect_column_values_to_match_regex/2` - Pattern matching
- `expect_column_values_to_not_be_null/1` - Null checking
- `expect_column_values_to_be_unique/1` - Uniqueness
- `expect_column_values_to_be_increasing/1` - Monotonicity
- `expect_column_values_to_be_decreasing/1` - Monotonicity
- `expect_column_value_lengths_to_be_between/3` - Length validation

**Statistical Expectations (`ExDataCheck.Expectations.Statistical`)**
- `expect_column_mean_to_be_between/3` - Mean bounds
- `expect_column_median_to_be_between/3` - Median bounds
- `expect_column_stdev_to_be_between/3` - Standard deviation bounds
- `expect_column_quantile_to_be/3` - Quantile validation
- `expect_column_values_to_be_normal/1` - Normality testing

**ML-Specific Expectations (`ExDataCheck.Expectations.ML`)**
- `expect_label_balance/2` - Classification label balance
- `expect_label_cardinality/2` - Label count bounds
- `expect_feature_correlation/3` - Correlation checking
- `expect_no_missing_values/1` - Completeness
- `expect_table_row_count_to_be_between/2` - Dataset size
- `expect_no_data_drift/3` - Drift detection

#### Supporting Modules

- `ExDataCheck.Profile` - Profiling results with JSON/Markdown export
- `ExDataCheck.ValidationResult` - Aggregated validation results
- `ExDataCheck.Drift` - Drift detection (KS test, PSI)
- `ExDataCheck.Statistics` - Statistical utilities
- `ExDataCheck.Correlation` - Pearson/Spearman correlations
- `ExDataCheck.Outliers` - Outlier detection (IQR, Z-score)

### Dependencies

```elixir
# Runtime
{:jason, "~> 1.4"}

# Dev/Test
{:ex_doc, "~> 0.31", only: :dev}
{:stream_data, "~> 1.1", only: :test}
```

### Architecture Insights

The library uses a clean functional architecture:
1. **Expectation DSL** - Declarative expectations with validator functions
2. **Validator Pattern** - Each expectation returns a validator closure
3. **Result Aggregation** - ValidationResult collects all expectation results
4. **Column Extraction** - Universal column extraction from maps/keyword lists

### Integration Opportunities with crucible_framework

#### High Priority

1. **Dataset Validation Layer** - Add ExDataCheck to `crucible_datasets` for benchmark validation
   ```elixir
   # In crucible_datasets/lib/dataset.ex
   def validate_benchmark(dataset) do
     expectations = [
       ExDataCheck.expect_column_to_exist(:input),
       ExDataCheck.expect_column_to_exist(:expected_output),
       ExDataCheck.expect_no_missing_values(:input)
     ]
     ExDataCheck.validate!(dataset, expectations)
   end
   ```

2. **Experiment Data Quality** - Validate experiment data before statistical analysis in `crucible_bench`
   ```elixir
   # Pre-validation before running statistical tests
   def validate_experiment_data(results) do
     ExDataCheck.validate!(results, [
       ExDataCheck.expect_column_values_to_be_between(:accuracy, 0.0, 1.0),
       ExDataCheck.expect_no_missing_values(:model_id),
       ExDataCheck.expect_column_values_to_be_normal(:latency_ms)
     ])
   end
   ```

3. **Drift Detection for Model Monitoring** - Integrate with `crucible_telemetry`
   ```elixir
   # Detect drift in production data
   baseline = ExDataCheck.create_baseline(training_metrics)
   drift_result = ExDataCheck.detect_drift(production_metrics, baseline)
   ```

4. **Data Profiling for Research** - Add profiling to experiment reports
   ```elixir
   profile = ExDataCheck.profile(experiment_data, detailed: true)
   # Include correlation_matrix and outliers in experiment report
   ```

#### Medium Priority

5. **Feature Validation for Adversarial Testing** - Validate adversarial perturbation bounds
6. **Training Data Quality Gates** - Integrate with potential Tinkex training pipelines
7. **Benchmark Caching Validation** - Ensure cached datasets maintain integrity

---

## 2. ExFairness - Fairness and Bias Detection Library

### Overview

ExFairness is a comprehensive fairness and bias detection library for Elixir AI/ML systems. It provides group fairness metrics, bias detection algorithms, and mitigation techniques with GPU acceleration via Nx.

**Version:** 0.2.0
**License:** MIT
**Tests:** 134 (102 unit + 32 doctests)
**Coverage:** Zero warnings, full type specs

### Key APIs and Modules

#### Core Module: `ExFairness`

```elixir
# Fairness Metrics
ExFairness.demographic_parity(predictions, sensitive_attr, opts)
ExFairness.equalized_odds(predictions, labels, sensitive_attr, opts)
ExFairness.equal_opportunity(predictions, labels, sensitive_attr, opts)
ExFairness.predictive_parity(predictions, labels, sensitive_attr, opts)

# Comprehensive Reporting
ExFairness.fairness_report(predictions, labels, sensitive_attr, opts)
```

#### Metrics Modules

**Demographic Parity (`ExFairness.Metrics.DemographicParity`)**
- `compute/3` - P(Y=1|A=0) = P(Y=1|A=1)
- Returns: group rates, disparity, pass/fail, interpretation

**Equalized Odds (`ExFairness.Metrics.EqualizedOdds`)**
- `compute/4` - Equal TPR and FPR across groups
- Returns: TPR/FPR per group, disparities, pass/fail

**Equal Opportunity (`ExFairness.Metrics.EqualOpportunity`)**
- `compute/4` - Equal TPR across groups
- Returns: TPR per group, disparity, pass/fail

**Predictive Parity (`ExFairness.Metrics.PredictiveParity`)**
- `compute/4` - Equal PPV across groups
- Returns: PPV per group, disparity, pass/fail

#### Detection Module

**Disparate Impact (`ExFairness.Detection.DisparateImpact`)**
- `detect/2` - EEOC 80% rule compliance
- Returns: selection rates, ratio, legal compliance status

#### Mitigation Module

**Reweighting (`ExFairness.Mitigation.Reweighting`)**
- `compute_weights/3` - Sample weights for demographic parity or equalized odds
- Returns: Nx tensor of normalized weights (mean = 1.0)

#### Reporting Module

**Report (`ExFairness.Report`)**
- `generate/4` - Multi-metric fairness assessment
- `to_markdown/1` - Human-readable export
- `to_json/1` - Machine-readable export

#### Utility Modules

- `ExFairness.Utils` - Core tensor utilities
- `ExFairness.Utils.Metrics` - Confusion matrix, TPR, FPR, PPV calculations
- `ExFairness.Validation` - Input validation with helpful error messages
- `ExFairness.Error` - Custom error handling

### Dependencies

```elixir
# Runtime
{:nx, "~> 0.7"}

# Dev/Test
{:ex_doc, "~> 0.31", only: :dev}
{:dialyxir, "~> 1.4", only: [:dev, :test]}
{:excoveralls, "~> 0.18", only: :test}
{:credo, "~> 1.7", only: [:dev, :test]}
{:stream_data, "~> 1.0", only: :test}
```

### Architecture Insights

1. **Nx-Powered Computation** - All metrics use `Nx.Defn` for GPU acceleration
2. **Comprehensive Validation** - Input validation with detailed error messages
3. **Interpretation Layer** - Each metric returns human-readable interpretations
4. **Research Foundation** - Based on seminal papers (Dwork, Hardt, Chouldechova, Kleinberg)

### Integration Opportunities with crucible_framework

#### High Priority

1. **Ensemble Fairness Testing** - Add fairness validation to `crucible_ensemble`
   ```elixir
   # After ensemble voting, check fairness
   def validate_ensemble_fairness(predictions, labels, sensitive_attrs) do
     report = ExFairness.fairness_report(predictions, labels, sensitive_attrs)

     if report.failed_count > 0 do
       Logger.warning("Ensemble fairness issues: #{report.overall_assessment}")
     end

     report
   end
   ```

2. **Legal Compliance Layer** - Add to `crucible_harness` experiment reports
   ```elixir
   # In experiment analysis
   di_result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attrs)
   if !di_result.passes_80_percent_rule do
     add_warning("EEOC 80% rule violation detected")
   end
   ```

3. **Fairness Metrics in crucible_bench** - Add fairness as first-class statistical metric
   ```elixir
   # Compare fairness across models
   CrucibleBench.compare_groups(%{
     hypothesis: "Model B improves fairness over Model A",
     groups: [
       %{name: "model_a", scores: model_a_fairness_scores},
       %{name: "model_b", scores: model_b_fairness_scores}
     ]
   })
   ```

4. **Bias Mitigation Integration** - Add to training pipelines
   ```elixir
   # Pre-training mitigation
   weights = ExFairness.Mitigation.Reweighting.compute_weights(
     labels, sensitive_attrs, target: :demographic_parity
   )
   ```

5. **Fairness Monitoring in crucible_telemetry** - Real-time fairness tracking
   ```elixir
   # Emit fairness telemetry events
   :telemetry.execute(
     [:crucible, :fairness, :check],
     %{passed: report.passed_count, failed: report.failed_count},
     %{experiment_id: experiment_id}
   )
   ```

#### Medium Priority

6. **Adversarial Fairness Testing** - Test robustness of fairness under adversarial conditions
7. **Cross-Group Reliability** - Ensure ensemble reliability is equitable
8. **Fairness in XAI** - Integrate with `crucible_xai` for fairness explanations

### Impossibility Theorem Considerations

ExFairness documents the impossibility results (Chouldechova 2017, Kleinberg 2016): when base rates differ, calibration, equal TPR, and equal TNR cannot all be satisfied. Integration should:
- Surface trade-offs clearly in reports
- Allow users to prioritize which fairness property matters
- Document when metric conflicts are mathematically inevitable

---

## 3. coalas-lab - Research Documentation Repository

### Overview

coalas-lab is a research documentation repository containing due diligence analyses and integration studies. It provides strategic guidance for the crucible ecosystem's development.

**Type:** Documentation/Research
**Files:** 4 markdown documents

### Key Documents

#### 1. Crucible NLP Research Integration Analysis (77KB)

**Purpose:** Comprehensive analysis of crucible ecosystem for NLP research applications (UCLA CoalasLab)

**Key Sections:**
- Research Context: Long-form language generation evaluation, Visual-grounded accessibility
- Infrastructure Analysis: All 15+ repositories mapped to research needs
- Statistical Testing: 15+ hypothesis tests with automatic selection
- Experiment Orchestration: Declarative DSL for reproducible experiments
- Benchmark Management: MMLU, HumanEval, GSM8K integration

**Integration Insights:**
- Identifies gaps in vision-language benchmarks
- Proposes accessibility-focused dataset extensions
- Maps crucible capabilities to specific research problems

#### 2. NLP Research Infrastructure Analysis (35KB)

**Purpose:** Technical analysis of NLP infrastructure capabilities

**Key Content:**
- Performance benchmarks for different BEAM configurations
- Memory management strategies for large datasets
- Distributed execution patterns for multi-node experiments

#### 3. Publications Due Diligence

**Purpose:** Track publication readiness and citation requirements

#### 4. Research Due Diligence

**Purpose:** Evaluate research quality and rigor standards

### Integration Opportunities

1. **Research Methodology Alignment** - Use coalas-lab analyses to guide crucible_framework documentation
2. **Gap Analysis Reference** - Identify missing capabilities (vision-language benchmarks)
3. **Publication Preparation** - Ensure crucible experiments meet publication standards
4. **Accessibility Research** - Consider accessibility as a fairness dimension

---

## 4. tinkerer - Development Workspace

### Overview

tinkerer is the primary active development workspace for the North-Shore-AI organization. It contains the CNS 3.0 (Computational Narrative Synthesis) project and related experimental work.

**Type:** Development/Project
**Status:** Most Active Project

### Directory Structure

```
tinkerer/
├── .claude/           # Claude Code configuration
├── brainstorm/        # Ideation and planning (11 subdirectories)
├── brainstorms/       # Additional brainstorming
├── cns-support-models/ # CNS model support (7 subdirectories)
├── cns2/              # CNS version 2
├── cns3/              # CNS version 3 materials
├── dashboard/         # Visualization dashboard
├── dashboard_data/    # Dashboard data files
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── thinker/           # Thinker evaluation system
├── thinking-machines-labs/ # Lab experiments
├── tinker-docs/       # Tinker documentation
├── runs/              # Experiment runs
└── artifacts/         # Generated artifacts
```

### Key Components from CLAUDE.md

#### CNS 3.0 Architecture

The CNS 3.0 system uses an **adversarial collaboration** architecture with three core agents:

1. **Proposer (Thesis)** - Extracts claims from documents, emits Structured Narrative Objects (SNOs)
2. **Antagonist (Antithesis)** - Stress-tests SNOs, surfaces contradictions, quantifies topological holes
3. **Synthesizer (Synthesis)** - Resolves high-chirality SNO pairs, generates candidate syntheses

**Evaluation Philosophy:**
- Semantic-first metrics (cosine similarity, entailment scores)
- No exact-match testing (incompatible with CNS mandate)
- Beta-1 reduction as quality metric

#### Thinker System

The Thinker system provides:
- Data setup and validation (`thinker.cli data setup`)
- LoRA training orchestration (HF/PEFT or Tinker backend)
- Evaluation with semantic validation pipeline
- JSONL artifact generation under `runs/`

#### Critic Ensemble (Planned)

| Critic | Function | Status |
|--------|----------|--------|
| **Grounding** | DeBERTa-v3 entailment | Specified |
| **Logic** | Graph Attention Network | Designed |
| **Novelty/Parsimony** | Embedding novelty | Planned |
| **Bias/Causal** | Correlation vs causation | Future |

### Integration Opportunities with crucible_framework

#### Medium Priority

1. **Critic Integration** - Align tinkerer critics with crucible_bench statistical testing
   ```elixir
   # Map critic scores to crucible_bench effect sizes
   grounding_score = thinker_evaluate(sno)
   effect_size = CrucibleBench.cohens_d(grounding_score, baseline_score)
   ```

2. **SNO Validation** - Use ExDataCheck for SNO schema validation
   ```elixir
   # Validate SNO structure
   ExDataCheck.validate!(sno, [
     ExDataCheck.expect_column_to_exist(:hypothesis),
     ExDataCheck.expect_column_to_exist(:claims),
     ExDataCheck.expect_column_to_exist(:evidence)
   ])
   ```

3. **Fairness in Synthesis** - Use ExFairness to ensure synthesized claims don't introduce bias

4. **Telemetry Integration** - Connect tinkerer metrics to crucible_telemetry
   ```elixir
   # Emit CNS evaluation events
   :telemetry.execute(
     [:tinkerer, :sno, :evaluated],
     %{grounding: score, beta1: beta1},
     %{run_id: run_id}
   )
   ```

#### Lower Priority (Future Work)

5. **Antagonist as Adversarial Testing** - Map to crucible_adversary patterns
6. **Research Harness Integration** - Use crucible_harness DSL for CNS experiments
7. **Publication Pipeline** - Generate LaTeX reports from tinkerer experiments

---

## 5. Dependency Graph and Architecture Recommendations

### Proposed Integration Architecture

```
crucible_framework (v0.2.1)
├── tinkex (existing dependency)
├── ex_data_check (NEW - recommended)
│   └── jason
└── ex_fairness (NEW - recommended)
    └── nx
```

### Mix.exs Changes for crucible_framework

```elixir
defp deps do
  [
    # Core dependency
    {:tinkex, "~> 0.1.1"},

    # NEW: Data validation and quality
    {:ex_data_check, "~> 0.2.0"},

    # NEW: Fairness and bias detection
    {:ex_fairness, "~> 0.2.0"},

    # Testing...
  ]
end
```

### Integration Modules to Create

1. **`Crucible.DataQuality`** - Wrapper for ExDataCheck in crucible context
2. **`Crucible.Fairness`** - Wrapper for ExFairness in crucible context
3. **`Crucible.Experiment.Validators`** - Pre/post validation for experiments
4. **`Crucible.Report.FairnessSection`** - Fairness reporting in experiment outputs

---

## 6. Implementation Recommendations

### Phase 1: Direct Integration (Week 1-2)

1. Add ExDataCheck and ExFairness as dependencies
2. Create wrapper modules with crucible-specific defaults
3. Add data validation to `crucible_datasets` benchmark loading
4. Add fairness metrics to experiment reports

### Phase 2: Deep Integration (Week 3-4)

1. Integrate drift detection with `crucible_telemetry`
2. Add fairness as first-class metric in `crucible_bench`
3. Create fairness validation expectation for `crucible_harness` DSL
4. Add data quality gates to experiment workflows

### Phase 3: Advanced Features (Week 5-6)

1. GPU-accelerated fairness analysis for large experiments
2. Streaming data validation for continuous monitoring
3. Intersectional fairness analysis (future ExFairness feature)
4. Custom expectations for LLM-specific validation

---

## 7. Risks and Considerations

### Dependency Risks

1. **Nx Version Compatibility** - ExFairness uses Nx ~> 0.7, ensure compatibility with other Nx users
2. **Jason vs Jason.* Conflicts** - Both libraries use Jason, should be fine

### Architecture Risks

1. **Performance Overhead** - Data validation adds latency; make optional in hot paths
2. **Metric Proliferation** - Many fairness metrics; need clear guidance on selection
3. **GPU Memory** - ExFairness GPU acceleration may compete for GPU memory

### Research Risks

1. **Impossibility Results** - Users may expect all fairness metrics to pass simultaneously
2. **Overfitting to Fairness** - Excessive fairness constraints may reduce model utility
3. **Binary Limitations** - Current ExFairness only supports binary sensitive attributes

---

## 8. Conclusion

ExDataCheck and ExFairness are mature, production-ready libraries that provide essential capabilities missing from the current crucible_framework ecosystem:

- **ExDataCheck** brings Great Expectations-style data validation to Elixir ML pipelines, enabling rigorous data quality gates throughout the experiment lifecycle.

- **ExFairness** brings comprehensive fairness metrics and bias detection with GPU acceleration, essential for responsible AI research.

Both libraries share the North-Shore-AI quality standards (>90% coverage, zero warnings, comprehensive docs) and use compatible technologies (Elixir 1.14+, OTP 25+).

**Recommendation:** Proceed with Phase 1 integration immediately. The cost is minimal (two dependencies) and the benefit is substantial (data quality + fairness throughout the research pipeline).

The coalas-lab and tinkerer repositories provide valuable research context and future integration pathways, but are lower priority for immediate inclusion.

---

## Appendix A: API Quick Reference

### ExDataCheck

```elixir
# Validation
ExDataCheck.validate(dataset, expectations) -> ValidationResult
ExDataCheck.validate!(dataset, expectations) -> ValidationResult | raise

# Profiling
ExDataCheck.profile(dataset) -> Profile
ExDataCheck.profile(dataset, detailed: true) -> Profile with outliers/correlations

# Drift Detection
ExDataCheck.create_baseline(dataset) -> baseline
ExDataCheck.detect_drift(dataset, baseline) -> DriftResult

# Expectations (22 total)
ExDataCheck.expect_column_to_exist(column)
ExDataCheck.expect_column_values_to_be_between(column, min, max)
ExDataCheck.expect_column_mean_to_be_between(column, min, max)
ExDataCheck.expect_no_data_drift(column, baseline)
# ... and 18 more
```

### ExFairness

```elixir
# Metrics
ExFairness.demographic_parity(predictions, sensitive_attr, opts)
ExFairness.equalized_odds(predictions, labels, sensitive_attr, opts)
ExFairness.equal_opportunity(predictions, labels, sensitive_attr, opts)
ExFairness.predictive_parity(predictions, labels, sensitive_attr, opts)

# Reporting
ExFairness.fairness_report(predictions, labels, sensitive_attr, opts)
ExFairness.Report.to_markdown(report)
ExFairness.Report.to_json(report)

# Detection
ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr)

# Mitigation
ExFairness.Mitigation.Reweighting.compute_weights(labels, sensitive_attr, opts)
```

---

## Appendix B: Example Integration Code

### Experiment with Data Quality and Fairness

```elixir
defmodule MyExperiment do
  use CrucibleHarness.Experiment
  import ExDataCheck

  experiment "fair_llm_evaluation" do
    models ["gpt-4", "claude-3", "llama-3.1"]

    # Data quality validation
    pre_validate do
      validate!(dataset, [
        expect_column_to_exist(:prompt),
        expect_column_to_exist(:expected_response),
        expect_column_to_exist(:sensitive_attr),
        expect_no_missing_values(:prompt),
        expect_column_values_to_be_in_set(:sensitive_attr, [0, 1])
      ])
    end

    # Run experiment
    metrics [:accuracy, :latency_p99, :cost]

    # Post-experiment analysis
    analyze do
      # Statistical testing
      compare_models with: :mann_whitney
      compute_effect_sizes with: :cohens_d

      # Fairness analysis
      fairness_report = ExFairness.fairness_report(
        predictions, labels, sensitive_attrs
      )

      if fairness_report.failed_count > 0 do
        add_warning(fairness_report.overall_assessment)
      end

      include_in_report(:fairness, fairness_report)
    end
  end
end
```

---

*End of Research Document*
