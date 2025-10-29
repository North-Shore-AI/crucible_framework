# CrucibleFramework Examples

This directory contains working examples that demonstrate the core functionality of the CrucibleFramework.

## Running Examples

All examples can be run directly with `mix run`:

```bash
# Navigate to the crucible_framework directory
cd /path/to/crucible_framework

# Run any example
mix run examples/01_basic_usage.exs
mix run examples/02_statistics.exs
mix run examples/03_experiment_definition.exs
mix run examples/04_statistical_analysis.exs
```

## Example Descriptions

### 01_basic_usage.exs

**Purpose:** Introduction to basic framework features

**Demonstrates:**
- Getting framework version and system information
- Listing available components
- Checking component status
- Understanding the component architecture

**Use Case:** New users getting familiar with the framework

**Run Time:** < 1 second

---

### 02_statistics.exs

**Purpose:** Statistical analysis of experimental data

**Demonstrates:**
- Calculating descriptive statistics (mean, median, std dev)
- Computing percentiles (P50, P95, P99)
- Comparing baseline vs treatment conditions
- Analyzing both latency and accuracy metrics

**Use Case:** Researchers analyzing experiment results

**Run Time:** < 1 second

**Key Concepts:**
- Summary statistics for continuous metrics
- Comparison of experimental conditions
- Effect size calculation (percentage change)

---

### 03_experiment_definition.exs

**Purpose:** Defining and configuring experiments

**Demonstrates:**
- Creating experiments with `Experiment.new/1`
- Validating experiment configurations
- Using the `Experiment` behaviour in custom modules
- Defining multiple experiment types

**Use Case:** Setting up research experiments

**Run Time:** < 1 second

**Key Concepts:**
- Experiment configuration structure
- Required vs optional fields
- Experiment validation
- Best practices for experiment definition

---

### 04_statistical_analysis.exs

**Purpose:** Comprehensive statistical analysis workflow

**Demonstrates:**
- Simulating experimental data
- Computing complete statistical summaries
- Effect size calculation (Cohen's d)
- Cost-benefit analysis
- Research conclusions and recommendations

**Use Case:** Complete research workflow from data to conclusions

**Run Time:** < 1 second

**Key Concepts:**
- Accuracy vs latency trade-offs
- Cost-effectiveness analysis
- Effect size interpretation
- Research decision-making framework

**Statistical Concepts:**
- Cohen's d effect sizes (small: 0.2, medium: 0.5, large: 0.8)
- 95% confidence intervals
- Percentile analysis
- Cost per percentage point accuracy gain

---

## Example Output Structure

Each example follows this pattern:

1. **Header** - Clear title and description
2. **Setup** - Data preparation or configuration
3. **Execution** - Running the analysis or operation
4. **Results** - Formatted output with key metrics
5. **Interpretation** - What the results mean

## Learning Path

Recommended order for learning:

1. **Start here:** `01_basic_usage.exs` - Understand the framework structure
2. **Next:** `02_statistics.exs` - Learn statistical calculations
3. **Then:** `03_experiment_definition.exs` - Set up experiments
4. **Finally:** `04_statistical_analysis.exs` - Complete research workflow

## Advanced Usage

For production research, these examples demonstrate the basic building blocks. The actual component libraries provide:

- **Ensemble** - Multi-model voting with 4 strategies
- **Hedging** - Tail latency reduction
- **Bench** - 15+ statistical tests with p-values
- **TelemetryResearch** - Complete event instrumentation
- **DatasetManager** - Standard benchmark datasets
- **CausalTrace** - LLM decision provenance
- **ResearchHarness** - Automated experiment orchestration
- **Reporter** - Multi-format report generation

See the main [README.md](../README.md) and component documentation for details.

## Troubleshooting

### Example won't run

Ensure you're in the correct directory:
```bash
cd /path/to/crucible_framework
mix run examples/example_name.exs
```

### Compilation errors

Clean and recompile:
```bash
mix clean
mix compile
mix run examples/example_name.exs
```

### Missing dependencies

Install dependencies:
```bash
mix deps.get
```

## Contributing Examples

To contribute a new example:

1. Follow the naming convention: `NN_descriptive_name.exs`
2. Include a clear header with purpose and description
3. Add comprehensive comments explaining key concepts
4. Format output clearly with visual separators
5. Add the example to this README
6. Test thoroughly with `mix run`

## Questions?

- **Documentation:** See main [README.md](../README.md)
- **Issues:** https://github.com/North-Shore-AI/crucible_framework/issues
- **Discussions:** https://github.com/North-Shore-AI/crucible_framework/discussions
