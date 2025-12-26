# Getting Started: Crucible Framework

**A Comprehensive Guide for Researchers New to the Framework**

Version: 0.2.0
Last Updated: 2025-11-21
Target Audience: PhD students, ML researchers, scientists requiring rigorous LLM experimentation
Estimated Reading Time: 45 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Your First Experiment: Step-by-Step Walkthrough](#your-first-experiment-step-by-step-walkthrough)
6. [Understanding Your Results](#understanding-your-results)
7. [Core Concepts](#core-concepts)
8. [LoRA Adapter Layer (Tinkex Default)](#lora-adapter-layer-tinkex-default)
9. [Common Patterns and Examples](#common-patterns-and-examples)
10. [Troubleshooting](#troubleshooting)
11. [Common Issues and Solutions](#common-issues-and-solutions)
12. [Configuration Reference](#configuration-reference)
13. [Next Steps](#next-steps)
14. [Additional Resources](#additional-resources)

---

## Introduction

### What is the Crucible Framework?

The Crucible Framework is a scientifically-rigorous infrastructure for conducting **reproducible, statistically-valid experiments** on large language model (LLM) reliability, performance, and cost optimization. Unlike ad-hoc testing scripts, this framework provides:

- **Reproducibility**: Every experiment can be exactly reproduced with deterministic seeding
- **Statistical Rigor**: 15+ statistical tests with automatic test selection and assumption checking
- **Scalability**: Leverage Elixir's BEAM VM to run thousands of concurrent API requests
- **Observability**: Complete instrumentation captures every event for analysis
- **Publication-Ready Output**: Generate reports in Markdown, LaTeX, HTML, and Jupyter formats

### Who Should Use This Framework?

This framework is designed for:

- **PhD Students**: Conducting dissertation research on LLM reliability
- **ML Researchers**: Publishing papers requiring rigorous experimental validation
- **Research Labs**: Running systematic studies of AI system behavior
- **Engineers**: Evaluating model performance with scientific rigor

### What You'll Learn

By the end of this guide, you will be able to:

1. Install and configure the framework on your system
2. Run your first ensemble reliability experiment
3. Interpret statistical results and reports
4. Design custom experiments for your research questions
5. Troubleshoot common issues
6. Scale experiments from 10 to 10,000 queries

### Time Commitment

- **Basic Setup**: 30 minutes
- **First Experiment**: 1 hour
- **Mastery**: 1-2 weeks of regular use

---

## Prerequisites

### Required Knowledge

**Minimal Requirements:**
- Basic command line usage (cd, ls, running commands)
- Understanding of scientific experiments (control vs treatment groups)
- Familiarity with API keys and environment variables

**Helpful But Not Required:**
- Elixir programming (framework provides high-level DSL)
- Functional programming concepts
- Statistical hypothesis testing
- Git version control

**Learning Resources:**
If you're new to Elixir, these 30-minute tutorials will give you enough background:
- [Elixir Official Getting Started](https://elixir-lang.org/getting-started/introduction.html) - Chapters 1-5
- [Pattern Matching in Elixir](https://elixir-lang.org/getting-started/pattern-matching.html)
- [Pipe Operator Tutorial](https://elixir-lang.org/getting-started/enumerables-and-streams.html)

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+)
- macOS (11.0 Big Sur or newer)
- Windows (via WSL2 - Windows Subsystem for Linux)

**Hardware:**
- **CPU**: 2+ cores (4+ recommended for parallel experiments)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 2GB free space (500MB for framework + 1.5GB for dataset cache)
- **Network**: Stable internet connection for API calls

**Software:**
- **Elixir**: 1.14 or higher
- **Erlang/OTP**: 25 or higher
- **Git**: 2.30+ (for cloning repository)
- **PostgreSQL**: 14+ (optional, for persistent telemetry storage)

### Required API Keys

You'll need API keys for at least one LLM provider:

**Option 1: Start with Free Tiers** (Recommended for Learning)
- **Google Gemini API**: Free tier includes 60 requests/minute
  - Sign up: https://makersuite.google.com/app/apikey
  - Cost: Free up to rate limits, then $0.075 per 1M tokens

**Option 2: Minimal Cost** (Best for Small Experiments)
- **OpenAI GPT-4o Mini**: Very low cost, high quality
  - Sign up: https://platform.openai.com/api-keys
  - Cost: $0.15 per 1M input tokens, $0.60 per 1M output tokens
  - Budget: $5-10 sufficient for 10,000+ queries

**Option 3: Full Ensemble** (For Production Research)
- **Google Gemini** (as above)
- **OpenAI** (GPT-4o Mini, GPT-4o): https://platform.openai.com/api-keys
- **Anthropic** (Claude Haiku, Sonnet, Opus): https://console.anthropic.com/

**Cost Estimation:**
- 100 queries with 3-model ensemble: ~$0.05-0.50
- 1,000 queries with 5-model ensemble: ~$5-50
- 10,000 queries with 5-model ensemble: ~$50-500

The framework provides cost estimation before running experiments.

---

## Installation

We provide three installation methods. Choose the one that fits your workflow.

### Method 1: From GitHub (Recommended)

Clone the canonical repository to get the latest code, docs, and examples.

**Step 1: Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/North-Shore-AI/crucible_framework.git
cd crucible_framework

# Verify you're in the right place
ls -la
# You should see: lib/, docs/, README.md, mix.exs
```

**Step 2: Install Elixir and Erlang**

**On macOS (using Homebrew):**
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Elixir (includes Erlang)
brew install elixir

# Verify installation
elixir --version
# Should show: Elixir 1.14.x (compiled with Erlang/OTP 25+)
```

**On Ubuntu/Debian:**
```bash
# Add Erlang Solutions repository
wget https://packages.erlang-solutions.com/erlang-solutions_2.0_all.deb
sudo dpkg -i erlang-solutions_2.0_all.deb
sudo apt-get update

# Install Elixir and Erlang
sudo apt-get install elixir

# Verify installation
elixir --version
```

**On Fedora:**
```bash
# Install Elixir and Erlang
sudo dnf install elixir erlang

# Verify installation
elixir --version
```

**On Windows (via WSL2):**
```bash
# First, install WSL2 and Ubuntu from Microsoft Store
# Then follow Ubuntu instructions above
```

**Step 3: Install Dependencies**

```bash
# Fetch all Elixir dependencies
mix deps.get

# This downloads ~30 packages including:
# - req (HTTP client)
# - jason (JSON parsing)
# - telemetry (instrumentation)
# - statistex (statistical functions)
# - nx (numerical computing)
```

**Step 4: Compile the Framework**

```bash
# Compile all applications
mix compile

# This compiles 8 applications:
# - ensemble, hedging, bench, telemetry_research
# - causal_trace, dataset_manager, research_harness
```

**Step 5: Run Tests to Verify**

```bash
# Run all tests (takes ~30 seconds)
mix test

# You should see:
# Finished in X.X seconds
# XX tests, 0 failures
```

If all tests pass, your installation is complete!

### Method 2: From Hex (When Published)

Once the package is on Hex.pm, add it to your `mix.exs` and let Mix handle updates.

```elixir
# mix.exs
def deps do
  [
    {:crucible_framework, "~> X.Y.Z"}
  ]
end
```

```bash
mix deps.get
```

> Note: the Hex package is not yet published. Track the GitHub install until an official release is announced.

### Method 3: Local Development Setup

For contributors or anyone hacking on Crucible itself.

```bash
# Clone the development branch if you need unreleased work
git clone https://github.com/North-Shore-AI/crucible_framework.git
cd crucible_framework
git checkout develop  # optional

# Install dependencies including dev tools
mix deps.get

# Install git hooks for pre-commit checks
mix git_hooks.install

# Run full test suite with coverage
mix test --cover

# Generate documentation locally
mix docs
open doc/index.html  # macOS
xdg-open doc/index.html  # Linux
```

### Verifying Your Installation

Run this quick check:

```bash
# Start interactive Elixir shell
iex -S mix

# In IEx, try loading a module
iex> Ensemble
# Should show: Ensemble module documentation

# Exit IEx
iex> :q
```

If you see module documentation, everything is working!

---

## Configuration

### Setting Up API Keys

The framework reads API keys from environment variables for security.

**Option 1: Export in Shell** (Quick Testing)

```bash
# Set API keys for current session
export GEMINI_API_KEY="your-gemini-key-here"
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"

# Verify they're set
echo $GEMINI_API_KEY
```

**Option 2: .env File** (Recommended)

```bash
# Create .env file in project root
cat > .env << 'EOF'
# LLM API Keys
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: PostgreSQL for persistent telemetry
DATABASE_URL=postgresql://user:pass@localhost/crucible_framework_dev
EOF

# Add .env to .gitignore (IMPORTANT!)
echo ".env" >> .gitignore

# Load .env in your shell
set -a
source .env
set +a
```

**Option 3: Shell Profile** (Permanent)

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
echo 'export GEMINI_API_KEY="your-key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

### Framework Configuration

Create or edit `config/config.exs`:

```elixir
import Config

# Ensemble Configuration
config :ensemble,
  # Default models for ensemble predictions
  default_models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],

  # API timeout (milliseconds)
  default_timeout: 5_000,

  # Rate limiting (requests per second)
  rate_limit: 50

# Dataset Manager Configuration
config :dataset_manager,
  # Where to cache downloaded datasets
  cache_dir: Path.expand("~/.cache/crucible_framework/datasets"),

  # Dataset versions (for reproducibility)
  dataset_versions: %{
    mmlu: "1.0.0",
    human_eval: "1.0.0",
    gsm8k: "1.0.0"
  }

# Telemetry Research Configuration
config :telemetry_research,
  # Storage backend: :ets (in-memory) or :postgres (persistent)
  storage_backend: :ets,

  # For PostgreSQL backend (optional)
  # database_url: System.get_env("DATABASE_URL")

# Research Harness Configuration
config :research_harness,
  # Where to save experiment checkpoints
  checkpoint_dir: "./checkpoints",

  # Where to save results
  results_dir: "./results",

  # Checkpoint frequency (every N queries)
  checkpoint_interval: 50,

  # Enable cost confirmation prompts
  confirm_costs: true

# Logging (adjust for your needs)
config :logger,
  level: :info,
  format: "$time [$level] $metadata$message\n",
  metadata: [:experiment_id, :condition]
```

### Creating Directory Structure

```bash
# Create directories for outputs
mkdir -p results checkpoints research/experiments research/output

# Create dataset cache directory
mkdir -p ~/.cache/crucible_framework/datasets

# Verify structure
tree -L 2
# Should show:
# .
# ├── apps/
# ├── config/
# ├── results/
# ├── checkpoints/
# ├── research/
# │   ├── experiments/
# │   └── output/
```

### Testing Your Configuration

```bash
iex -S mix
```

```elixir
# Test that API keys are loaded
iex> System.get_env("GEMINI_API_KEY")
"your-gemini-key-here"  # Should show your key

# Test a simple prediction (costs ~$0.0001)
iex> {:ok, result} = Ensemble.predict("What is 2+2?", models: [:gemini_flash])
{:ok, %{answer: "4", metadata: %{...}}}

# Success! You're ready to run experiments.
```

---

## Your First Experiment: Step-by-Step Walkthrough

Let's run a complete experiment to compare single-model vs. ensemble reliability.

### Experiment Overview

**Research Question**: Does a 3-model ensemble achieve higher accuracy than a single model on factual questions?

**Hypothesis**: A 3-model ensemble (Gemini Flash, GPT-4o Mini, Claude Haiku) will achieve ≥5% higher accuracy than GPT-4o Mini alone on MMLU STEM questions.

**Experimental Design**:
- Control: Single model (GPT-4o Mini)
- Treatment: 3-model majority-vote ensemble
- Dataset: 100 questions from MMLU STEM benchmark
- Repetitions: 3 (for statistical power)
- Total queries: 100 × 2 conditions × 3 repetitions = 600 queries
- Estimated cost: ~$0.40-1.20
- Estimated time: ~3-5 minutes

### Step 1: Create the Experiment File

Create `research/experiments/ensemble_vs_single.exs`:

```elixir
defmodule Experiments.EnsembleVsSingle do
  @moduledoc """
  Experiment 1: Ensemble vs Single Model Reliability

  Research Question:
  Does a 3-model ensemble achieve higher accuracy than a single model?

  Hypothesis:
  Ensemble accuracy will be ≥5% higher than single model (p < 0.05).

  Design:
  - Control: GPT-4o Mini (single model)
  - Treatment: 3-model ensemble (Gemini Flash, GPT-4o Mini, Claude Haiku)
  - Dataset: MMLU STEM (100 questions)
  - Repetitions: 3
  - Total queries: 600
  """

  use ResearchHarness.Experiment

  # Experiment metadata
  name "Ensemble vs Single Model Reliability"

  description """
  This experiment tests whether multi-model ensemble voting improves
  accuracy on factual questions compared to single-model predictions.

  We use MMLU STEM questions, which are multiple-choice questions across
  science, technology, engineering, and mathematics topics.
  """

  # Dataset configuration
  dataset :mmlu_stem
  dataset_config %{
    sample_size: 100,
    seed: 42  # For reproducible sampling
  }

  # Experimental conditions
  conditions [
    %{
      name: "single_model",
      description: "GPT-4o Mini alone",
      fn: &single_model/1
    },
    %{
      name: "ensemble_3model",
      description: "3-model majority vote ensemble",
      fn: &ensemble_3model/1
    }
  ]

  # Metrics to collect
  metrics [
    :accuracy,      # Correctness of prediction
    :latency_p50,   # Median latency
    :latency_p95,   # 95th percentile latency
    :latency_p99,   # 99th percentile latency
    :cost_per_query # Cost in USD per query
  ]

  # Experimental design parameters
  repeat 3              # Run each condition 3 times
  randomize true        # Randomize query order to prevent bias
  seed 42               # Overall random seed for reproducibility

  # Resource limits
  timeout 10_000        # 10 seconds per query
  rate_limit 50         # Max 50 queries per second
  checkpoint_every 25   # Save progress every 25 queries

  # Statistical analysis configuration
  statistical_analysis %{
    significance_level: 0.05,
    confidence_interval: 0.95,
    effect_size_measure: :cohens_d,
    multiple_testing_correction: :bonferroni
  }

  # Implementation of single model condition
  def single_model(question) do
    start_time = System.monotonic_time(:millisecond)

    # Call OpenAI GPT-4o Mini
    {:ok, response} = call_openai_model(
      "gpt-4o-mini",
      question,
      temperature: 0.0  # Deterministic for reproducibility
    )

    end_time = System.monotonic_time(:millisecond)

    # Return result with metadata
    %{
      prediction: extract_answer(response),
      latency: end_time - start_time,
      cost: calculate_cost(response, :gpt4o_mini),
      metadata: %{
        model: :gpt4o_mini,
        raw_response: response
      }
    }
  end

  # Implementation of ensemble condition
  def ensemble_3model(question) do
    start_time = System.monotonic_time(:millisecond)

    # Use Ensemble library for multi-model voting
    {:ok, result} = Ensemble.predict(
      question,
      models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
      strategy: :majority,        # Majority vote
      execution: :parallel,       # All models simultaneously
      normalization: :lowercase_trim,  # Normalize responses
      timeout: 8_000              # 8 second timeout per model
    )

    end_time = System.monotonic_time(:millisecond)

    # Return result with metadata
    %{
      prediction: result.answer,
      latency: end_time - start_time,
      cost: result.metadata.cost_usd,
      consensus: result.metadata.consensus,
      metadata: %{
        models: result.metadata.models_used,
        votes: result.metadata.votes,
        successes: result.metadata.successes,
        failures: result.metadata.failures
      }
    }
  end

  # Helper functions

  defp call_openai_model(model, question, opts) do
    # Simplified - actual implementation uses Req library
    # See apps/ensemble/lib/ensemble/executor.ex for full version

    api_key = System.get_env("OPENAI_API_KEY")

    body = %{
      model: model,
      messages: [
        %{role: "user", content: question}
      ],
      temperature: Keyword.get(opts, :temperature, 0.0),
      max_tokens: 100
    }

    case Req.post("https://api.openai.com/v1/chat/completions",
      json: body,
      auth: {:bearer, api_key}
    ) do
      {:ok, %{status: 200, body: response}} ->
        {:ok, response["choices"] |> List.first() |> get_in(["message", "content"])}

      {:ok, %{status: status}} ->
        {:error, {:api_error, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp extract_answer(response) do
    # Extract answer from response
    # For MMLU, answers are typically A, B, C, or D
    response
    |> String.trim()
    |> String.downcase()
    |> String.first()
  end

  defp calculate_cost(response, model) do
    # Simplified cost calculation
    # Actual implementation in apps/ensemble/lib/ensemble/pricing.ex

    token_count = String.length(response) / 4  # Rough estimate

    case model do
      :gpt4o_mini -> token_count * 0.00015 / 1_000_000
      :gemini_flash -> token_count * 0.000075 / 1_000_000
      :anthropic_haiku -> token_count * 0.00025 / 1_000_000
      _ -> 0.0
    end
  end
end
```

### Step 2: Estimate Cost and Time

Before running, check the estimates:

```bash
iex -S mix
```

```elixir
# Load the experiment
iex> Code.require_file("research/experiments/ensemble_vs_single.exs")

# Get cost and time estimates
iex> {:ok, estimates} = ResearchHarness.estimate(Experiments.EnsembleVsSingle)

# View cost estimate
iex> IO.puts("Total estimated cost: $#{estimates.cost.total_cost}")
Total estimated cost: $0.85

iex> IO.puts("Cost breakdown:")
iex> IO.inspect(estimates.cost.per_condition, label: "Per condition")
Per condition: %{
  single_model: 0.19,
  ensemble_3model: 0.66
}

# View time estimate
iex> IO.puts("Estimated duration: #{div(estimates.time.estimated_duration, 1000)}s")
Estimated duration: 180s  # ~3 minutes

iex> IO.inspect(estimates.time, label: "Time breakdown")
Time breakdown: %{
  estimated_duration: 180_000,  # milliseconds
  per_query_avg: 900,            # average per query
  parallel_speedup: 2.0,         # speedup from parallelism
  wall_clock_time: "3m 0s"
}
```

The estimates show:
- **Total cost**: ~$0.85 (well within budget)
- **Duration**: ~3 minutes (acceptable for initial run)
- **Per query cost**: $0.0014 average

### Step 3: Run the Experiment

```elixir
# Run with confirmation prompt
iex> {:ok, report} = ResearchHarness.run(
  Experiments.EnsembleVsSingle,
  output_dir: "results/ensemble_vs_single",
  formats: [:markdown, :html, :latex]
)

# You'll see:
# ========================================
# Experiment: Ensemble vs Single Model Reliability
# ========================================
#
# Dataset: mmlu_stem (100 samples)
# Conditions: 2 (single_model, ensemble_3model)
# Repetitions: 3
# Total queries: 600
#
# Estimated cost: $0.85
# Estimated time: 3m 0s
#
# Continue? [y/N]: y

# Progress updates:
# [===>                    ] 15% (90/600) - ETA 2m 15s
# [=======>                ] 30% (180/600) - ETA 1m 45s
# [===========>            ] 50% (300/600) - ETA 1m 15s
# [===============>        ] 70% (420/600) - ETA 45s
# [==================>     ] 85% (510/600) - ETA 20s
# [====================    ] 95% (570/600) - ETA 5s
# [========================] 100% (600/600) - Complete!
#
# Experiment complete in 3m 12s
# Results saved to: results/ensemble_vs_single/

{:ok, %ResearchHarness.Report{
  experiment_id: "exp_a3f2c1b",
  reports: %{
    markdown: "results/ensemble_vs_single/exp_a3f2c1b_report.md",
    html: "results/ensemble_vs_single/exp_a3f2c1b_report.html",
    latex: "results/ensemble_vs_single/exp_a3f2c1b_report.tex"
  },
  results_csv: "results/ensemble_vs_single/exp_a3f2c1b_results.csv",
  analysis_json: "results/ensemble_vs_single/exp_a3f2c1b_analysis.json"
}}
```

### Step 4: View Your Results

```bash
# View Markdown report in terminal
cat results/ensemble_vs_single/exp_a3f2c1b_report.md

# Open HTML report in browser (more readable)
open results/ensemble_vs_single/exp_a3f2c1b_report.html  # macOS
xdg-open results/ensemble_vs_single/exp_a3f2c1b_report.html  # Linux
```

---

## Understanding Your Results

Your experiment generates comprehensive reports. Let's walk through each section.

### Example Report Structure

```markdown
# Experiment Report: Ensemble vs Single Model Reliability

**Experiment ID**: exp_a3f2c1b
**Started**: 2025-10-08 14:30:00 UTC
**Completed**: 2025-10-08 14:33:12 UTC
**Duration**: 3m 12s
**Total Cost**: $0.87

## 1. Executive Summary

The 3-model ensemble achieved significantly higher accuracy (94.7%) compared
to the single model baseline (89.3%), representing a 5.4 percentage point
improvement. This difference is statistically significant (p < 0.001) with a
large effect size (Cohen's d = 1.82).

**Key Findings:**
- ✓ Ensemble accuracy significantly higher (p < 0.001)
- ✓ Effect size is large (d = 1.82)
- ✗ Ensemble latency 35% higher (acceptable trade-off)
- ✓ Cost increase (3.5×) acceptable for reliability gain

**Recommendation**: Use ensemble for critical applications where accuracy
outweighs cost considerations.

## 2. Descriptive Statistics

### Accuracy

| Condition | N | Mean | SD | 95% CI | Min | Max |
|-----------|---|------|----|----|-----|-----|
| single_model | 300 | 0.893 | 0.021 | [0.890, 0.896] | 0.83 | 0.95 |
| ensemble_3model | 300 | 0.947 | 0.015 | [0.945, 0.949] | 0.91 | 0.98 |

**Interpretation**:
The ensemble shows higher mean accuracy and lower variance (SD = 0.015 vs
0.021), indicating both better performance and more consistency.

### Latency (milliseconds)

| Condition | P50 | P95 | P99 | Mean | SD |
|-----------|-----|-----|-----|------|-----|
| single_model | 820 | 1850 | 3200 | 950 | 450 |
| ensemble_3model | 1250 | 2450 | 4100 | 1450 | 580 |

**Interpretation**:
Ensemble latency is higher as expected (parallel execution waits for slowest
model). P50 increased 52%, P95 increased 32%, P99 increased 28%.

### Cost Per Query (USD)

| Condition | Mean | SD | Total |
|-----------|------|----|-------|
| single_model | 0.00019 | 0.00002 | $0.057 |
| ensemble_3model | 0.00066 | 0.00005 | $0.198 |

**Cost-Accuracy Trade-off**:
- Cost increase: 3.5× ($0.00066 vs $0.00019 per query)
- Accuracy improvement: 5.4 percentage points
- Cost per accuracy point: $0.025 per 1% improvement

## 3. Statistical Analysis

### Primary Comparison: Accuracy

**Test Used**: Welch's t-test
**Reason**: Normal distributions (Shapiro-Wilk p > 0.05), unequal variances (Levene p = 0.03)

**Results**:
- t(567.3) = 8.91
- p < 0.001 (two-tailed)
- Cohen's d = 1.82 (very large effect)
- 95% CI for difference: [0.048, 0.060]

**Interpretation**:
The ensemble achieved significantly higher accuracy (M = 0.947, SD = 0.015)
compared to the single model (M = 0.893, SD = 0.021), t(567.3) = 8.91,
p < 0.001, d = 1.82.

This represents a very large effect size (d > 0.8), providing strong
evidence that ensemble voting substantially improves accuracy.

### Power Analysis

**Achieved Power**: 0.99
**Minimum Detectable Effect**: 0.35 (small effect)

The experiment was well-powered to detect the observed effect. With 300
observations per group, we could reliably detect effects as small as d = 0.35.

### Assumptions Check

**Normality**:
- single_model: Shapiro-Wilk W = 0.987, p = 0.18 ✓ Pass
- ensemble_3model: Shapiro-Wilk W = 0.992, p = 0.41 ✓ Pass

**Homogeneity of Variance**:
- Levene's test: F(1, 598) = 4.72, p = 0.03 ✗ Fail

Interpretation: Data is normally distributed but variances differ. Welch's
t-test (which doesn't assume equal variances) is appropriate.

## 4. Visualizations

[In HTML report, interactive charts appear here showing:]
- Box plots of accuracy by condition
- Violin plots showing distribution shapes
- Time series of accuracy across experiment
- Cost vs accuracy scatter plot

## 5. Detailed Results

### Per-Question Analysis

| Question ID | Topic | Single Correct? | Ensemble Correct? | Ensemble Vote |
|-------------|-------|-----------------|-------------------|---------------|
| mmlu_stem_1 | Physics | Yes | Yes | 3/3 unanimous |
| mmlu_stem_2 | Chemistry | No | Yes | 2/3 majority |
| mmlu_stem_3 | Biology | Yes | Yes | 3/3 unanimous |
| ... | ... | ... | ... | ... |

### Failure Analysis

**Single Model Failures**: 32 questions (10.7%)
- Physics: 8 questions
- Chemistry: 12 questions
- Mathematics: 7 questions
- Biology: 5 questions

**Ensemble Failures**: 16 questions (5.3%)
- Physics: 4 questions (50% reduction)
- Chemistry: 6 questions (50% reduction)
- Mathematics: 3 questions (57% reduction)
- Biology: 3 questions (40% reduction)

**Pattern**: Ensemble reduces failures across all topics, with greatest
improvement in mathematics (57% reduction).

## 6. Ensemble Consensus Analysis

| Consensus Level | Count | Percentage | Accuracy |
|----------------|-------|------------|----------|
| Unanimous (3/3) | 267 | 89.0% | 98.1% |
| Majority (2/3) | 33 | 11.0% | 72.7% |
| No consensus | 0 | 0.0% | N/A |

**Key Insight**: When all 3 models agree (89% of cases), accuracy is 98.1%.
When only 2/3 agree, accuracy drops to 72.7%. This suggests consensus level
is a useful confidence indicator.

## 7. Cost-Benefit Analysis

| Metric | Single Model | Ensemble | Change |
|--------|--------------|----------|--------|
| Accuracy | 89.3% | 94.7% | +5.4 pp |
| Cost per query | $0.00019 | $0.00066 | +247% |
| Cost per 1% accuracy | $0.00021 | $0.00070 | +233% |
| Errors per 1000 queries | 107 | 53 | -50% |
| Cost to avoid 1 error | $0.0018 | $0.0125 | +594% |

**Interpretation**:
- Ensemble reduces errors by 50% (54 fewer errors per 1000 queries)
- Each avoided error costs $0.0125
- For applications where errors are costly, this is worthwhile

**Example ROI**:
If each error costs $1 to fix manually:
- Single model: 107 errors × $1 = $107 per 1000 queries
- Ensemble: 53 errors × $1 = $53 per 1000 queries
- Savings: $54 per 1000 queries
- Additional cost: $0.47 per 1000 queries
- Net benefit: $53.53 per 1000 queries

## 8. Conclusions

### Hypothesis Evaluation

**Hypothesis**: Ensemble accuracy will be ≥5% higher than single model (p < 0.05)

**Result**: ✓ SUPPORTED
- Observed difference: 5.4 percentage points
- Statistical significance: p < 0.001
- Effect size: d = 1.82 (very large)

### Scientific Implications

1. **Multi-model voting works**: Consistent with ensemble learning theory
2. **Cost-accuracy trade-off**: 3.5× cost for 5.4pp accuracy gain
3. **Consensus as confidence**: Unanimous votes → 98% accuracy
4. **Practical applicability**: Suitable for high-stakes decisions

### Limitations

1. **Single dataset**: Results specific to MMLU STEM questions
2. **Model selection**: Different models may yield different results
3. **Temporal validity**: Model versions may change over time
4. **Cost assumptions**: API pricing subject to change

### Future Work

1. Test on other datasets (HumanEval, GSM8K)
2. Explore 5-model and 7-model ensembles
3. Investigate weighted voting vs majority
4. Study cost optimization with sequential execution

## 9. Reproducibility

### Configuration

```elixir
%{
  experiment: "Ensemble vs Single Model Reliability",
  dataset: :mmlu_stem,
  sample_size: 100,
  repetitions: 3,
  seed: 42,
  models: %{
    single: [:openai_gpt4o_mini],
    ensemble: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
  },
  voting_strategy: :majority,
  execution_strategy: :parallel
}
```

### Environment

```yaml
framework_version: X.Y.Z
elixir_version: 1.14.0
erlang_version: 25.0
dataset_version: mmlu-1.0.0
model_versions:
  gemini_flash: gemini-1.5-flash-latest
  openai_gpt4o_mini: gpt-4o-mini-2024-07-18
  anthropic_haiku: claude-3-haiku-20240307
```

### Artifacts

All data and code to reproduce this experiment:
- Configuration: `results/ensemble_vs_single/exp_a3f2c1b_config.json`
- Raw results: `results/ensemble_vs_single/exp_a3f2c1b_results.csv`
- Analysis: `results/ensemble_vs_single/exp_a3f2c1b_analysis.json`
- Checkpoints: `checkpoints/exp_a3f2c1b/`

To reproduce:
```bash
git checkout <commit-sha>
mix deps.get
export EXPERIMENT_SEED=42
mix run research/experiments/ensemble_vs_single.exs
```

## 10. References

### Related Work

1. Dean, J., & Barroso, L. A. (2013). The tail at scale. Communications of the ACM, 56(2), 74-80.
2. Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.
3. Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. ICLR.

### Framework Documentation

- Architecture: ARCHITECTURE.md
- Statistical Methods: See `crucible_bench` package
- Ensemble Guide: See `crucible_ensemble` package
```

### Key Sections to Understand

**1. Executive Summary**
- Quick overview of findings
- Decision: Use ensemble or not?
- Most important metrics

**2. Descriptive Statistics**
- Raw numbers: means, standard deviations, ranges
- Compare conditions side-by-side
- Look for patterns

**3. Statistical Analysis**
- Is the difference real or random chance?
- p-value < 0.05 = statistically significant
- Effect size: how big is the difference?
  - Small: d = 0.2-0.5
  - Medium: d = 0.5-0.8
  - Large: d > 0.8

**4. Assumptions Check**
- Normality: Is data normally distributed?
- Variance: Are variances equal?
- If assumptions fail, different test used

**5. Consensus Analysis** (Ensemble-specific)
- How often do models agree?
- Is consensus related to accuracy?
- Use for confidence estimation

**6. Cost-Benefit Analysis**
- Is improvement worth the cost?
- Context-dependent decision
- Calculate ROI for your use case

---

## Core Concepts

### 1. Ensemble Voting

**What is it?**
Query multiple models and aggregate their responses.

**Why does it work?**
If models make independent errors, ensemble reduces error probability exponentially.

**Example:**
- 3 models, each 90% accurate
- If errors are independent: 1 - (0.1)³ = 99.9% accuracy
- In practice: ~95-97% (errors partially correlated)

**Voting Strategies:**

```elixir
# Majority vote (most common answer wins)
{:ok, result} = Ensemble.predict(query, strategy: :majority)

# Weighted vote (weight by confidence scores)
{:ok, result} = Ensemble.predict(query, strategy: :weighted)

# Best confidence (highest confidence answer)
{:ok, result} = Ensemble.predict(query, strategy: :best_confidence)

# Unanimous (all models must agree)
{:ok, result} = Ensemble.predict(query, strategy: :unanimous)
```

### 2. Request Hedging

**What is it?**
Send a backup request if primary is slow.

**Why does it work?**
Exploits high variance in API latencies. Backup request races primary.

**Example:**
```elixir
# Hedge after P95 latency (learned from history)
Hedging.request(fn ->
  call_api()
end, strategy: :percentile, percentile: 95)

# Results:
# - P99 latency: 5000ms → 2000ms (60% reduction)
# - Cost: 1.0× → 1.1× (10% increase)
# - Hedge fires ~5-10% of time
```

**Strategies:**
- `:fixed` - Fixed delay (e.g., 100ms)
- `:percentile` - Delay = Pth percentile latency
- `:adaptive` - ML-based delay prediction
- `:workload_aware` - Different delays per request type

### 3. Statistical Testing

**What is it?**
Determine if observed differences are statistically significant.

**Why is it important?**
Distinguishes real effects from random noise.

**Example:**
```elixir
control = [0.89, 0.87, 0.90, 0.88, 0.91]  # Baseline accuracies
treatment = [0.96, 0.97, 0.94, 0.95, 0.98]  # Ensemble accuracies

result = Bench.compare(control, treatment)

# Result:
# %{
#   test: :welch_t_test,
#   p_value: 0.00012,
#   effect_size: %{cohens_d: 4.52},
#   interpretation: "Treatment significantly higher (p < 0.001)"
# }
```

**Key Metrics:**
- **p-value**: Probability result is due to chance
  - p < 0.05 = statistically significant
  - p < 0.01 = highly significant
  - p < 0.001 = very highly significant
- **Effect size**: Magnitude of difference
  - Cohen's d = (M₁ - M₂) / pooled_SD
  - d = 0.2 (small), 0.5 (medium), 0.8 (large)
- **Confidence interval**: Range of plausible values
  - 95% CI = [0.05, 0.10] means true difference likely 5-10%

### 4. Experimental Design

**Key Principles:**

**Randomization**: Prevent order effects
```elixir
randomize true  # Randomize query order
seed 42         # Reproducible randomization
```

**Repetition**: Increase statistical power
```elixir
repeat 3  # Run each condition 3 times
# Power increases with √n repetitions
```

**Control**: Compare against baseline
```elixir
conditions [
  %{name: "baseline", fn: &baseline/1},  # Control
  %{name: "treatment", fn: &treatment/1}  # Treatment
]
```

**Blinding**: Prevent bias (automated in framework)
- Experimenter doesn't know which condition is which
- Analysis automated to prevent cherry-picking

### 5. Reproducibility

**Three Pillars:**

**1. Deterministic Seeding**
```elixir
seed 42  # All randomness deterministic

# Query sampling: deterministic
# Model selection: deterministic
# Tie-breaking: deterministic
```

**2. Version Tracking**
```yaml
  # Automatically saved in results
  framework_version: X.Y.Z
  elixir_version: 1.14.0
  dataset_version: mmlu-1.0.0
model_versions:
  gpt4: gpt-4-0613
  claude: claude-3-opus-20240229
```

**3. Complete Artifact Preservation**
```
results/exp_abc123/
├── config.json          # Full configuration
├── environment.json     # System info
├── dataset.jsonl        # Exact dataset used
├── results.csv          # Raw results
├── analysis.json        # Statistical analysis
└── checkpoints/         # Intermediate state
```

**Verification:**
```bash
# Anyone can reproduce your results
git checkout <your-commit>
mix deps.get
export EXPERIMENT_SEED=42
mix run experiments/your_experiment.exs

# Results should match exactly
diff results/original results/reproduction
# No differences = perfect reproduction
```

---

## LoRA Adapter Layer (Tinkex Default)

**New in X.Y.Z:** Tinkex overlay configuration now resolves under the `:crucible_framework` application environment (not `:crucible_tinkex`), keeping API tokens, default models, runner mode, and submission hooks centralized in the framework config. The adapter-neutral LoRA interface (`Crucible.Lora`) remains the default surface, and the bundled adapter still targets the [Tinkex](https://hex.pm/packages/tinkex) SDK. To swap adapters, implement the `Crucible.Lora.Adapter` behaviour and set `config :crucible_framework, :lora_adapter, MyAdapter`.

### Building Blocks
- `Crucible.Lora` – facade for experiment creation, batching, formatting, metrics, and checkpoint helpers
- `Crucible.Tinkex.Config` – wraps API keys, base URLs, retry logic, default LoRA hyperparameters, and quality targets
- `Crucible.Tinkex.Experiment` – declarative structure for datasets, sweeps, checkpoints, and repeat counts
- `Crucible.Tinkex.Telemetry` – emits `[:crucible, :tinkex, ...]` events so fine-tuning runs show up alongside ResearchHarness metrics
- `Crucible.Tinkex.QualityValidator` – enforces CNS3-derived schema/citation/entailment gates
- `Crucible.Tinkex.Results` – aggregates training/evaluation metrics and exports CSV-ready reports

### Minimal Setup

```elixir
# runtime.exs
config :crucible_framework, :lora_adapter, Crucible.Tinkex

config = Crucible.Tinkex.Config.new(
  api_key: System.fetch_env!("TINKEX_KEY"),
  base_url: "https://tinkex.example.com"
)

{:ok, experiment} =
  Crucible.Lora.create_experiment(
    name: "SciFact Claim Extractor",
    base_model: config.default_base_model,
    training: %{epochs: 4, batch_size: 16},
    parameters: %{learning_rate: [1.0e-4, 2.0e-4]}
  )

Crucible.Tinkex.Telemetry.attach(experiment_id: experiment.id)

experiment
|> Crucible.Tinkex.Experiment.generate_runs()
|> Enum.each(fn run ->
  dataset
  |> Crucible.Lora.batch_dataset(16)
  |> Enum.each(fn batch ->
    formatted = Crucible.Lora.format_training_data(batch)
    # pass formatted batch to adapter forward/backward call
  end)
end)
```

### Quality Gates and Reporting
- After evaluation completes, call `Crucible.Tinkex.QualityValidator.validate(results)` to verify schema compliance, citation accuracy, entailment, and pass rates
- Persist checkpoints with `Crucible.Lora.checkpoint_name/2` for deterministic artifact naming
- Build final summaries via `Crucible.Tinkex.Results.to_report_data/1` and include them in ResearchHarness experiment exports

Read the README section on the LoRA adapter layer plus [INSTRUMENTATION.md](./INSTRUMENTATION.md) for telemetry wiring and PostgreSQL export guidance.

---

## Common Patterns and Examples

### Example 1: A/B Testing Two Prompts

```elixir
defmodule Experiments.PromptABTest do
  use ResearchHarness.Experiment

  name "Prompt Engineering A/B Test"
  dataset :gsm8k, sample_size: 200

  conditions [
    %{name: "prompt_a", fn: &prompt_a/1},
    %{name: "prompt_b", fn: &prompt_b/1}
  ]

  metrics [:accuracy, :cost_per_query]
  repeat 3

  def prompt_a(question) do
    prompt = "Solve this math problem step by step:\n#{question}"
    call_model(:gpt4o_mini, prompt)
  end

  def prompt_b(question) do
    prompt = """
    You are a math teacher. Solve this problem showing your work:

    #{question}

    Solution:
    """
    call_model(:gpt4o_mini, prompt)
  end
end
```

### Example 2: Model Comparison

```elixir
defmodule Experiments.ModelComparison do
  use ResearchHarness.Experiment

  name "Compare GPT-4o Mini vs Claude Haiku vs Gemini Flash"
  dataset :mmlu_stem, sample_size: 100

  conditions [
    %{name: "gpt4o_mini", fn: fn q -> call_model(:gpt4o_mini, q) end},
    %{name: "claude_haiku", fn: fn q -> call_model(:claude_haiku, q) end},
    %{name: "gemini_flash", fn: fn q -> call_model(:gemini_flash, q) end}
  ]

  metrics [:accuracy, :latency_p99, :cost_per_query]
  repeat 5

  # Framework automatically applies:
  # - One-way ANOVA (3+ groups)
  # - Post-hoc pairwise comparisons
  # - Bonferroni correction for multiple testing
end
```

### Example 3: Ensemble Size Optimization

```elixir
defmodule Experiments.EnsembleSize do
  use ResearchHarness.Experiment

  name "Optimal Ensemble Size (1, 3, 5, 7 models)"
  dataset :mmlu_stem, sample_size: 150

  conditions [
    %{name: "single", fn: &single/1},
    %{name: "ensemble_3", fn: &ensemble_3/1},
    %{name: "ensemble_5", fn: &ensemble_5/1},
    %{name: "ensemble_7", fn: &ensemble_7/1}
  ]

  metrics [:accuracy, :cost_per_query, :latency_p99]
  repeat 3

  def single(q), do: call_model(:gpt4o_mini, q)

  def ensemble_3(q) do
    Ensemble.predict(q, models: [:gpt4o_mini, :claude_haiku, :gemini_flash])
  end

  def ensemble_5(q) do
    Ensemble.predict(q, models: [:gpt4o_mini, :claude_haiku, :gemini_flash,
                                   :gpt35_turbo, :claude_sonnet])
  end

  def ensemble_7(q) do
    Ensemble.predict(q, models: [:gpt4o_mini, :claude_haiku, :gemini_flash,
                                   :gpt35_turbo, :claude_sonnet, :gpt4o, :gemini_pro])
  end
end
```

### Example 4: Hedging Latency Study

```elixir
defmodule Experiments.HedgingLatency do
  use ResearchHarness.Experiment

  name "Request Hedging for Tail Latency Reduction"
  dataset :test_dataset, sample_size: 1000  # Large sample for latency

  conditions [
    %{name: "baseline", fn: &baseline/1},
    %{name: "hedged_p95", fn: &hedged_p95/1},
    %{name: "hedged_p99", fn: &hedged_p99/1}
  ]

  metrics [:latency_p50, :latency_p95, :latency_p99, :cost_per_query]
  repeat 1  # Don't need repetitions for latency study

  def baseline(q) do
    call_model(:claude_haiku, q)
  end

  def hedged_p95(q) do
    Hedging.request(fn ->
      call_model(:claude_haiku, q)
    end, strategy: :percentile, percentile: 95)
  end

  def hedged_p99(q) do
    Hedging.request(fn ->
      call_model(:claude_haiku, q)
    end, strategy: :percentile, percentile: 99)
  end
end
```

### Example 5: Voting Strategy Comparison

```elixir
defmodule Experiments.VotingStrategies do
  use ResearchHarness.Experiment

  name "Compare Voting Strategies: Majority vs Weighted vs Best Confidence"
  dataset :mmlu_stem, sample_size: 100

  conditions [
    %{name: "majority", fn: &majority/1},
    %{name: "weighted", fn: &weighted/1},
    %{name: "best_confidence", fn: &best_confidence/1}
  ]

  metrics [:accuracy, :consensus]
  repeat 3

  @models [:gpt4o_mini, :claude_haiku, :gemini_flash]

  def majority(q) do
    Ensemble.predict(q, models: @models, strategy: :majority)
  end

  def weighted(q) do
    Ensemble.predict(q, models: @models, strategy: :weighted)
  end

  def best_confidence(q) do
    Ensemble.predict(q, models: @models, strategy: :best_confidence)
  end
end
```

### Example 6: Cost-Accuracy Trade-off

```elixir
defmodule Experiments.CostAccuracyTradeoff do
  use ResearchHarness.Experiment

  name "Cost-Accuracy Trade-off: Cheap vs Expensive Models"
  dataset :mmlu_stem, sample_size: 100

  conditions [
    %{name: "cheap_single", fn: &cheap_single/1},
    %{name: "cheap_ensemble", fn: &cheap_ensemble/1},
    %{name: "expensive_single", fn: &expensive_single/1},
    %{name: "mixed_ensemble", fn: &mixed_ensemble/1}
  ]

  metrics [:accuracy, :cost_per_query]
  repeat 3

  def cheap_single(q), do: call_model(:gemini_flash, q)

  def cheap_ensemble(q) do
    Ensemble.predict(q, models: [:gemini_flash, :gpt4o_mini, :claude_haiku])
  end

  def expensive_single(q), do: call_model(:gpt4o, q)

  def mixed_ensemble(q) do
    Ensemble.predict(q, models: [:gemini_flash, :gpt4o_mini, :gpt4o])
  end

  # Custom analysis: cost per accuracy point
  def analyze_results(results) do
    Enum.map(results, fn {condition, metrics} ->
      efficiency = metrics.accuracy / metrics.cost_per_query
      {condition, %{efficiency: efficiency}}
    end)
    |> Enum.into(%{})
  end
end
```

### Example 7: Sequential vs Parallel Execution

```elixir
defmodule Experiments.ExecutionStrategies do
  use ResearchHarness.Experiment

  name "Sequential vs Parallel Ensemble Execution"
  dataset :mmlu_stem, sample_size: 100

  conditions [
    %{name: "parallel", fn: &parallel/1},
    %{name: "sequential", fn: &sequential/1},
    %{name: "cascade", fn: &cascade/1}
  ]

  metrics [:accuracy, :latency_p50, :cost_per_query]
  repeat 3

  @models [:gemini_flash, :gpt4o_mini, :claude_haiku]

  def parallel(q) do
    Ensemble.predict(q, models: @models, execution: :parallel)
  end

  def sequential(q) do
    Ensemble.predict(q, models: @models, execution: :sequential)
  end

  def cascade(q) do
    # Start with cheap/fast, escalate if needed
    Ensemble.predict(q, models: @models, execution: :cascade)
  end
end
```

### Example 8: Dataset Comparison

```elixir
defmodule Experiments.DatasetComparison do
  use ResearchHarness.Experiment

  name "Model Performance Across Different Datasets"

  # Run same model on multiple datasets
  conditions [
    %{name: "mmlu_stem", fn: fn q ->
      {:ok, dataset} = DatasetManager.load(:mmlu_stem, sample_size: 100)
      test_on_dataset(dataset, :gpt4o_mini)
    end},
    %{name: "gsm8k", fn: fn q ->
      {:ok, dataset} = DatasetManager.load(:gsm8k, sample_size: 100)
      test_on_dataset(dataset, :gpt4o_mini)
    end},
    %{name: "human_eval", fn: fn q ->
      {:ok, dataset} = DatasetManager.load(:human_eval, sample_size: 50)
      test_on_dataset(dataset, :gpt4o_mini)
    end}
  ]

  metrics [:accuracy]
  repeat 3

  defp test_on_dataset(dataset, model) do
    results = Enum.map(dataset.items, fn item ->
      {:ok, response} = call_model(model, item.input)
      %{predicted: response, expected: item.expected}
    end)

    {:ok, eval} = DatasetManager.evaluate(results, dataset: dataset)
    %{accuracy: eval.accuracy}
  end
end
```

### Example 9: Temperature Sensitivity

```elixir
defmodule Experiments.TemperatureSensitivity do
  use ResearchHarness.Experiment

  name "Effect of Temperature on Accuracy"
  dataset :mmlu_stem, sample_size: 100

  conditions [
    %{name: "temp_0.0", fn: fn q -> call_model(:gpt4o_mini, q, temperature: 0.0) end},
    %{name: "temp_0.3", fn: fn q -> call_model(:gpt4o_mini, q, temperature: 0.3) end},
    %{name: "temp_0.7", fn: fn q -> call_model(:gpt4o_mini, q, temperature: 0.7) end},
    %{name: "temp_1.0", fn: fn q -> call_model(:gpt4o_mini, q, temperature: 1.0) end}
  ]

  metrics [:accuracy]
  repeat 5  # More repetitions for noisy temperature
end
```

### Example 10: Real-World Application Scenario

```elixir
defmodule Experiments.CustomerSupportClassification do
  use ResearchHarness.Experiment

  name "Customer Support Ticket Classification"
  description """
  Real-world scenario: Classify customer support tickets into categories.
  Compare single model vs ensemble for production deployment decision.
  """

  dataset :custom_support_tickets  # Your custom dataset
  dataset_config %{
    path: "data/support_tickets.jsonl",
    sample_size: 500
  }

  conditions [
    %{
      name: "production_single",
      description: "Current production: GPT-4o Mini",
      fn: &production_single/1
    },
    %{
      name: "proposed_ensemble",
      description: "Proposed: 3-model ensemble",
      fn: &proposed_ensemble/1
    }
  ]

  metrics [
    :accuracy,
    :precision,
    :recall,
    :f1_score,
    :latency_p95,
    :cost_per_query
  ]

  repeat 3

  # Production budget constraints
  cost_budget %{
    max_total: 50.00,        # $50 total budget
    max_per_condition: 25.00
  }

  latency_budget %{
    p95_max: 2000  # 2 second P95 latency SLA
  }

  def production_single(ticket) do
    prompt = build_classification_prompt(ticket)
    {:ok, response} = call_model(:gpt4o_mini, prompt, temperature: 0.0)

    %{
      prediction: parse_category(response),
      latency: measure_latency(),
      cost: calculate_cost(:gpt4o_mini, response)
    }
  end

  def proposed_ensemble(ticket) do
    prompt = build_classification_prompt(ticket)

    {:ok, result} = Ensemble.predict(
      prompt,
      models: [:gemini_flash, :gpt4o_mini, :claude_haiku],
      strategy: :weighted,
      execution: :parallel
    )

    %{
      prediction: parse_category(result.answer),
      latency: result.metadata.latency_ms,
      cost: result.metadata.cost_usd,
      consensus: result.metadata.consensus
    }
  end

  defp build_classification_prompt(ticket) do
    """
    Classify this customer support ticket into one category:
    - billing
    - technical
    - account
    - general

    Ticket: #{ticket.text}

    Category:
    """
  end

  defp parse_category(response) do
    response
    |> String.downcase()
    |> String.trim()
    |> String.split()
    |> List.first()
  end

  # Custom analysis: ROI calculation
  def analyze_results(results) do
    baseline = results["production_single"]
    proposed = results["proposed_ensemble"]

    # Calculate misclassification cost
    # Assume each misclassification costs $5 in customer service time
    baseline_errors = (1 - baseline.accuracy) * 1000 * 5.00  # per 1000 tickets
    proposed_errors = (1 - proposed.accuracy) * 1000 * 5.00

    cost_savings = baseline_errors - proposed_errors
    additional_cost = (proposed.cost_per_query - baseline.cost_per_query) * 1000

    net_benefit = cost_savings - additional_cost
    roi = net_benefit / additional_cost * 100

    %{
      cost_savings: cost_savings,
      additional_cost: additional_cost,
      net_benefit: net_benefit,
      roi_percent: roi,
      recommendation: if roi > 100, do: "Deploy ensemble", else: "Keep single model"
    }
  end
end
```

---

## Troubleshooting

### Interactive Debugging

**Start IEx with Mix:**
```bash
iex -S mix
```

**Test Individual Components:**
```elixir
# Test API keys
iex> System.get_env("OPENAI_API_KEY")
"sk-..."  # Should show your key

# Test simple API call
iex> {:ok, result} = Ensemble.predict("What is 2+2?", models: [:gemini_flash])
{:ok, %{answer: "4", ...}}

# Test dataset loading
iex> {:ok, dataset} = DatasetManager.load(:mmlu_stem, sample_size: 10)
{:ok, %DatasetManager.Dataset{...}}

# Test statistical functions
iex> Bench.Stats.mean([1, 2, 3, 4, 5])
3.0
```

**Enable Verbose Logging:**
```elixir
# In config/config.exs
config :logger, level: :debug

# Or in IEx
iex> Logger.configure(level: :debug)
```

**Inspect Telemetry Events:**
```elixir
# Attach to all telemetry events
:telemetry.attach_many(
  "debug-handler",
  [
    [:ensemble, :predict, :start],
    [:ensemble, :predict, :stop],
    [:ensemble, :predict, :exception]
  ],
  fn event, measurements, metadata, _config ->
    IO.inspect({event, measurements, metadata}, label: "TELEMETRY")
  end,
  nil
)

# Now run your code and see all events
Ensemble.predict("test", models: [:gemini_flash])
```

### Common Debugging Patterns

**Pattern 1: Isolate the Problem**
```elixir
# Start with simplest possible test
{:ok, _} = Ensemble.predict("test", models: [:gemini_flash])

# Add complexity incrementally
{:ok, _} = Ensemble.predict("test", models: [:gemini_flash, :gpt4o_mini])

# Add ensemble features
{:ok, _} = Ensemble.predict("test",
  models: [:gemini_flash, :gpt4o_mini],
  strategy: :majority
)
```

**Pattern 2: Check Intermediate Values**
```elixir
# Use IO.inspect with labels
result = some_value
|> IO.inspect(label: "After step 1")
|> transform()
|> IO.inspect(label: "After transform")
|> final_step()
```

**Pattern 3: Use Guard Clauses**
```elixir
def my_function(value) do
  IO.puts("Input: #{inspect(value)}")

  result = process(value)
  IO.puts("Result: #{inspect(result)}")

  result
end
```

---

## Common Issues and Solutions

### Issue 1: API Key Not Found

**Symptom:**
```
** (RuntimeError) API key not found for model: gemini_flash
```

**Solution:**
```bash
# Check if environment variable is set
echo $GEMINI_API_KEY

# If empty, set it
export GEMINI_API_KEY="your-key-here"

# Verify in IEx
iex> System.get_env("GEMINI_API_KEY")
"your-key-here"
```

**Permanent Fix:**
Add to your shell profile (`~/.bashrc`, `~/.zshrc`):
```bash
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue 2: Rate Limit Exceeded

**Symptom:**
```
** (RuntimeError) Rate limit exceeded for model: openai_gpt4o_mini
Status: 429
```

**Solution:**
```elixir
# Reduce concurrency
config :research_harness,
  rate_limit: 10  # Reduce from default 50

# Or in experiment
timeout 15_000  # Increase timeout
rate_limit 10   # Reduce rate limit
```

**Alternative:**
```elixir
# Add delays between requests
defmodule MyExperiment do
  use ResearchHarness.Experiment

  # ... other config ...

  def my_condition(query) do
    Process.sleep(100)  # 100ms delay
    call_model(:gpt4o_mini, query)
  end
end
```

### Issue 3: Timeout Errors

**Symptom:**
```
** (Task.TimeoutError) task timed out after 5000ms
```

**Solution:**
```elixir
# Increase timeout in experiment
timeout 30_000  # 30 seconds instead of default 10

# Or in individual call
Ensemble.predict(query,
  models: [:gpt4o],
  timeout: 15_000  # 15 seconds per model
)
```

### Issue 4: Out of Memory

**Symptom:**
```
** (System.OutOfMemoryError) beam process limit reached
```

**Solution:**
```elixir
# Reduce concurrency
config :research_harness,
  max_concurrent_queries: 25  # Default is 50

# Or run sequentially
dataset_config %{
  sample_size: 10  # Start small
}

# Process in batches
defmodule BatchedExperiment do
  def run_in_batches(queries, batch_size \\ 10) do
    queries
    |> Enum.chunk_every(batch_size)
    |> Enum.flat_map(fn batch ->
      Enum.map(batch, &process_query/1)
    end)
  end
end
```

### Issue 5: Dataset Download Fails

**Symptom:**
```
** (RuntimeError) Failed to download dataset: mmlu_stem
Connection timeout
```

**Solution:**
```bash
# Manually download dataset
mkdir -p ~/.cache/crucible_framework/datasets
cd ~/.cache/crucible_framework/datasets

# Download from HuggingFace or source
wget https://huggingface.co/datasets/mmlu/...

# Or use custom dataset
{:ok, dataset} = DatasetManager.load(:custom,
  path: "/path/to/your/dataset.jsonl"
)
```

### Issue 6: Compilation Errors

**Symptom:**
```
** (CompileError) lib/my_module.ex:10: undefined function my_func/1
```

**Solution:**
```bash
# Clean and recompile
mix clean
mix deps.clean --all
mix deps.get
mix compile

# Check syntax
mix format --check-formatted

# Run specific file
elixir path/to/file.exs
```

### Issue 7: Test Failures

**Symptom:**
```
1) test my_test (MyModuleTest)
   Expected true, got false
```

**Solution:**
```bash
# Run single test
mix test test/my_module_test.exs:10

# Run with verbose output
mix test --trace

# Check for async issues
# In test file:
use ExUnit.Case, async: false  # Disable async
```

### Issue 8: Statistical Test Selection Wrong

**Symptom:**
```
Warning: Data is not normally distributed, but t-test was used
```

**Solution:**
```elixir
# Let Bench auto-select test
result = Bench.compare(control, treatment)
# Bench will choose Mann-Whitney U if not normal

# Or explicitly use non-parametric
result = Bench.compare(control, treatment, test: :mann_whitney_u)

# Check assumptions first
{:ok, normality} = Bench.test_normality(data)
if normality.p_value < 0.05 do
  # Use non-parametric test
end
```

### Issue 9: Checkpoint Recovery Fails

**Symptom:**
```
** (RuntimeError) Checkpoint file corrupted: exp_abc123
```

**Solution:**
```bash
# Check checkpoint directory
ls -la checkpoints/exp_abc123/

# Try earlier checkpoint
ResearchHarness.resume("exp_abc123", checkpoint: "checkpoint_001.json")

# Or start fresh
ResearchHarness.run(MyExperiment, force: true)
```

### Issue 10: Cost Overrun

**Symptom:**
```
Warning: Estimated cost $125.00 exceeds budget $100.00
```

**Solution:**
```elixir
# Set strict budget
cost_budget %{
  max_total: 100.00,
  max_per_condition: 50.00,
  abort_on_exceed: true  # Stop if exceeded
}

# Or use cheaper models
models: [:gemini_flash, :gpt4o_mini]  # Instead of GPT-4

# Or reduce sample size
dataset_config %{sample_size: 100}  # Instead of 1000

# Or use sequential execution
execution: :sequential  # Stops at consensus, saves cost
```

---

## Configuration Reference

### Complete Configuration Example

```elixir
# config/config.exs
import Config

# =============================================================================
# Ensemble Configuration
# =============================================================================
config :ensemble,
  # Default models for ensemble predictions
  # Options: :gemini_flash, :gemini_pro, :openai_gpt35_turbo, :openai_gpt4o_mini,
  #          :openai_gpt4o, :anthropic_haiku, :anthropic_sonnet, :anthropic_opus
  default_models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],

  # Default voting strategy
  # Options: :majority, :weighted, :best_confidence, :unanimous
  default_voting_strategy: :majority,

  # Default execution strategy
  # Options: :parallel, :sequential, :hedged, :cascade
  default_execution_strategy: :parallel,

  # API timeout (milliseconds)
  default_timeout: 5_000,

  # Rate limiting (requests per second per model)
  rate_limit: 50,

  # Retry configuration
  max_retries: 3,
  retry_delay: 1_000,  # milliseconds

  # Response normalization
  # Options: :none, :lowercase_trim, :boolean, :numeric
  default_normalization: :lowercase_trim

# =============================================================================
# Hedging Configuration
# =============================================================================
config :hedging,
  # Default hedging strategy
  # Options: :fixed, :percentile, :adaptive, :workload_aware
  default_strategy: :percentile,

  # For :percentile strategy
  default_percentile: 95,

  # For :fixed strategy
  fixed_delay_ms: 100,

  # Enable request cancellation when hedge wins
  enable_cancellation: true,

  # History window size for percentile calculation
  history_window: 1000,

  # Multi-level hedging
  enable_multi_level: false,
  multi_level_delays: [100, 500, 1000]  # milliseconds

# =============================================================================
# Dataset Manager Configuration
# =============================================================================
config :dataset_manager,
  # Cache directory for downloaded datasets
  cache_dir: Path.expand("~/.cache/crucible_framework/datasets"),

  # Dataset versions (for reproducibility)
  dataset_versions: %{
    mmlu: "1.0.0",
    mmlu_stem: "1.0.0",
    human_eval: "1.0.0",
    gsm8k: "1.0.0",
    custom: "latest"
  },

  # Auto-download datasets
  auto_download: true,

  # Download timeout (milliseconds)
  download_timeout: 300_000,  # 5 minutes

  # Supported evaluation metrics
  available_metrics: [
    :exact_match,
    :f1_score,
    :precision,
    :recall,
    :bleu,
    :code_bleu,
    :rouge_l
  ]

# =============================================================================
# Statistical Testing (Bench) Configuration
# =============================================================================
config :bench,
  # Default significance level
  alpha: 0.05,

  # Default confidence interval level
  confidence_level: 0.95,

  # Automatic test selection
  auto_select_test: true,

  # Check assumptions before testing
  check_assumptions: true,

  # Multiple testing correction method
  # Options: :none, :bonferroni, :holm, :fdr (Benjamini-Hochberg)
  multiple_testing_correction: :bonferroni,

  # Effect size measures to calculate
  effect_size_measures: [:cohens_d, :eta_squared, :omega_squared],

  # Bootstrap iterations for confidence intervals
  bootstrap_iterations: 10_000,

  # Power analysis configuration
  power_analysis: %{
    enable: true,
    desired_power: 0.80,
    min_detectable_effect: 0.2
  }

# =============================================================================
# Telemetry Research Configuration
# =============================================================================
config :telemetry_research,
  # Storage backend
  # Options: :ets (in-memory), :postgres (persistent)
  storage_backend: :ets,

  # For PostgreSQL backend
  database_url: System.get_env("DATABASE_URL"),

  # Event buffer size (events buffered before write)
  buffer_size: 100,

  # Flush interval (milliseconds)
  flush_interval: 5_000,

  # Export formats
  # Options: :csv, :jsonl, :parquet
  export_formats: [:csv, :jsonl],

  # Metrics calculation
  calculate_metrics: true,
  metrics_aggregation_interval: 10_000,  # milliseconds

  # Event filtering (reduce storage size)
  event_filters: [
    # Only store certain events
    # {event_pattern, :include | :exclude}
    {[:ensemble, :predict, :*], :include},
    {[:hedging, :request, :*], :include}
  ]

# =============================================================================
# Research Harness Configuration
# =============================================================================
config :research_harness,
  # Results directory
  results_dir: "./results",

  # Checkpoint directory
  checkpoint_dir: "./checkpoints",

  # Checkpoint frequency (every N queries)
  checkpoint_interval: 50,

  # Auto-resume from checkpoint on crash
  auto_resume: true,

  # Confirm costs before running
  confirm_costs: true,

  # Maximum concurrent queries
  max_concurrent_queries: 50,

  # Rate limiting (global)
  global_rate_limit: 100,  # queries per second

  # Report generation
  default_report_formats: [:markdown, :html],
  available_report_formats: [:markdown, :html, :latex, :jupyter],

  # Progress reporting
  progress_update_interval: 1_000,  # milliseconds
  show_progress_bar: true,

  # Cost budgets (default, can be overridden per experiment)
  default_cost_budget: %{
    max_total: 1000.00,
    max_per_condition: 500.00,
    warn_threshold: 0.8,  # Warn at 80% of budget
    abort_on_exceed: false
  },

  # Latency budgets
  default_latency_budget: %{
    p95_max: 5000,  # milliseconds
    p99_max: 10_000
  },

  # Statistical analysis defaults
  statistical_analysis: %{
    significance_level: 0.05,
    confidence_interval: 0.95,
    effect_size_measure: :cohens_d,
    multiple_testing_correction: :bonferroni,
    power_analysis: true
  }

# =============================================================================
# Causal Trace Configuration
# =============================================================================
config :causal_trace,
  # Output directory
  output_dir: "./research/output/causal_traces",

  # Visualization settings
  enable_html_visualization: true,
  html_template: "default",

  # Storage format
  storage_format: :json,

  # Event types to capture
  capture_events: [
    :task_decomposed,
    :hypothesis_formed,
    :pattern_applied,
    :alternative_considered,
    :constraint_identified,
    :assumption_made,
    :decision_made,
    :spec_referenced,
    :uncertainty_noted,
    :synthesis
  ],

  # Search indexing
  enable_search: true,
  index_fields: [:type, :decision, :confidence, :timestamp]

# =============================================================================
# Logging Configuration
# =============================================================================
config :logger,
  # Log level
  # Options: :debug, :info, :warning, :error
  level: :info,

  # Log format
  format: "$time [$level] $metadata$message\n",

  # Metadata to include in logs
  metadata: [:experiment_id, :condition, :model, :query_id],

  # Truncate long messages
  truncate: 8096,

  # Console backend
  backends: [:console]

# =============================================================================
# Environment-Specific Configuration
# =============================================================================

# Development
if Mix.env() == :dev do
  config :logger, level: :debug

  config :research_harness,
    confirm_costs: false  # Don't ask for confirmation in dev
end

# Test
if Mix.env() == :test do
  config :logger, level: :warning

  config :ensemble,
    default_timeout: 1_000  # Faster timeouts in tests

  config :research_harness,
    checkpoint_interval: 10,  # Checkpoint more frequently in tests
    confirm_costs: false

  config :telemetry_research,
    storage_backend: :ets,  # Always use in-memory in tests
    buffer_size: 10
end

# Production
if Mix.env() == :prod do
  config :logger, level: :info

  config :telemetry_research,
    storage_backend: :postgres  # Use persistent storage in prod

  config :research_harness,
    confirm_costs: true,
    abort_on_exceed: true  # Strictly enforce budgets in prod
end
```

### Environment Variables Reference

```bash
# =============================================================================
# API Keys (Required)
# =============================================================================
# Google Gemini API
export GEMINI_API_KEY="your-gemini-api-key"

# OpenAI API
export OPENAI_API_KEY="sk-your-openai-api-key"

# Anthropic API
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"

# =============================================================================
# Database (Optional - for PostgreSQL telemetry backend)
# =============================================================================
export DATABASE_URL="postgresql://user:password@localhost:5432/crucible_framework_dev"

# =============================================================================
# Experiment Configuration (Optional)
# =============================================================================
# Override random seed
export EXPERIMENT_SEED=42

# Override results directory
export RESULTS_DIR="/path/to/results"

# Override checkpoint directory
export CHECKPOINT_DIR="/path/to/checkpoints"

# =============================================================================
# Logging (Optional)
# =============================================================================
# Log level: debug, info, warning, error
export LOG_LEVEL=info

# =============================================================================
# Debugging (Optional)
# =============================================================================
# Enable verbose telemetry
export VERBOSE_TELEMETRY=true

# Enable HTTP request logging
export LOG_HTTP_REQUESTS=true

# Disable rate limiting (for testing)
export DISABLE_RATE_LIMITING=false
```

---

## Next Steps

### Immediate Next Steps (Week 1)

1. **Run Example Experiments**
   ```bash
   # Try the built-in examples
   cd apps/ensemble
   iex -S mix
   Code.require_file("examples/basic_usage.exs")
   ```

2. **Modify an Example**
   - Copy `examples/basic_usage.exs` to `my_experiment.exs`
   - Change the models or voting strategy
   - Run it and compare results

3. **Create Your Own Experiment**
   - Use the patterns from this guide
   - Start with a small sample (10-50 queries)
   - Gradually increase scale

4. **Explore the Reports**
   - Open generated HTML reports
   - Understand each section
   - Look for interesting patterns

### Short-Term Goals (Weeks 2-4)

1. **Master Core Libraries**
   - See `crucible_ensemble` package for ensemble strategies
   - See `crucible_bench` package for statistical testing
   - Experiment with different configurations

2. **Run Replication Studies**
   - Replicate published LLM reliability studies
   - Verify your framework produces consistent results
   - Practice statistical analysis

3. **Design Custom Experiments**
   - Identify your research questions
   - Design experiments to test them
   - Run pilot studies

4. **Contribute Back**
   - Report bugs or issues on GitHub
   - Suggest improvements
   - Share your findings with the community

### Medium-Term Goals (Months 2-3)

1. **Publish Your Research**
   - Use generated LaTeX reports in papers
   - Cite the framework appropriately
   - Share preprints on arXiv

2. **Extend the Framework**
   - Add custom voting strategies
   - Implement new statistical tests
   - Create custom report formats

3. **Scale Up Experiments**
   - Move from 100 to 1,000+ queries
   - Use distributed execution
   - Optimize costs

4. **Build Tooling**
   - Create experiment templates
   - Automate common workflows
   - Build visualization dashboards

### Long-Term Goals (Months 4+)

1. **Become a Framework Expert**
   - Understand internals deeply
   - Contribute code improvements
   - Help other users

2. **Publish Major Research**
   - Multi-paper research program
   - Novel reliability techniques
   - Open-source datasets

3. **Build on the Framework**
   - Create domain-specific extensions
   - Integrate with other tools
   - Develop new applications

---

## Additional Resources

### Official Documentation

- **README.md** - Project overview and features
- **ARCHITECTURE.md** - System design and internals
- **RESEARCH_METHODOLOGY.md** - The 6 hypotheses and experimental designs

### Library-Specific Guides

- **INSTRUMENTATION.md** - Telemetry for complete observability
- **DATASETS.md** - Working with benchmark datasets

### Related Packages

The following functionality now lives in separate packages:

- **Ensemble voting** - See `crucible_ensemble` package
- **Request hedging** - See `crucible_hedging` package
- **Statistical testing** - See `crucible_bench` package
- **Causal transparency** - See `crucible_trace` package

### Contributing and Publishing

- **CONTRIBUTING.md** - How to contribute to the framework
- **FAQ.md** - Frequently asked questions

### External Resources

**Elixir Learning:**
- [Elixir Official Docs](https://elixir-lang.org/docs.html)
- [Elixir School](https://elixirschool.com/)
- [Exercism Elixir Track](https://exercism.org/tracks/elixir)

**Statistical Methods:**
- Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.)
- Cumming, G. (2014). The new statistics: Why and how. Psychological science, 25(1), 7-29.

**Ensemble Methods:**
- Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.
- Dietterich, T. G. (2000). Ensemble methods in machine learning.

**Distributed Systems:**
- Dean, J., & Barroso, L. A. (2013). The tail at scale. Communications of the ACM, 56(2), 74-80.

**LLM Evaluation:**
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. ICLR.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.

### Community and Support

**GitHub:**
- Issues: https://github.com/North-Shore-AI/crucible_framework/issues
- Discussions: https://github.com/North-Shore-AI/crucible_framework/discussions

**Documentation:**
- HexDocs: https://hexdocs.pm/crucible_framework (when published)
- Examples: https://github.com/North-Shore-AI/crucible_framework/tree/main/examples

**Contact:**
- Please open an issue or discussion for support/feature requests.

---

## Conclusion

Congratulations! You now have everything you need to start conducting rigorous LLM reliability research with the Crucible Framework.

**Key Takeaways:**

1. **Start Small**: Begin with simple experiments (10-100 queries)
2. **Check Estimates**: Always run cost/time estimates before full runs
3. **Understand Results**: Take time to read and interpret statistical reports
4. **Iterate Rapidly**: Use the framework to test hypotheses quickly
5. **Be Rigorous**: Let the framework handle statistical validity
6. **Share Findings**: Publish results and contribute back to the community

**Remember:**

- The framework handles complexity (concurrency, statistics, telemetry)
- You focus on research questions and experimental design
- Start with examples, then customize for your needs
- Don't hesitate to ask for help (GitHub Discussions)

**Your Journey:**

```
Week 1: Run examples, understand basics
Week 2-4: Create custom experiments, master core concepts
Month 2-3: Publish research, extend framework
Month 4+: Become expert, contribute major features
```

**Now go forth and conduct excellent science!**

---

**Document Version**: 1.0
**Last Updated**: 2025-10-08
**Next Review**: 2025-11-08
**Feedback**: Please open an issue on GitHub or email research@example.com

**Happy Researching!**
