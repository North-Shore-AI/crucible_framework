# Frequently Asked Questions (FAQ)

This comprehensive FAQ covers common questions about installation, configuration, usage, troubleshooting, research design, and publication.

**Table of Contents**

- [Installation & Setup](#installation--setup)
- [Configuration & API Keys](#configuration--api-keys)
- [Performance & Optimization](#performance--optimization)
- [Statistical Testing](#statistical-testing)
- [Ensemble Usage](#ensemble-usage)
- [Hedging Usage](#hedging-usage)
- [Dataset Management](#dataset-management)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [Research Design](#research-design)
- [Reproducibility](#reproducibility)
- [Publication & Citation](#publication--citation)
- [Cost Management](#cost-management)
- [Architecture & Design](#architecture--design)
- [Contributing](#contributing)

---

## Installation & Setup

### Q1: What are the minimum system requirements?

**A:** Minimum requirements:
- **Elixir:** 1.14 or higher
- **Erlang/OTP:** 25 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 1GB for framework + datasets
- **OS:** Linux, macOS, or Windows (via WSL2)

**Recommended for production research:**
- **RAM:** 16GB+ for large-scale experiments
- **CPU:** 4+ cores for parallel processing
- **SSD:** For faster dataset loading

### Q2: How do I install Elixir on my system?

**A:** Installation by platform:

**macOS:**
```bash
brew install elixir
```

**Ubuntu/Debian:**
```bash
# Add Erlang Solutions repo
wget https://packages.erlang-solutions.com/erlang-solutions_2.0_all.deb
sudo dpkg -i erlang-solutions_2.0_all.deb
sudo apt-get update

# Install Elixir
sudo apt-get install elixir
```

**Windows:**
Use WSL2 (Windows Subsystem for Linux) and follow Ubuntu instructions, or:
```powershell
choco install elixir
```

**Verify installation:**
```bash
elixir --version
# Should show: Elixir 1.14+ and Erlang/OTP 25+
```

### Q3: Installation fails with dependency errors. What should I do?

**A:** Try these steps in order:

```bash
# 1. Clean existing dependencies
mix deps.clean --all

# 2. Remove build artifacts
rm -rf _build

# 3. Remove dependencies
rm -rf deps

# 4. Get fresh dependencies
mix deps.get

# 5. Compile
mix compile
```

If still failing:
```bash
# Check Elixir and Erlang versions
elixir --version

# Update Hex (package manager)
mix local.hex --force

# Update Rebar (build tool)
mix local.rebar --force

# Try again
mix deps.get && mix compile
```

### Q4: How do I verify the installation was successful?

**A:** Run the test suite:

```bash
# Run all tests
mix test

# If all tests pass, you'll see:
# Finished in X.X seconds
# XX tests, 0 failures
```

Also verify each library loads:
```bash
# Start IEx (Interactive Elixir)
iex -S mix

# Try loading libraries
iex> Ensemble.version()
"0.1.0"

iex> Hedging.version()
"0.1.0"

iex> Bench.version()
"0.1.0"
```

### Q5: Can I use this framework without installing PostgreSQL?

**A:** Yes! PostgreSQL is optional. The framework uses in-memory storage (ETS) by default.

**Using ETS (default):**
```elixir
# config/config.exs
config :telemetry_research,
  storage_backend: :ets  # Default
```

**Using PostgreSQL (for persistent storage):**
```elixir
config :telemetry_research,
  storage_backend: :postgres,
  postgres_config: [
    hostname: "localhost",
    username: "postgres",
    password: "postgres",
    database: "telemetry_research_dev"
  ]
```

---

## Configuration & API Keys

### Q6: How do I set up API keys securely?

**A:** Never hardcode API keys. Use environment variables:

**Method 1: Export in shell**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

**Method 2: .env file (recommended)**
```bash
# Create .env file (add to .gitignore!)
echo ".env" >> .gitignore

# Add keys to .env
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF

# Load in shell
source .env
```

**Method 3: Config file (reads from environment)**
```elixir
# config/config.exs
import Config

config :ensemble,
  openai_api_key: System.get_env("OPENAI_API_KEY"),
  anthropic_api_key: System.get_env("ANTHROPIC_API_KEY"),
  google_api_key: System.get_env("GOOGLE_API_KEY")
```

### Q7: I'm getting "API key not found" errors. What's wrong?

**A:** Check these in order:

```bash
# 1. Verify environment variable is set
echo $OPENAI_API_KEY
# Should print your key, not empty

# 2. Check key is loaded in IEx
iex -S mix
iex> System.get_env("OPENAI_API_KEY")
# Should return your key

# 3. Verify config reads it
iex> Application.get_env(:ensemble, :openai_api_key)
# Should return your key
```

If still not working:
- Restart your shell after setting environment variables
- Check for typos in environment variable names
- Ensure .env is sourced before running `mix` commands

### Q8: How do I configure different API keys for testing vs production?

**A:** Use environment-specific configuration:

```elixir
# config/dev.exs
import Config

config :ensemble,
  openai_api_key: System.get_env("OPENAI_API_KEY_DEV")

# config/prod.exs
import Config

config :ensemble,
  openai_api_key: System.get_env("OPENAI_API_KEY_PROD")

# config/test.exs
import Config

config :ensemble,
  openai_api_key: "test-key-not-used"  # Tests use mocks
```

Run with specific environment:
```bash
MIX_ENV=dev mix run experiments/my_experiment.exs
MIX_ENV=prod mix run experiments/my_experiment.exs
```

### Q9: Can I use custom model endpoints (e.g., Azure OpenAI)?

**A:** Yes, configure custom endpoints:

```elixir
# config/config.exs
config :ensemble,
  openai_config: [
    api_key: System.get_env("AZURE_OPENAI_KEY"),
    base_url: "https://your-resource.openai.azure.com",
    api_version: "2024-02-15-preview"
  ]
```

---

## Performance & Optimization

### Q10: How can I speed up my experiments?

**A:** Several optimization strategies:

**1. Increase parallelism:**
```elixir
# Run more queries concurrently
{:ok, results} = ResearchHarness.run(MyExperiment,
  max_concurrency: 50  # Default: 10
)
```

**2. Use streaming for large datasets:**
```elixir
dataset
|> Stream.filter(&valid?/1)
|> Stream.map(&process/1)
|> Enum.take(100)
```

**3. Enable result caching:**
```elixir
config :research_harness,
  cache_results: true,
  cache_dir: "./cache"
```

**4. Use cheaper/faster models for development:**
```elixir
# Development
models: [:gpt4_mini, :claude_haiku, :gemini_flash]

# Production
models: [:gpt4, :claude_opus, :gemini_pro]
```

**5. Sample smaller datasets during development:**
```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 50  # Small for testing
)

# Production: full dataset or larger sample
```

### Q11: My experiments are using too much memory. How do I reduce it?

**A:** Memory optimization techniques:

**1. Use streaming instead of loading all data:**
```elixir
# Bad - loads everything into memory
queries = DatasetManager.load_all(:mmlu_stem)

# Good - streams data
queries =
  DatasetManager.stream(:mmlu_stem)
  |> Stream.take(1000)
```

**2. Process in batches:**
```elixir
queries
|> Stream.chunk_every(100)
|> Stream.each(fn batch ->
  process_batch(batch)
  # Garbage collection happens between batches
end)
|> Stream.run()
```

**3. Configure smaller ETS tables:**
```elixir
config :telemetry_research,
  max_events: 10_000  # Limit stored events
```

**4. Clean up processes:**
```elixir
# Stop experiments when done
{:ok, exp} = TelemetryResearch.start_experiment(...)
# ... do work ...
TelemetryResearch.stop_experiment(exp.id)
TelemetryResearch.cleanup_experiment(exp.id)  # Free memory
```

### Q12: How can I monitor experiment progress?

**A:** Several monitoring options:

**1. Built-in progress tracking:**
```elixir
ResearchHarness.run(MyExperiment,
  show_progress: true  # Shows progress bar
)
```

**2. Custom progress callbacks:**
```elixir
ResearchHarness.run(MyExperiment,
  on_progress: fn completed, total ->
    IO.puts("Progress: #{completed}/#{total}")
  end
)
```

**3. Telemetry events:**
```elixir
:telemetry.attach(
  "my-handler",
  [:research_harness, :query, :complete],
  fn _event, measurements, _metadata, _config ->
    IO.inspect(measurements)
  end,
  nil
)
```

**4. Check experiment status:**
```elixir
{:ok, status} = ResearchHarness.status(experiment_id)
IO.inspect(status)
# %{completed: 150, total: 200, progress: 0.75}
```

---

## Statistical Testing

### Q13: Which statistical test should I use?

**A:** Test selection guide:

**Comparing two groups:**

| Data Type | Independent | Paired | Recommendation |
|-----------|-------------|---------|----------------|
| Normal, equal variance | Yes | No | Independent t-test |
| Normal, unequal variance | Yes | No | Welch's t-test |
| Non-normal | Yes | No | Mann-Whitney U test |
| Any | No | Yes | Paired t-test (if normal) |
| Any | No | Yes | Wilcoxon signed-rank (if non-normal) |

**Comparing 3+ groups:**

| Data Type | Independent | Recommendation |
|-----------|-------------|----------------|
| Normal, equal variance | Yes | One-way ANOVA |
| Normal, unequal variance | Yes | Welch's ANOVA |
| Non-normal | Yes | Kruskal-Wallis test |

**Let the framework choose:**
```elixir
# Automatic test selection based on data properties
result = Bench.auto_compare(control, treatment)
# Checks assumptions and selects appropriate test
```

### Q14: How do I interpret p-values and effect sizes?

**A:** Interpretation guidelines:

**P-values:**
- **p < 0.001:** Very strong evidence against null hypothesis
- **p < 0.01:** Strong evidence
- **p < 0.05:** Moderate evidence (conventional threshold)
- **p ≥ 0.05:** Insufficient evidence to reject null

**Important:** Small p-values don't mean large effects!

**Cohen's d (effect size):**
- **d < 0.2:** Negligible effect
- **d ≥ 0.2:** Small effect
- **d ≥ 0.5:** Medium effect
- **d ≥ 0.8:** Large effect
- **d ≥ 1.2:** Very large effect

**Example interpretation:**
```elixir
result = Bench.compare(control, treatment)

# result.p_value = 0.001, result.effect_size.cohens_d = 0.3

# Interpretation:
# "We found a statistically significant difference (p < 0.001)
#  with a small effect size (d = 0.3)."
```

**Report both p-value AND effect size!**

### Q15: What sample size do I need?

**A:** Use power analysis:

```elixir
# A priori power analysis (before experiment)
{:ok, n} = Bench.power_analysis(
  effect_size: 0.5,  # Expected Cohen's d
  alpha: 0.05,       # Significance level
  power: 0.80        # Desired power (80%)
)

IO.puts("Need n ≥ #{n} per group")
```

**Common rules of thumb:**

| Effect Size | Samples per Group (α=0.05, power=0.80) |
|-------------|---------------------------------------|
| Small (d=0.2) | ~400 |
| Medium (d=0.5) | ~64 |
| Large (d=0.8) | ~26 |

**Post-hoc power analysis (after experiment):**
```elixir
{:ok, power} = Bench.achieved_power(
  n: 100,
  effect_size: 0.5,
  alpha: 0.05
)

# If power < 0.80, you may have missed real effects (Type II error)
```

### Q16: How do I handle multiple comparisons?

**A:** Apply corrections to control false discovery rate:

```elixir
# Multiple t-tests without correction (WRONG!)
p_values = [0.03, 0.04, 0.02]  # All < 0.05

# Apply Bonferroni correction
adjusted = Bench.correct_multiple_comparisons(p_values, method: :bonferroni)
# [0.09, 0.12, 0.06]  # None significant at 0.05!

# Apply Benjamini-Hochberg (less conservative)
adjusted = Bench.correct_multiple_comparisons(p_values, method: :bh)
```

**Correction methods:**

- **Bonferroni:** Most conservative, controls family-wise error rate
- **Holm:** Less conservative than Bonferroni, still controls FWER
- **Benjamini-Hochberg:** Controls false discovery rate (FDR), less conservative
- **Benjamini-Yekutieli:** For dependent tests

**When comparing multiple conditions:**
```elixir
conditions = [control, treatment1, treatment2, treatment3]

# Automatic correction
{:ok, result} = Bench.compare_multiple(conditions,
  correction: :bonferroni
)
```

---

## Ensemble Usage

### Q17: Which voting strategy should I use?

**A:** Strategy selection guide:

**Majority (`:majority`):**
- **Use when:** All models equally reliable
- **Best for:** Classification, multiple choice
- **Example:** "What is the capital of France?"
```elixir
Ensemble.predict(query, strategy: :majority)
```

**Weighted (`:weighted`):**
- **Use when:** Models have different reliability
- **Best for:** Models provide confidence scores
- **Example:** Weighing GPT-4 (90% confidence) more than GPT-3.5 (60%)
```elixir
Ensemble.predict(query,
  strategy: :weighted,
  weights: %{gpt4: 2.0, gpt3: 1.0}  # GPT-4 counts 2×
)
```

**Best Confidence (`:best_confidence`):**
- **Use when:** Trust high confidence predictions
- **Best for:** When one model is much more confident
- **Example:** Take GPT-4's answer at 95% confidence over others at 70%
```elixir
Ensemble.predict(query, strategy: :best_confidence)
```

**Unanimous (`:unanimous`):**
- **Use when:** Maximum reliability required
- **Best for:** High-stakes decisions
- **Example:** Medical diagnosis, legal reasoning
```elixir
Ensemble.predict(query, strategy: :unanimous)
# Returns error if models disagree
```

### Q18: How many models should I use in my ensemble?

**A:** Trade-off between reliability and cost:

| Models | Accuracy | Cost | Latency | Recommendation |
|--------|----------|------|---------|----------------|
| 1 | 89% | 1× | 1× | Baseline only |
| 3 | 94-96% | 3× | ~1× (parallel) | Good balance |
| 5 | 96-98% | 5× | ~1× (parallel) | High reliability |
| 7+ | 98%+ | 7×+ | ~1× (parallel) | Diminishing returns |

**Recommendations:**

- **Development/testing:** 3 models
- **Production (moderate stakes):** 3-5 models
- **High stakes:** 5-7 models
- **Maximum reliability:** 7+ models with unanimous voting

**Empirical testing:**
```elixir
# Test different ensemble sizes
for n <- [1, 3, 5, 7] do
  models = Enum.take(all_models, n)
  result = Ensemble.predict(query, models: models)
  # Compare accuracy vs cost
end
```

### Q19: How do I add a new model to the ensemble?

**A:** Configure model clients:

```elixir
# Option 1: Use built-in models
models = [:gpt4, :claude, :gemini]

# Option 2: Add custom model
config :ensemble,
  models: [
    my_custom_model: [
      module: MyApp.CustomModel,
      api_key: System.get_env("CUSTOM_API_KEY"),
      endpoint: "https://api.custom.com/v1/chat"
    ]
  ]

# Use in ensemble
Ensemble.predict(query, models: [:gpt4, :my_custom_model])
```

**Implement custom model:**
```elixir
defmodule MyApp.CustomModel do
  @behaviour Ensemble.ModelClient

  @impl true
  def query(prompt, opts) do
    # Call your custom API
    response = call_api(prompt)
    {:ok, response}
  end

  @impl true
  def parse_response(response) do
    # Extract answer from response
    {:ok, answer}
  end
end
```

### Q20: Ensemble queries are too expensive. How can I reduce cost?

**A:** Cost optimization strategies:

**1. Sequential execution with early stopping:**
```elixir
Ensemble.predict(query,
  execution: :sequential,
  stop_on_consensus: true  # Stop when 3 models agree
)
```

**2. Cascade strategy (cheap first, expensive if needed):**
```elixir
Ensemble.predict(query,
  execution: :cascade,
  cascade_order: [:gpt4_mini, :claude_haiku, :gpt4]  # Cheap → expensive
)
```

**3. Use cheaper models:**
```elixir
# Expensive (~$0.01/query)
models: [:gpt4, :claude_opus, :gemini_pro]

# Cheaper (~$0.001/query)
models: [:gpt4_mini, :claude_haiku, :gemini_flash]
```

**4. Adaptive ensemble sizing:**
```elixir
# Use 3 models for easy queries, 5 for hard ones
Ensemble.predict(query,
  adaptive: true,
  min_models: 3,
  max_models: 5
)
```

---

## Hedging Usage

### Q21: When should I use request hedging?

**A:** Use hedging when:

✅ **Good use cases:**
- Tail latency is a problem (P99 >> P50)
- Cost increase of 10-15% is acceptable
- Users sensitive to slow requests
- SLA requires strict latency bounds

❌ **Poor use cases:**
- Latency already consistent (P99 ≈ P50)
- Cost is primary constraint
- API has strict rate limits
- Batch processing (not latency-sensitive)

**Test if hedging helps:**
```elixir
# Measure baseline latencies
baseline_latencies = measure_latencies(no_hedging: true)

# Measure with hedging
hedged_latencies = measure_latencies(hedging: :p95)

# Compare P99
baseline_p99 = Bench.percentile(baseline_latencies, 0.99)
hedged_p99 = Bench.percentile(hedged_latencies, 0.99)

improvement = (baseline_p99 - hedged_p99) / baseline_p99
IO.puts("P99 improvement: #{improvement * 100}%")
```

### Q22: Which hedging strategy should I choose?

**A:** Strategy comparison:

**Fixed Delay (`:fixed`):**
```elixir
Hedging.request(fn -> call_api() end,
  strategy: :fixed,
  delay_ms: 1000  # Hedge after 1 second
)
```
- **Use when:** Simple, predictable latency
- **Pros:** Easy to tune
- **Cons:** Not adaptive to changing conditions

**Percentile (`:percentile`):**
```elixir
Hedging.request(fn -> call_api() end,
  strategy: :percentile,
  percentile: 95  # Hedge at P95 latency
)
```
- **Use when:** Latency varies, want data-driven threshold
- **Pros:** Adapts to actual latency distribution
- **Cons:** Needs warmup period

**Adaptive (`:adaptive`):**
```elixir
Hedging.request(fn -> call_api() end,
  strategy: :adaptive  # Learns optimal delay
)
```
- **Use when:** Latency changes over time
- **Pros:** Self-tuning
- **Cons:** More complex

**Recommendation:** Start with `:percentile` at P95.

### Q23: Hedging increased my costs too much. How do I tune it?

**A:** Tune hedging to balance latency vs cost:

**1. Increase percentile threshold:**
```elixir
# More aggressive (fires often, high cost)
percentile: 90  # Hedge after 90th percentile

# Balanced (recommended)
percentile: 95

# Conservative (fires rarely, low cost)
percentile: 99
```

**2. Disable cancellation:**
```elixir
Hedging.request(fn -> call_api() end,
  enable_cancellation: false  # Pay for both requests if both complete
)
```

**3. Set cost budget:**
```elixir
Hedging.request(fn -> call_api() end,
  max_cost_increase: 0.10  # Max 10% cost increase
)
```

**4. Monitor hedge fire rate:**
```elixir
stats = Hedging.statistics()
IO.puts("Hedge fire rate: #{stats.hedge_fire_rate * 100}%")

# Target: 5-10% fire rate for good balance
```

---

## Dataset Management

### Q24: How do I add a custom dataset?

**A:** Step-by-step process:

**1. Create dataset module:**
```elixir
# apps/dataset_manager/lib/dataset_manager/datasets/my_dataset.ex
defmodule DatasetManager.Datasets.MyDataset do
  @behaviour DatasetManager.Loader

  @impl true
  def load(opts) do
    path = Keyword.get(opts, :path, "data/my_dataset.jsonl")

    items =
      path
      |> File.stream!()
      |> Stream.map(&Jason.decode!/1)
      |> Stream.map(&parse_item/1)
      |> Enum.to_list()

    {:ok, %DatasetManager.Dataset{
      name: :my_dataset,
      items: items,
      metadata: %{version: "1.0.0"}
    }}
  end

  defp parse_item(json) do
    %DatasetManager.Item{
      id: json["id"],
      input: json["query"],
      expected: json["answer"],
      metadata: json["metadata"] || %{}
    }
  end
end
```

**2. Register dataset:**
```elixir
# apps/dataset_manager/lib/dataset_manager.ex
@datasets %{
  mmlu_stem: DatasetManager.Datasets.MMLU,
  humaneval: DatasetManager.Datasets.HumanEval,
  my_dataset: DatasetManager.Datasets.MyDataset  # Add here
}
```

**3. Add tests:**
```elixir
# test/dataset_manager/datasets/my_dataset_test.exs
defmodule DatasetManager.Datasets.MyDatasetTest do
  use ExUnit.Case

  test "loads dataset successfully" do
    {:ok, dataset} = DatasetManager.load(:my_dataset)
    assert length(dataset.items) > 0
  end
end
```

**4. Use dataset:**
```elixir
{:ok, dataset} = DatasetManager.load(:my_dataset)
```

### Q25: How do I sample from a large dataset?

**A:** Several sampling strategies:

**Random sampling:**
```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 100,
  sampling: :random,
  seed: 42  # For reproducibility
)
```

**Stratified sampling:**
```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 100,
  sampling: :stratified,
  stratify_by: :difficulty  # Balance by difficulty
)
```

**K-fold cross-validation:**
```elixir
{:ok, folds} = DatasetManager.load(:mmlu_stem,
  sampling: :kfold,
  k: 5  # 5-fold CV
)

for {train, test} <- folds do
  # Train on train, evaluate on test
end
```

**Sequential (first N items):**
```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 100,
  sampling: :sequential
)
```

### Q26: Dataset loading is slow. How can I speed it up?

**A:** Optimization techniques:

**1. Enable caching:**
```elixir
config :dataset_manager,
  cache_dir: "~/.cache/crucible_framework/datasets",
  cache_enabled: true  # Default
```

**2. Use streaming for large datasets:**
```elixir
# Don't load everything at once
DatasetManager.stream(:mmlu_stem)
|> Stream.take(1000)
|> Enum.each(&process/1)
```

**3. Preload datasets:**
```elixir
# Preload during application startup
defmodule MyApp.Application do
  def start(_type, _args) do
    Task.start(fn ->
      DatasetManager.preload([:mmlu_stem, :humaneval])
    end)

    # ... rest of startup
  end
end
```

**4. Use binary format:**
```elixir
# Export to binary for faster loading
DatasetManager.export(:mmlu_stem, format: :etf, path: "mmlu.etf")

# Load binary (much faster than JSON)
DatasetManager.load_binary("mmlu.etf")
```

---

## Debugging & Troubleshooting

### Q27: My experiment crashes with "timeout" errors. What should I do?

**A:** Increase timeout settings:

```elixir
# Increase query timeout
Ensemble.predict(query,
  timeout: 60_000  # 60 seconds (default: 30s)
)

# Increase experiment timeout
ResearchHarness.run(MyExperiment,
  query_timeout: 60_000,
  experiment_timeout: :infinity  # No overall timeout
)
```

**Debug slow queries:**
```elixir
# Add timing
{time_us, result} = :timer.tc(fn ->
  Ensemble.predict(query)
end)

IO.puts("Query took #{time_us / 1000}ms")

# Identify slow models
Ensemble.predict(query,
  on_model_complete: fn model, time_ms ->
    IO.puts("#{model}: #{time_ms}ms")
  end
)
```

### Q28: How do I enable debug logging?

**A:** Configure logger level:

```elixir
# config/config.exs
import Config

config :logger,
  level: :debug  # :debug, :info, :warn, :error

# Per-application logging
config :logger, :console,
  format: "[$level] $message\n",
  metadata: [:application, :module]
```

**Runtime logging:**
```elixir
# In IEx
Logger.configure(level: :debug)

# Add debug statements
require Logger
Logger.debug("Processing query: #{inspect(query)}")
```

**Telemetry debugging:**
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
    IO.inspect({event, measurements, metadata})
  end,
  nil
)
```

### Q29: Tests fail with "connection refused" errors. What's wrong?

**A:** You're likely using real API calls in tests. Use mocks instead:

```elixir
# test/test_helper.exs

# Define mocks
Mox.defmock(Ensemble.LLMClientMock, for: Ensemble.LLMClient.Behaviour)

# Configure app to use mocks in tests
Application.put_env(:ensemble, :llm_client, Ensemble.LLMClientMock)
```

**In tests:**
```elixir
defmodule EnsembleTest do
  use ExUnit.Case, async: true
  import Mox

  setup :verify_on_exit!

  test "predicts correctly" do
    # Mock API calls
    expect(Ensemble.LLMClientMock, :query, fn _model, _prompt ->
      {:ok, "Answer"}
    end)

    {:ok, result} = Ensemble.predict("Question?")
    assert result.answer == "Answer"
  end
end
```

### Q30: How do I debug a failing statistical test?

**A:** Diagnostic steps:

**1. Check assumptions:**
```elixir
data = [1, 2, 3, 4, 5]

# Check normality
{:ok, normality} = Bench.test_normality(data)
IO.inspect(normality)  # %{is_normal: true/false, test: :shapiro_wilk, p_value: ...}

# Check variance homogeneity
{:ok, variance} = Bench.test_homogeneity([group1, group2])
IO.inspect(variance)
```

**2. Visualize data:**
```elixir
# Summary statistics
Bench.describe(data)
# %{mean: 3.0, median: 3, sd: 1.58, min: 1, max: 5}

# Distribution shape
Bench.histogram(data)
# Prints ASCII histogram
```

**3. Try robust alternatives:**
```elixir
# If normality violated, use non-parametric test
result = Bench.compare(control, treatment,
  test: :mann_whitney  # Instead of t-test
)
```

**4. Check for outliers:**
```elixir
outliers = Bench.detect_outliers(data, method: :iqr)
IO.puts("Outliers: #{inspect(outliers)}")

# Remove outliers if justified
clean_data = Bench.remove_outliers(data)
```

---

## Research Design

### Q31: How do I design a randomized controlled trial (RCT)?

**A:** RCT template:

```elixir
defmodule MyRCT do
  use ResearchHarness.Experiment

  # Research question
  name "Does ensemble improve accuracy?"

  # Hypothesis
  hypothesis """
  H1: 5-model ensemble accuracy > single model accuracy
  """

  # Independent variable (manipulated)
  conditions [
    %{name: "control", fn: &single_model/1},     # IV: single model
    %{name: "treatment", fn: &ensemble_5/1}       # IV: 5-model ensemble
  ]

  # Dependent variable (measured)
  metrics [:accuracy]

  # Sampling
  dataset :mmlu_stem, sample_size: 200

  # Replication
  repeat 3  # Accounts for stochastic variation

  # Randomization
  seed 42  # Reproducible randomization

  # Implementation
  def single_model(query) do
    {:ok, result} = call_gpt4(query)
    %{prediction: result.answer}
  end

  def ensemble_5(query) do
    {:ok, result} = Ensemble.predict(query, models: 5)
    %{prediction: result.answer}
  end
end
```

**Key elements:**
1. ✅ Clear hypothesis
2. ✅ Manipulated IV (condition)
3. ✅ Measured DV (accuracy)
4. ✅ Random assignment (seed)
5. ✅ Replication (repeat)
6. ✅ Control group (baseline)

### Q32: How many repetitions should my experiment have?

**A:** Depends on variability:

**High variability (e.g., creative tasks):**
```elixir
repeat 5  # or more
```

**Medium variability (e.g., MMLU questions):**
```elixir
repeat 3
```

**Low variability (deterministic tasks):**
```elixir
repeat 1  # May be sufficient if deterministic
```

**Empirical approach:**
```elixir
# Run experiment with increasing repetitions
for n_reps <- [1, 2, 3, 4, 5] do
  results = run_with_repetitions(n_reps)
  cv = coefficient_of_variation(results)
  IO.puts("#{n_reps} reps: CV = #{cv}")

  # Stop when CV < 0.1 (10% variation)
  if cv < 0.1, do: break
end
```

### Q33: Should I use a within-subjects or between-subjects design?

**A:** Trade-offs:

**Within-subjects (same queries in all conditions):**
```elixir
# Same queries for all conditions
dataset :mmlu_stem, sample_size: 100
conditions [
  %{name: "baseline", fn: &baseline/1},
  %{name: "treatment", fn: &treatment/1}
]
# Each query tested in both conditions
```

**Pros:**
- More statistical power (paired data)
- Controls for query difficulty
- Fewer total queries needed

**Cons:**
- Order effects possible
- Can't use if conditions interact

**Between-subjects (different queries per condition):**
```elixir
# Different queries for each condition
dataset :mmlu_stem, sample_size: 200  # Split 100/100
between_subjects: true
```

**Pros:**
- No order effects
- Conditions independent

**Cons:**
- Less power
- More queries needed
- Queries may differ in difficulty

**Recommendation:** Use within-subjects unless conditions could interfere (e.g., learning effects).

### Q34: How do I control for confounding variables?

**A:** Several strategies:

**1. Randomization:**
```elixir
# Randomize query order
seed 42  # Fixed seed for reproducibility

# ResearchHarness automatically randomizes
```

**2. Counterbalancing:**
```elixir
# Half participants see baseline first, half see treatment first
counterbalance :condition_order
```

**3. Blocking:**
```elixir
# Control for known confound (e.g., difficulty)
block_by :difficulty  # Easy, medium, hard blocks
```

**4. Stratification:**
```elixir
# Ensure equal difficulty distribution
dataset :mmlu_stem,
  sampling: :stratified,
  stratify_by: :difficulty
```

**5. Covariate measurement:**
```elixir
# Measure and control for confounds statistically
metrics [
  :accuracy,
  :query_length,      # Potential confound
  :query_difficulty   # Potential confound
]

# Use ANCOVA to control for covariates
Bench.ancova(accuracy ~ condition + query_length + query_difficulty)
```

---

## Reproducibility

### Q35: How do I ensure my results are reproducible?

**A:** Reproducibility checklist:

**1. Fix random seeds:**
```elixir
seed 42  # All randomization uses this seed
```

**2. Record exact versions:**
```elixir
# Automatically recorded in experiment metadata
# - Framework version
# - Elixir version
# - Model versions (e.g., gpt-4-0613)
```

**3. Save complete configuration:**
```elixir
ResearchHarness.run(MyExperiment,
  save_config: true  # Saves full configuration
)
```

**4. Archive artifacts:**
```elixir
# All artifacts saved:
# - config.json (configuration)
# - dataset.jsonl (exact queries used)
# - results.csv (raw results)
# - analysis.json (statistical analysis)
# - environment.json (system info)
```

**5. Use version control:**
```bash
git add experiments/my_experiment.exs
git commit -m "Add reproducible experiment"
git tag v1.0.0
```

**6. Document hardware:**
```elixir
# Automatically captured in environment.json
# - CPU, memory, OS
# - Elixir/Erlang versions
```

### Q36: How do I reproduce someone else's experiment?

**A:** Reproduction steps:

```bash
# 1. Clone repository
git clone https://github.com/username/experiment-repo
cd experiment-repo

# 2. Checkout exact version
git checkout v1.0.0

# 3. Install dependencies
mix deps.get

# 4. Verify environment matches
cat environment.json
# Check your system matches

# 5. Set environment variables
export OPENAI_API_KEY="sk-..."

# 6. Run experiment
mix run experiments/experiment.exs

# 7. Compare results
diff results/original results/reproduction
```

**If results differ:**
- Check model versions match
- Verify random seeds
- Compare configurations
- Check for API changes
- Contact original authors

---

## Publication & Citation

### Q37: How should I cite the framework in my paper?

**A:** Citation format (see PUBLICATIONS.md for details):

**In text:**
```latex
We used CrucibleFramework \cite{crucible_framework2025}
for experiment orchestration and statistical analysis.
```

**In references:**
```bibtex
@software{crucible_framework2025,
  title = {CrucibleFramework: Infrastructure for LLM Reliability Research},
  author = {{North Shore AI}},
  year = {2025},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.3}
}
```

**If using specific library extensively, also cite:**
```bibtex
@software{crucible_framework_component2025,
  title = {CrucibleFramework Component Library},
  author = {{North Shore AI}},
  year = {2025},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.3},
  note = {Reference the specific module you relied on}
}
```

### Q38: Should I share my code and data?

**A:** **Yes!** Open science benefits everyone:

**Benefits:**
- Enables replication
- Increases impact
- Improves credibility
- Helps community

**What to share:**
```
my-experiment/
├── README.md               # Reproduction instructions
├── experiments/
│   └── my_experiment.exs   # Experiment code
├── data/
│   ├── dataset.jsonl       # Dataset (if allowed)
│   └── results.csv         # Raw results
├── config/
│   └── config.exs          # Configuration
└── analysis/
    └── analyze.exs         # Analysis scripts
```

**Where to share:**
- **Code:** GitHub
- **Data:** Zenodo, OSF, Hugging Face
- **Paper:** arXiv preprint

**License:**
- Code: MIT
- Data: CC-BY (with attribution)

---

## Cost Management

### Q39: How much will my experiment cost?

**A:** Estimate before running:

```elixir
# Get cost estimate
{:ok, estimate} = ResearchHarness.estimate_cost(MyExperiment)

IO.puts("""
Estimated cost: $#{estimate.total_cost}
- Per query: $#{estimate.per_query_cost}
- Total queries: #{estimate.total_queries}
""")

# Approve before running
if estimate.total_cost < 100.00 do
  ResearchHarness.run(MyExperiment)
else
  IO.puts("Cost too high! Consider smaller sample.")
end
```

**Cost calculation:**
```
Total Cost = (Queries × Repetitions × Models) × Cost per Query

Example:
- 200 queries
- 3 repetitions
- 5 models
- $0.002 per query

Total = (200 × 3 × 5) × $0.002 = $6.00
```

### Q40: How can I reduce experiment costs?

**A:** Cost reduction strategies:

**1. Use cheaper models:**
```elixir
# Expensive: $0.01/query
models: [:gpt4, :claude_opus, :gemini_pro]

# Cheap: $0.0005/query
models: [:gpt4_mini, :claude_haiku, :gemini_flash]

# Cost reduction: 95%
```

**2. Smaller sample for pilot:**
```elixir
# Pilot with small sample
dataset :mmlu_stem, sample_size: 50  # $1.50
# If promising, run full experiment
dataset :mmlu_stem, sample_size: 200  # $6.00
```

**3. Sequential execution with early stopping:**
```elixir
Ensemble.predict(query,
  execution: :sequential,
  stop_on_consensus: true
)
# Average 3.2 models instead of 5
```

**4. Cache results:**
```elixir
config :research_harness,
  cache_results: true
# Don't re-run identical queries
```

**5. Fewer repetitions:**
```elixir
# Development
repeat 1

# Final experiment
repeat 3
```

### Q41: How do I track spending in real-time?

**A:** Enable cost tracking:

```elixir
# In experiment
ResearchHarness.run(MyExperiment,
  track_cost: true,
  cost_limit: 50.00,  # Stop if exceeds $50
  on_cost_update: fn current_cost ->
    IO.puts("Current cost: $#{current_cost}")
  end
)

# Check cost after completion
{:ok, report} = ResearchHarness.get_report(exp_id)
IO.puts("Total cost: $#{report.total_cost}")
```

**Cost breakdown by component:**
```elixir
{:ok, breakdown} = ResearchHarness.cost_breakdown(exp_id)

IO.inspect(breakdown)
# %{
#   gpt4: 4.50,
#   claude: 3.20,
#   gemini: 2.30,
#   total: 10.00
# }
```

---

## Architecture & Design

### Q42: Why use Elixir instead of Python?

**A:** Elixir advantages for research:

**1. Concurrency:**
```elixir
# Elixir: 10,000 concurrent requests easily
queries
|> Task.async_stream(&call_api/1, max_concurrency: 1000)
|> Enum.to_list()

# Python: Limited by GIL, need asyncio/multiprocessing
```

**2. Fault tolerance:**
```elixir
# Elixir: Supervisor restarts failed processes automatically
# Python: Manual error handling, experiment stops on error
```

**3. Immutability:**
```elixir
# Elixir: No shared state, no race conditions
# Python: Mutable state causes subtle bugs
```

**4. Distribution:**
```elixir
# Elixir: Scale across machines with no code changes
# Python: Need Celery, Ray, or similar
```

**When to use Python instead:**
- Heavy numerical computing (NumPy/SciPy)
- Deep learning training
- Existing Python ML pipelines
- Team only knows Python

**Best of both:**
```elixir
# Call Python from Elixir for ML tasks
Porcelain.exec("python", ["analyze.py", "--data", data])
```

### Q43: How does the framework scale?

**A:** Scaling characteristics:

**Vertical scaling (single machine):**
- **Queries:** Up to 10,000 concurrent
- **Memory:** ~2KB per query process
- **Limited by:** CPU cores, API rate limits

**Horizontal scaling (multiple machines):**
```elixir
# Distribute across nodes
Node.connect(:"node2@hostname")

# Spawn queries on all nodes
results =
  queries
  |> Task.async_stream(&call_api/1,
    max_concurrency: 1000,
    node: :random  # Distribute across nodes
  )
  |> Enum.to_list()
```

**Benchmarks:**

| Configuration | Queries/sec | Max Concurrent |
|---------------|-------------|----------------|
| 1 core | 10 | 100 |
| 4 cores | 40 | 500 |
| 16 cores | 150 | 2000 |
| 4 nodes × 16 cores | 600 | 8000 |

**Bottleneck:** Usually API rate limits, not the framework.

---

## Contributing

### Q44: How can I contribute to the framework?

**A:** Many ways to contribute:

**1. Code contributions:**
- New voting strategies
- Additional statistical tests
- Performance optimizations
- Bug fixes

**2. Documentation:**
- Improve guides
- Add examples
- Fix typos
- Write tutorials

**3. Research:**
- Share experimental results
- Validate hypotheses
- Contribute datasets

**4. Community:**
- Answer questions
- Review PRs
- Report bugs

See CONTRIBUTING.md for detailed guidelines.

### Q45: I found a bug. How do I report it?

**A:** Bug report template:

```markdown
## Bug Description

[Clear description of bug]

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

[What should happen]

## Actual Behavior

[What actually happens]

## Environment

- Framework version: 0.1.3
- Elixir version: 1.17.0
- Erlang version: 26.2.1
- OS: Ubuntu 22.04

## Code Sample

Minimal example to reproduce the issue:

```elixir
result = Ensemble.predict("test")
```

## Error Message

```
** (RuntimeError) error message here
    stacktrace here
```

## Additional Context

[Any other relevant information]
```

### Q46: How do I request a new feature?

**A:** Feature request template:

```markdown
## Feature Description

[Clear description of desired feature]

## Use Case

[Why do you need this feature?]

## Proposed API

```elixir
# How you envision using it
Ensemble.predict(query,
  new_option: value
)
```

## Alternatives Considered

[Other approaches you've considered]

## Additional Context

[Relevant papers, examples from other tools, etc.]
```

---

## Additional Questions

### Q47: Is this framework production-ready?

**A:** Depends on component:

**Production-ready:**
- ✅ Ensemble library (stable API, well-tested)
- ✅ Hedging library (proven algorithms)
- ✅ Bench library (validated against R/Python)

**Research-focused:**
- ⚠️ ResearchHarness (designed for experiments, not production)
- ⚠️ TelemetryResearch (overhead acceptable for research)

**Recommendation:**
- **Research:** Use entire framework
- **Production:** Extract Ensemble/Hedging libraries, add production monitoring

### Q48: Can I use this framework commercially?

**A:** Yes! MIT license allows commercial use.

**Requirements:**
- Include MIT license text
- Attribute the framework

**You can:**
- Use in commercial products
- Modify the code
- Keep modifications private
- Sell products using the framework

**Best practices:**
- Contribute improvements back (optional but appreciated)
- Cite in research papers
- Follow community guidelines

### Q49: How often is the framework updated?

**A:** Release schedule:

- **Patch releases:** Monthly (bug fixes)
- **Minor releases:** Quarterly (new features)
- **Major releases:** Annually (breaking changes)

**Stay updated:**
```bash
# Watch repository on GitHub
# Subscribe to releases

# Check for updates
mix hex.outdated
```

### Q50: Where can I get help?

**A:** Start with the self-serve resources that ship with CrucibleFramework:
- README.md and GETTING_STARTED.md for setup guidance
- Component guides (Ensemble, Hedging, Statistical Testing, etc.) for domain specifics
- CHANGELOG.md for behavioural changes and migration notes
- CONTRIBUTING.md for the reporting process and code standards
- This FAQ for operational and research guidance

**Before escalating an issue:**
1. Re-run the failing command with verbose output enabled
2. Confirm dependencies match the documented versions
3. Build a minimal reproduction script or test case
4. Capture logs, stack traces, and key configuration values
5. Document expected vs. actual behaviour, including metrics if relevant

---

## Quick Reference

### Common Commands

```bash
# Installation
mix deps.get
mix compile

# Testing
mix test
mix test --cover

# Running experiments
mix run experiments/my_experiment.exs

# Documentation
mix docs
open doc/index.html

# Formatting
mix format

# Type checking (if dialyxir installed)
mix dialyzer
```

### Common Patterns

```elixir
# Basic ensemble
{:ok, result} = Ensemble.predict(query,
  models: [:gpt4, :claude, :gemini],
  strategy: :majority
)

# Statistical comparison
result = Bench.compare(control, treatment)

# Load dataset
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 100
)

# Run experiment
ResearchHarness.run(MyExperiment,
  output_dir: "results"
)
```

---

**Last Updated:** 2025-11-21
**Version:** 0.1.3
**Maintainers:** North Shore AI

---

Built with ❤️ by researchers, for researchers.
