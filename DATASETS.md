# Dataset Management Guide

**Comprehensive guide to dataset loading, evaluation, and management in the Elixir AI Research framework.**

Version: 1.0
Last Updated: 2025-10-08

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Supported Datasets](#supported-datasets)
4. [Loading Datasets](#loading-datasets)
5. [Dataset Caching](#dataset-caching)
6. [Sampling Strategies](#sampling-strategies)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Batch Evaluation](#batch-evaluation)
9. [Custom Datasets](#custom-datasets)
10. [Data Preprocessing](#data-preprocessing)
11. [Advanced Usage](#advanced-usage)
12. [Performance Optimization](#performance-optimization)
13. [Research Best Practices](#research-best-practices)
14. [API Reference](#api-reference)

---

## Introduction

The DatasetManager library provides a unified interface for working with AI evaluation benchmarks in Elixir. It handles dataset loading, caching, evaluation, and sampling with a focus on reproducibility and research rigor.

### Key Features

- **Unified API**: Single interface for multiple benchmarks (MMLU, HumanEval, GSM8K)
- **Automatic Caching**: Intelligent caching with version tracking
- **Multiple Metrics**: Exact match, F1 score, Pass@k, and custom metrics
- **Sampling Strategies**: Random, stratified, k-fold cross-validation
- **Reproducibility**: Seeded random operations for consistent results
- **Extensibility**: Easy integration of custom datasets

### Why Dataset Management Matters

Proper dataset management is critical for AI research:

1. **Reproducibility**: Consistent dataset versions ensure experiments can be replicated
2. **Efficiency**: Caching prevents redundant downloads and parsing
3. **Comparability**: Standardized evaluation enables fair model comparisons
4. **Statistical Validity**: Proper sampling ensures representative evaluation

### Research Background

The importance of standardized benchmarks has been extensively documented:

- **Hendrycks et al. (2021)**: "Measuring Massive Multitask Language Understanding" - Introduced MMLU and highlighted the need for diverse evaluation
- **Chen et al. (2021)**: "Evaluating Large Language Models Trained on Code" - Established HumanEval as the standard code generation benchmark
- **Cobbe et al. (2021)**: "Training Verifiers to Solve Math Word Problems" - Demonstrated the importance of math reasoning evaluation with GSM8K

---

## Core Concepts

### Dataset Structure

Every dataset in the system follows a consistent structure:

```elixir
%DatasetManager.Dataset{
  name: "mmlu_stem",
  version: "1.0",
  items: [
    %{
      id: "mmlu_stem_0",
      input: %{
        question: "What is the capital of France?",
        choices: ["London", "Paris", "Berlin", "Madrid"]
      },
      expected: 1,
      metadata: %{
        subject: "geography",
        difficulty: "easy"
      }
    },
    # ... more items
  ],
  metadata: %{
    source: "huggingface:cais/mmlu",
    license: "MIT",
    domain: "STEM",
    subjects: ["mathematics", "physics", "biology", ...]
  }
}
```

### Dataset Item Fields

- **id**: Unique identifier for the item (used for result tracking)
- **input**: The question/prompt (structure varies by dataset)
- **expected**: Ground truth answer
- **metadata**: Additional information (subject, difficulty, etc.)

### Prediction Format

When evaluating model outputs, use this prediction format:

```elixir
%{
  id: "mmlu_stem_0",           # Must match dataset item ID
  predicted: 1,                # Model's prediction
  metadata: %{                 # Optional: model-specific metadata
    confidence: 0.95,
    reasoning_time_ms: 1234
  }
}
```

---

## Supported Datasets

### MMLU (Massive Multitask Language Understanding)

**Description**: 57-subject multiple-choice benchmark spanning STEM, humanities, social sciences, and more.

**Dataset Name**: `:mmlu` (all subjects) or `:mmlu_stem` (STEM only)

**Structure**:
```elixir
input: %{
  question: String.t(),
  choices: [String.t(), String.t(), String.t(), String.t()]
}
expected: integer()  # 0-3 (choice index)
```

**Subjects** (STEM subset):
- Mathematics: abstract_algebra, college_mathematics, elementary_mathematics
- Physics: college_physics, high_school_physics, conceptual_physics
- Biology: anatomy, college_biology, high_school_biology
- Chemistry: college_chemistry, high_school_chemistry
- Computer Science: college_computer_science, computer_security, machine_learning
- Engineering: electrical_engineering

**Example Usage**:
```elixir
# Load full MMLU
{:ok, dataset} = DatasetManager.load(:mmlu)

# Load STEM subjects only
{:ok, stem_dataset} = DatasetManager.load(:mmlu_stem, sample_size: 500)

# Load specific sample
{:ok, sample} = DatasetManager.load(:mmlu_stem)
{:ok, small_sample} = DatasetManager.random_sample(sample, size: 100, seed: 42)
```

**Evaluation Metrics**:
- Exact Match: Primary metric (correct choice index)
- Subject-level accuracy: Break down by subject area
- Difficulty-level accuracy: Performance by difficulty tier

**Citation**:
```
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

### HumanEval

**Description**: 164 Python programming problems with function signatures, docstrings, and test cases.

**Dataset Name**: `:humaneval`

**Structure**:
```elixir
input: %{
  signature: String.t(),      # Function signature with docstring
  tests: String.t(),          # Unit tests
  entry_point: String.t(),    # Function name
  description: String.t()     # Problem description
}
expected: String.t()          # Canonical solution
```

**Example Usage**:
```elixir
# Load HumanEval dataset
{:ok, humaneval} = DatasetManager.load(:humaneval)

# Evaluate code generation
predictions = Enum.map(humaneval.items, fn item ->
  code = MyCodeGenerator.generate(item.input.signature)

  %{
    id: item.id,
    predicted: code,
    metadata: %{
      generation_time_ms: ...,
      model: "gpt-4"
    }
  }
end)

{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: humaneval,
  metrics: [:pass_at_k],
  model_name: "gpt-4"
)
```

**Evaluation Metrics**:
- **Pass@k**: Probability that at least one of k generated samples passes tests
  - Pass@1: Standard metric
  - Pass@10: Generate 10 solutions, success if any passes
  - Pass@100: Used for estimating ceiling performance

**Pass@k Calculation**:
```elixir
# Pass@k is calculated using the formula:
# Pass@k = 1 - (C(n-c, k) / C(n, k))
# where n = total samples, c = correct samples

defmodule PassAtK do
  def compute(results, k) do
    grouped_by_problem = Enum.group_by(results, & &1.problem_id)

    Enum.map(grouped_by_problem, fn {_problem_id, samples} ->
      n = length(samples)
      c = Enum.count(samples, & &1.passed)

      if n >= k do
        1.0 - (combinations(n - c, k) / combinations(n, k))
      else
        if c > 0, do: 1.0, else: 0.0
      end
    end)
    |> average()
  end

  defp combinations(n, k) when k > n, do: 0
  defp combinations(n, k), do: factorial(n) / (factorial(k) * factorial(n - k))

  defp factorial(0), do: 1
  defp factorial(n), do: n * factorial(n - 1)

  defp average(list), do: Enum.sum(list) / length(list)
end
```

**Citation**:
```
@article{chen2021evaluating,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}
```

### GSM8K (Grade School Math 8K)

**Description**: 8,500 grade school math word problems requiring multi-step reasoning.

**Dataset Name**: `:gsm8k`

**Structure**:
```elixir
input: String.t()              # Math word problem
expected: %{
  answer: integer(),           # Numerical answer
  reasoning: String.t()        # Step-by-step solution
}
```

**Example Usage**:
```elixir
# Load GSM8K
{:ok, gsm8k} = DatasetManager.load(:gsm8k, sample_size: 100)

# Evaluate math reasoning
predictions = Enum.map(gsm8k.items, fn item ->
  result = MathSolver.solve(item.input)

  %{
    id: item.id,
    predicted: result.answer,
    metadata: %{
      reasoning_steps: result.steps,
      confidence: result.confidence
    }
  }
end)

{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: gsm8k,
  metrics: [:exact_match, :f1],
  model_name: "math_solver_v1"
)
```

**Evaluation Metrics**:
- Exact Match: Correct numerical answer (with tolerance)
- F1 Score: Token overlap in reasoning steps
- Step Accuracy: Correctness of intermediate steps

**Answer Extraction**:
```elixir
# GSM8K answers use the format:
# "Step 1: ...
#  Step 2: ...
#  #### 42"
# The final answer is after "####"

defmodule GSM8K.AnswerExtractor do
  def extract_answer(text) do
    text
    |> String.split("####")
    |> List.last()
    |> String.trim()
    |> String.replace(",", "")
    |> Integer.parse()
    |> case do
      {num, _} -> num
      :error -> nil
    end
  end
end
```

**Citation**:
```
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

---

## Loading Datasets

### Basic Loading

```elixir
# Load a standard dataset
{:ok, dataset} = DatasetManager.load(:mmlu_stem)

# Load with options
{:ok, dataset} = DatasetManager.load(:mmlu_stem,
  sample_size: 200,
  cache: true,
  version: "1.0"
)

# Load custom dataset
{:ok, dataset} = DatasetManager.load("my_dataset",
  source: "/path/to/data.jsonl"
)
```

### Loading Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:version` | String | "1.0" | Dataset version |
| `:subset` | String | nil | Subset name for multi-config datasets |
| `:cache` | Boolean | true | Enable caching |
| `:sample_size` | Integer | nil | Limit number of items |
| `:source` | String | nil | Custom source path |
| `:seed` | Integer | random | Random seed for sampling |

### Lazy Loading

For large datasets, use streaming:

```elixir
defmodule DatasetManager.Stream do
  @doc """
  Load dataset items as a stream for memory-efficient processing.
  """
  def load_stream(dataset_name, opts \\ []) do
    {:ok, dataset} = DatasetManager.load(dataset_name, Keyword.merge(opts, [cache: false]))

    Stream.resource(
      fn -> {dataset.items, 0} end,
      fn
        {[], _idx} -> {:halt, nil}
        {[item | rest], idx} -> {[item], {rest, idx + 1}}
      end,
      fn _ -> :ok end
    )
  end
end

# Usage
DatasetManager.Stream.load_stream(:mmlu, sample_size: 1000)
|> Stream.chunk_every(10)
|> Stream.map(fn batch ->
  evaluate_batch(batch)
end)
|> Enum.to_list()
```

### Error Handling

```elixir
case DatasetManager.load(:unknown_dataset) do
  {:ok, dataset} ->
    IO.puts("Loaded #{length(dataset.items)} items")

  {:error, {:unknown_dataset, name}} ->
    IO.puts("Unknown dataset: #{name}")

  {:error, {:file_read_error, reason}} ->
    IO.puts("Failed to read file: #{inspect(reason)}")

  {:error, reason} ->
    IO.puts("Load failed: #{inspect(reason)}")
end
```

---

## Dataset Caching

### How Caching Works

DatasetManager automatically caches parsed datasets to disk to avoid redundant downloads and parsing.

**Cache Structure**:
```
~/.elixir_ai_research/cache/
├── datasets/
│   ├── mmlu_stem_v1.0.etf
│   ├── humaneval_v1.0.etf
│   └── gsm8k_v1.0.etf
└── metadata/
    └── cache_index.json
```

### Cache Operations

```elixir
# Check cache status
cached_datasets = DatasetManager.list_cached()
IO.inspect(cached_datasets)
# => [
#   %{name: "mmlu_stem", version: "1.0", size_mb: 12.3, cached_at: ~U[...]},
#   %{name: "humaneval", version: "1.0", size_mb: 0.8, cached_at: ~U[...]}
# ]

# Invalidate specific dataset cache
:ok = DatasetManager.invalidate_cache(:mmlu_stem)

# Clear all caches
:ok = DatasetManager.clear_cache()

# Load without cache
{:ok, dataset} = DatasetManager.load(:mmlu_stem, cache: false)
```

### Cache Configuration

```elixir
# In config/config.exs
config :dataset_manager,
  cache_dir: "~/.elixir_ai_research/cache",
  cache_ttl_seconds: 86400 * 7,  # 7 days
  max_cache_size_mb: 1000,        # 1 GB
  auto_cleanup: true
```

### Manual Cache Management

```elixir
defmodule DatasetManager.Cache do
  @cache_dir Application.compile_env(:dataset_manager, :cache_dir)

  @doc "Get cache statistics"
  def statistics do
    cache_files = Path.wildcard("#{@cache_dir}/datasets/*.etf")

    total_size =
      Enum.reduce(cache_files, 0, fn file, acc ->
        %{size: size} = File.stat!(file)
        acc + size
      end)

    %{
      total_files: length(cache_files),
      total_size_mb: total_size / (1024 * 1024),
      cache_dir: @cache_dir
    }
  end

  @doc "Clean up old cache entries"
  def cleanup_old_entries(max_age_seconds) do
    now = System.system_time(:second)
    cache_files = Path.wildcard("#{@cache_dir}/datasets/*.etf")

    Enum.each(cache_files, fn file ->
      %{mtime: mtime} = File.stat!(file)
      age = now - mtime

      if age > max_age_seconds do
        File.rm!(file)
        IO.puts("Removed old cache file: #{Path.basename(file)}")
      end
    end)
  end
end
```

---

## Sampling Strategies

### Random Sampling

**Use Case**: Quick evaluation, prototyping, preliminary experiments

```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem)

# Random sample with seed for reproducibility
{:ok, sample} = DatasetManager.random_sample(dataset,
  size: 100,
  seed: 42
)

# Verify reproducibility
{:ok, sample2} = DatasetManager.random_sample(dataset,
  size: 100,
  seed: 42
)

sample.items == sample2.items  # => true
```

### Stratified Sampling

**Use Case**: Maintaining distribution of categories (e.g., subjects, difficulty levels)

```elixir
{:ok, dataset} = DatasetManager.load(:mmlu_stem)

# Sample 200 items while maintaining subject distribution
{:ok, stratified} = DatasetManager.stratified_sample(dataset,
  size: 200,
  strata_field: [:metadata, :subject]
)

# Verify distribution preservation
original_dist =
  dataset.items
  |> Enum.group_by(& &1.metadata.subject)
  |> Enum.map(fn {subject, items} ->
    {subject, length(items) / length(dataset.items)}
  end)
  |> Map.new()

sample_dist =
  stratified.items
  |> Enum.group_by(& &1.metadata.subject)
  |> Enum.map(fn {subject, items} ->
    {subject, length(items) / length(stratified.items)}
  end)
  |> Map.new()

# Distributions should be similar
```

**Stratified Sampling Algorithm**:

```elixir
defmodule StratifiedSampler do
  @doc """
  Perform stratified sampling to maintain proportions.

  1. Group items by strata field
  2. Calculate proportion for each stratum
  3. Sample from each stratum proportionally
  4. Adjust for rounding errors
  """
  def stratified_sample(items, total_size, strata_field) do
    # Group by strata
    groups = Enum.group_by(items, &get_in(&1, strata_field))
    total_items = length(items)

    # Calculate samples per stratum
    samples_per_stratum =
      groups
      |> Enum.map(fn {stratum, stratum_items} ->
        proportion = length(stratum_items) / total_items
        count = round(proportion * total_size)
        {stratum, count}
      end)
      |> Map.new()

    # Adjust if total doesn't match due to rounding
    allocated = samples_per_stratum |> Map.values() |> Enum.sum()

    samples_per_stratum =
      if allocated != total_size do
        adjust_allocation(samples_per_stratum, total_size - allocated)
      else
        samples_per_stratum
      end

    # Sample from each stratum
    groups
    |> Enum.flat_map(fn {stratum, stratum_items} ->
      n = Map.get(samples_per_stratum, stratum, 0)
      Enum.take_random(stratum_items, min(n, length(stratum_items)))
    end)
  end

  defp adjust_allocation(allocation, diff) when diff > 0 do
    # Add to largest strata first
    allocation
    |> Enum.sort_by(fn {_k, v} -> -v end)
    |> Enum.take(diff)
    |> Enum.reduce(allocation, fn {stratum, count}, acc ->
      Map.put(acc, stratum, count + 1)
    end)
  end

  defp adjust_allocation(allocation, diff) when diff < 0 do
    # Remove from largest strata first
    allocation
    |> Enum.sort_by(fn {_k, v} -> -v end)
    |> Enum.take(abs(diff))
    |> Enum.reduce(allocation, fn {stratum, count}, acc ->
      Map.put(acc, stratum, max(0, count - 1))
    end)
  end

  defp adjust_allocation(allocation, 0), do: allocation
end
```

### K-Fold Cross-Validation

**Use Case**: Model comparison, hyperparameter tuning, variance estimation

```elixir
{:ok, dataset} = DatasetManager.load(:gsm8k, sample_size: 500)

# Create 5-fold cross-validation splits
{:ok, folds} = DatasetManager.k_fold(dataset,
  k: 5,
  shuffle: true,
  seed: 42
)

# Train and evaluate on each fold
results = Enum.map(folds, fn {train, test} ->
  model = train_model(train)
  evaluate_model(model, test)
end)

# Calculate average performance and variance
avg_accuracy = Enum.map(results, & &1.accuracy) |> average()
std_accuracy = Enum.map(results, & &1.accuracy) |> std_dev()

IO.puts("Accuracy: #{avg_accuracy} ± #{std_accuracy}")
```

**K-Fold Implementation Details**:

```elixir
defmodule KFold do
  @doc """
  Create k-fold cross-validation splits.

  Each fold uses 1/k of data for testing and (k-1)/k for training.
  """
  def create_folds(items, k, opts \\ []) do
    shuffle = Keyword.get(opts, :shuffle, true)
    seed = Keyword.get(opts, :seed, :rand.uniform(1_000_000))

    # Optionally shuffle
    items = if shuffle do
      :rand.seed(:exsss, {seed, seed, seed})
      Enum.shuffle(items)
    else
      items
    end

    fold_size = div(length(items), k)

    # Create each fold
    0..(k-1)
    |> Enum.map(fn i ->
      # Test set for this fold
      test_start = i * fold_size
      test_end = min((i + 1) * fold_size, length(items))
      test_items = Enum.slice(items, test_start, test_end - test_start)

      # Training set is everything else
      train_items =
        Enum.take(items, test_start) ++
        Enum.drop(items, test_end)

      {train_items, test_items}
    end)
  end
end

# Example: Compare two models with k-fold CV
defmodule ModelComparison do
  def compare_models(dataset, model_a, model_b, k: 5) do
    {:ok, folds} = DatasetManager.k_fold(dataset, k: k, seed: 12345)

    results_a = Enum.map(folds, fn {train, test} ->
      evaluate_fold(model_a, train, test)
    end)

    results_b = Enum.map(folds, fn {train, test} ->
      evaluate_fold(model_b, train, test)
    end)

    # Statistical comparison
    {mean_a, std_a} = calculate_stats(results_a)
    {mean_b, std_b} = calculate_stats(results_b)

    # Paired t-test
    p_value = paired_t_test(results_a, results_b)

    %{
      model_a: %{mean: mean_a, std: std_a},
      model_b: %{mean: mean_b, std: std_b},
      significant: p_value < 0.05,
      p_value: p_value
    }
  end

  defp evaluate_fold(model, train, test) do
    # Train model on training fold
    trained = model.train(train)

    # Evaluate on test fold
    predictions = Enum.map(test.items, &trained.predict/1)
    {:ok, results} = DatasetManager.evaluate(predictions, dataset: test)

    results.accuracy
  end

  defp calculate_stats(values) do
    mean = Enum.sum(values) / length(values)
    variance = Enum.map(values, &(:math.pow(&1 - mean, 2))) |> Enum.sum() |> Kernel./(length(values))
    std = :math.sqrt(variance)
    {mean, std}
  end

  defp paired_t_test(sample_a, sample_b) do
    # Implementation of paired t-test
    # Returns p-value
    differences = Enum.zip(sample_a, sample_b) |> Enum.map(fn {a, b} -> a - b end)

    mean_diff = Enum.sum(differences) / length(differences)
    std_diff = :math.sqrt(
      Enum.map(differences, &(:math.pow(&1 - mean_diff, 2)))
      |> Enum.sum()
      |> Kernel./(length(differences) - 1)
    )

    t_statistic = mean_diff / (std_diff / :math.sqrt(length(differences)))
    df = length(differences) - 1

    # Calculate p-value from t-distribution
    Statistex.Distributions.T.cdf(abs(t_statistic), df) |> then(&(2 * (1 - &1)))
  end
end
```

### Train-Test Split

**Use Case**: Final model evaluation, holdout testing

```elixir
{:ok, dataset} = DatasetManager.load(:humaneval)

# 80/20 train/test split
{:ok, {train, test}} = DatasetManager.train_test_split(dataset,
  test_size: 0.2,
  shuffle: true,
  seed: 42
)

IO.puts("Train size: #{length(train.items)}")
IO.puts("Test size: #{length(test.items)}")

# Train on training set
model = train_model(train)

# Evaluate on held-out test set
predictions = generate_predictions(model, test)
{:ok, results} = DatasetManager.evaluate(predictions, dataset: test)
```

---

## Evaluation Metrics

### Exact Match

**Description**: Binary metric indicating if prediction exactly matches expected answer.

**Use Cases**:
- Multiple choice questions (MMLU)
- Numerical answers (GSM8K)
- Classification tasks

**Implementation**:

```elixir
defmodule ExactMatch do
  @moduledoc """
  Exact match with normalization for different data types.

  Handles:
  - String comparison (case-insensitive, normalized)
  - Numerical comparison (with tolerance for floating point)
  - Multiple choice (index comparison)
  - List/set comparison (order-independent)
  """

  # String comparison
  def compute(predicted, expected)
      when is_binary(predicted) and is_binary(expected) do
    normalize_string(predicted) == normalize_string(expected)
    |> if(do: 1.0, else: 0.0)
  end

  # Numerical comparison with tolerance
  def compute(predicted, expected)
      when is_number(predicted) and is_number(expected) do
    tolerance = abs(expected) * 0.01  # 1% tolerance
    abs(predicted - expected) <= tolerance
    |> if(do: 1.0, else: 0.0)
  end

  # Multiple choice (integer indices)
  def compute(predicted, expected)
      when is_integer(predicted) and is_integer(expected) do
    if predicted == expected, do: 1.0, else: 0.0
  end

  # List/set comparison (order-independent)
  def compute(predicted, expected)
      when is_list(predicted) and is_list(expected) do
    if MapSet.new(predicted) == MapSet.new(expected), do: 1.0, else: 0.0
  end

  # Normalize strings for comparison
  defp normalize_string(str) do
    str
    |> String.downcase()
    |> String.trim()
    |> String.replace(~r/\s+/, " ")          # Normalize whitespace
    |> String.replace(~r/[^\w\s]/, "")       # Remove punctuation
  end
end
```

**Example**:

```elixir
# MMLU evaluation
predictions = [
  %{id: "mmlu_0", predicted: 1, metadata: %{}},  # Predicted choice B
  %{id: "mmlu_1", predicted: 0, metadata: %{}},  # Predicted choice A
  %{id: "mmlu_2", predicted: 2, metadata: %{}}   # Predicted choice C
]

{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: mmlu_dataset,
  metrics: [:exact_match]
)

IO.inspect(results.metrics.exact_match)  # => 0.67 (2 out of 3 correct)
```

### F1 Score

**Description**: Harmonic mean of precision and recall, measuring token-level overlap.

**Use Cases**:
- Text generation with partial credit
- Answer extraction
- Reasoning step evaluation

**Formula**:

```
Precision = |predicted ∩ expected| / |predicted|
Recall = |predicted ∩ expected| / |expected|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Implementation**:

```elixir
defmodule F1Score do
  @doc """
  Compute token-level F1 score between predicted and expected text.
  """
  def compute(predicted, expected) when is_binary(predicted) and is_binary(expected) do
    predicted_tokens = tokenize(predicted)
    expected_tokens = tokenize(expected)

    # Token overlap
    common = MapSet.intersection(predicted_tokens, expected_tokens)
    common_count = MapSet.size(common)

    if common_count == 0 do
      0.0
    else
      precision = common_count / MapSet.size(predicted_tokens)
      recall = common_count / MapSet.size(expected_tokens)

      2 * (precision * recall) / (precision + recall)
    end
  end

  # Tokenize text into normalized tokens
  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, " ")
    |> String.split(~r/\s+/, trim: true)
    |> MapSet.new()
  end
end
```

**Example**:

```elixir
# GSM8K evaluation with F1 for reasoning steps
predictions = [
  %{
    id: "gsm8k_0",
    predicted: %{
      answer: 42,
      reasoning: "First multiply 6 by 7 to get 42"
    },
    metadata: %{}
  }
]

{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: gsm8k_dataset,
  metrics: [:exact_match, :f1]
)

# Exact match on numerical answer
IO.inspect(results.metrics.exact_match)  # => 1.0

# F1 on reasoning text
IO.inspect(results.metrics.f1)  # => 0.85 (token overlap)
```

### Custom Metrics

**Use Case**: Domain-specific evaluation, specialized scoring

```elixir
# Define custom metric function
custom_metric = fn predicted, expected ->
  # Example: Length-normalized edit distance
  distance = String.jaro_distance(predicted, expected)
  distance
end

# Use in evaluation
{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: dataset,
  metrics: [:exact_match, custom_metric]
)
```

**Advanced Custom Metric Example**:

```elixir
defmodule CodeQualityMetric do
  @doc """
  Custom metric for code generation quality.

  Evaluates:
  - Syntax correctness (0.4 weight)
  - Test passage (0.4 weight)
  - Code style (0.2 weight)
  """
  def compute(predicted_code, expected_code) do
    syntax_score = check_syntax(predicted_code)
    test_score = run_tests(predicted_code)
    style_score = check_style(predicted_code, expected_code)

    0.4 * syntax_score + 0.4 * test_score + 0.2 * style_score
  end

  defp check_syntax(code) do
    case Code.string_to_quoted(code) do
      {:ok, _ast} -> 1.0
      {:error, _} -> 0.0
    end
  end

  defp run_tests(code) do
    # Run test suite against generated code
    # Return proportion of passing tests
    case TestRunner.run(code) do
      {:ok, results} -> results.pass_rate
      {:error, _} -> 0.0
    end
  end

  defp check_style(predicted, expected) do
    # Compare code style (indentation, naming, etc.)
    StyleChecker.similarity(predicted, expected)
  end
end

# Use in evaluation
{:ok, results} = DatasetManager.evaluate(predictions,
  dataset: humaneval,
  metrics: [:exact_match, &CodeQualityMetric.compute/2]
)
```

### Pass@k (Code Generation)

**Description**: Probability that at least one of k generated samples passes all tests.

**Use Case**: HumanEval and code generation benchmarks

**Formula**:

```
Pass@k = E[1 - C(n-c, k) / C(n, k)]

where:
- n = number of samples per problem
- c = number of correct samples
- C(n, k) = binomial coefficient "n choose k"
```

**Implementation**:

```elixir
defmodule PassAtK do
  @moduledoc """
  Pass@k metric for code generation evaluation.

  Estimates the probability that at least one of k generated
  solutions passes all test cases.
  """

  @doc """
  Compute Pass@k metric.

  ## Arguments
  - results: List of evaluation results grouped by problem
  - k: Number of samples to consider

  ## Returns
  Pass@k score (0.0 to 1.0)
  """
  def compute(results, k) do
    # Group results by problem ID
    grouped = Enum.group_by(results, & &1.problem_id)

    # Calculate Pass@k for each problem
    pass_at_k_per_problem =
      Enum.map(grouped, fn {_problem_id, samples} ->
        n = length(samples)
        c = Enum.count(samples, & &1.all_tests_passed)

        compute_for_problem(n, c, k)
      end)

    # Average across all problems
    Enum.sum(pass_at_k_per_problem) / length(pass_at_k_per_problem)
  end

  defp compute_for_problem(n, c, k) when n < k do
    # If we have fewer samples than k, just check if any passed
    if c > 0, do: 1.0, else: 0.0
  end

  defp compute_for_problem(n, c, k) do
    # Use unbiased estimator
    1.0 - (binomial(n - c, k) / binomial(n, k))
  end

  # Compute binomial coefficient C(n, k)
  defp binomial(n, k) when k > n, do: 0.0
  defp binomial(n, k) when k == 0 or k == n, do: 1.0
  defp binomial(n, k) do
    # Use logarithms for numerical stability
    log_result =
      log_factorial(n) - log_factorial(k) - log_factorial(n - k)

    :math.exp(log_result)
  end

  # Log factorial using Stirling's approximation for large n
  defp log_factorial(n) when n > 20 do
    # Stirling's approximation: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
    n * :math.log(n) - n + 0.5 * :math.log(2 * :math.pi() * n)
  end

  defp log_factorial(n) do
    # Exact calculation for small n
    1..n
    |> Enum.reduce(0.0, fn i, acc -> acc + :math.log(i) end)
  end
end

# Usage example
defmodule HumanEvalEvaluator do
  def evaluate_pass_at_k(model, dataset, samples_per_problem: n, k_values: ks) do
    # Generate n samples per problem
    results =
      Enum.flat_map(dataset.items, fn problem ->
        Enum.map(1..n, fn _ ->
          code = model.generate(problem.input.signature)
          passed = run_tests(code, problem.input.tests)

          %{
            problem_id: problem.id,
            generated_code: code,
            all_tests_passed: passed
          }
        end)
      end)

    # Compute Pass@k for each k value
    Enum.map(ks, fn k ->
      score = PassAtK.compute(results, k)
      {k, score}
    end)
    |> Map.new()
  end

  defp run_tests(code, test_code) do
    # Execute tests and check if all pass
    # Returns true/false
    TestRunner.run_and_check(code, test_code)
  end
end

# Evaluate model
results = HumanEvalEvaluator.evaluate_pass_at_k(
  my_model,
  humaneval_dataset,
  samples_per_problem: 100,
  k_values: [1, 10, 100]
)

IO.inspect(results)
# => %{1 => 0.45, 10 => 0.72, 100 => 0.89}
```

**Interpreting Pass@k**:

- **Pass@1**: Most stringent - single sample must be correct
- **Pass@10**: More lenient - allows model to explore multiple solutions
- **Pass@100**: Estimates upper bound of model capabilities

**Best Practices**:

1. Generate at least 100 samples per problem for stable Pass@k estimation
2. Use temperature sampling (not greedy) to get diverse solutions
3. Report Pass@1 and Pass@10 as primary metrics
4. Consider computational cost when choosing k values

---

## Batch Evaluation

### Comparing Multiple Models

```elixir
defmodule ModelBenchmark do
  @doc """
  Evaluate multiple models on the same dataset.
  """
  def compare_models(dataset, models) do
    # Generate predictions for each model
    model_predictions =
      Enum.map(models, fn {name, model} ->
        predictions =
          Enum.map(dataset.items, fn item ->
            result = model.predict(item.input)

            %{
              id: item.id,
              predicted: result,
              metadata: %{
                model: name,
                inference_time_ms: result.time_ms
              }
            }
          end)

        {name, predictions}
      end)

    # Batch evaluate
    {:ok, results} = DatasetManager.evaluate_batch(
      model_predictions,
      dataset: dataset,
      metrics: [:exact_match, :f1]
    )

    # Format comparison table
    format_comparison_table(results)
  end

  defp format_comparison_table(results) do
    headers = ["Model", "Accuracy", "F1", "Avg Time (ms)"]

    rows =
      Enum.map(results, fn result ->
        [
          result.model_name,
          "#{Float.round(result.metrics.exact_match * 100, 2)}%",
          "#{Float.round(result.metrics.f1, 3)}",
          "#{Float.round(result.avg_inference_time_ms, 1)}"
        ]
      end)

    TableFormatter.format([headers | rows])
  end
end

# Example usage
models = [
  {"gpt-4", gpt4_model},
  {"claude-3", claude_model},
  {"llama-3-70b", llama_model},
  {"gemini-pro", gemini_model}
]

{:ok, dataset} = DatasetManager.load(:mmlu_stem, sample_size: 200)
comparison = ModelBenchmark.compare_models(dataset, models)

IO.puts(comparison)
# ┌────────────────┬──────────┬──────┬───────────────┐
# │ Model          │ Accuracy │ F1   │ Avg Time (ms) │
# ├────────────────┼──────────┼──────┼───────────────┤
# │ gpt-4          │ 87.50%   │ 0.89 │ 1234.5        │
# │ claude-3       │ 85.00%   │ 0.87 │ 987.2         │
# │ llama-3-70b    │ 82.50%   │ 0.84 │ 456.8         │
# │ gemini-pro     │ 86.00%   │ 0.88 │ 789.3         │
# └────────────────┴──────────┴──────┴───────────────┘
```

### Parallel Evaluation

```elixir
defmodule ParallelEvaluator do
  @doc """
  Evaluate predictions in parallel using Task.async_stream.
  """
  def evaluate_parallel(predictions, dataset, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 10)
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    # Create batches
    prediction_batches = Enum.chunk_every(predictions, batch_size)

    # Process batches in parallel
    results =
      Task.async_stream(
        prediction_batches,
        fn batch ->
          evaluate_batch(batch, dataset)
        end,
        max_concurrency: max_concurrency,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, result} -> result end)
      |> List.flatten()

    # Aggregate results
    aggregate_results(results)
  end

  defp evaluate_batch(batch, dataset) do
    Enum.map(batch, fn prediction ->
      item = Enum.find(dataset.items, &(&1.id == prediction.id))

      %{
        id: prediction.id,
        correct: prediction.predicted == item.expected,
        score: compute_score(prediction.predicted, item.expected)
      }
    end)
  end

  defp aggregate_results(results) do
    total = length(results)
    correct = Enum.count(results, & &1.correct)
    avg_score = Enum.map(results, & &1.score) |> Enum.sum() |> Kernel./(total)

    %{
      total: total,
      correct: correct,
      accuracy: correct / total,
      avg_score: avg_score
    }
  end

  defp compute_score(predicted, expected) do
    if predicted == expected, do: 1.0, else: 0.0
  end
end
```

### Distributed Evaluation

```elixir
defmodule DistributedEvaluator do
  @moduledoc """
  Distribute evaluation across multiple nodes.

  Useful for large-scale experiments or expensive models.
  """

  def evaluate_distributed(predictions, dataset, nodes) do
    # Partition predictions across nodes
    partitions = partition_predictions(predictions, length(nodes))

    # Start evaluation tasks on each node
    tasks =
      Enum.zip(nodes, partitions)
      |> Enum.map(fn {node, partition} ->
        Task.Supervisor.async(
          {EvalTaskSupervisor, node},
          fn ->
            DatasetManager.evaluate(partition,
              dataset: dataset,
              metrics: [:exact_match, :f1]
            )
          end
        )
      end)

    # Collect results
    results =
      Task.await_many(tasks, :infinity)
      |> Enum.map(fn {:ok, result} -> result end)

    # Merge results
    merge_evaluation_results(results)
  end

  defp partition_predictions(predictions, num_partitions) do
    chunk_size = ceil(length(predictions) / num_partitions)
    Enum.chunk_every(predictions, chunk_size)
  end

  defp merge_evaluation_results(results) do
    # Combine item-level results
    all_items = Enum.flat_map(results, & &1.items)

    # Recalculate aggregate metrics
    total = length(all_items)
    correct = Enum.count(all_items, & &1.correct)

    %{
      total: total,
      correct: correct,
      accuracy: correct / total,
      items: all_items
    }
  end
end

# Usage with cluster
nodes = [:"eval1@host", :"eval2@host", :"eval3@host"]
{:ok, dataset} = DatasetManager.load(:mmlu, sample_size: 5000)

results = DistributedEvaluator.evaluate_distributed(
  predictions,
  dataset,
  nodes
)
```

---

## Custom Datasets

### Creating a Custom Dataset

```elixir
defmodule CustomDatasetExample do
  alias DatasetManager.Dataset

  @doc """
  Create a custom dataset from raw data.
  """
  def create_custom_dataset do
    items = [
      %{
        id: "custom_1",
        input: "What is 2 + 2?",
        expected: 4,
        metadata: %{difficulty: "easy", topic: "arithmetic"}
      },
      %{
        id: "custom_2",
        input: "Solve x^2 = 16",
        expected: [4, -4],
        metadata: %{difficulty: "medium", topic: "algebra"}
      },
      %{
        id: "custom_3",
        input: "Integrate x^2 from 0 to 1",
        expected: 1/3,
        metadata: %{difficulty: "hard", topic: "calculus"}
      }
    ]

    Dataset.new(
      "custom_math",
      "1.0",
      items,
      %{
        description: "Custom math problem dataset",
        created_by: "research_team",
        license: "MIT"
      }
    )
  end
end
```

### Loading from JSONL

```elixir
defmodule JSONLLoader do
  @doc """
  Load custom dataset from JSONL file.

  JSONL format (one JSON object per line):
  {"id": "q1", "input": "...", "expected": "...", "metadata": {...}}
  {"id": "q2", "input": "...", "expected": "...", "metadata": {...}}
  """
  def load_from_jsonl(file_path) do
    items =
      File.stream!(file_path)
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.map(&parse_item/1)

    dataset_name = Path.basename(file_path, ".jsonl")

    Dataset.new(
      dataset_name,
      "1.0",
      items,
      %{source: file_path, format: "jsonl"}
    )
  end

  defp parse_item(json) do
    %{
      id: json["id"],
      input: json["input"],
      expected: json["expected"] || json["answer"],
      metadata: json["metadata"] || %{}
    }
  end
end

# Usage
{:ok, dataset} = DatasetManager.load("custom_dataset",
  source: "/path/to/dataset.jsonl"
)
```

### Loading from CSV

```elixir
defmodule CSVLoader do
  @doc """
  Load custom dataset from CSV file.

  CSV format:
  id,question,answer,category,difficulty
  q1,"What is ...","42",math,easy
  q2,"Explain ...","...",science,hard
  """
  def load_from_csv(file_path) do
    items =
      File.stream!(file_path)
      |> NimbleCSV.RFC4180.parse_stream(skip_headers: true)
      |> Enum.map(&parse_csv_row/1)

    dataset_name = Path.basename(file_path, ".csv")

    Dataset.new(
      dataset_name,
      "1.0",
      items,
      %{source: file_path, format: "csv"}
    )
  end

  defp parse_csv_row([id, question, answer, category, difficulty]) do
    %{
      id: id,
      input: question,
      expected: parse_answer(answer),
      metadata: %{
        category: category,
        difficulty: difficulty
      }
    }
  end

  defp parse_answer(answer) do
    # Try to parse as integer, fall back to string
    case Integer.parse(answer) do
      {num, ""} -> num
      _ -> answer
    end
  end
end
```

### Loading from HuggingFace

```elixir
defmodule HuggingFaceLoader do
  @doc """
  Load dataset from HuggingFace Hub.

  Requires: datasets Python library
  """
  def load_from_huggingface(repo_name, opts \\ []) do
    config = Keyword.get(opts, :config, "default")
    split = Keyword.get(opts, :split, "train")

    # Use Python via ErlPort
    python_code = """
    from datasets import load_dataset
    import json

    dataset = load_dataset('#{repo_name}', '#{config}', split='#{split}')

    # Convert to JSON
    items = []
    for i, item in enumerate(dataset):
        items.append({
            'id': f'{i}',
            'input': item.get('input', item.get('question')),
            'expected': item.get('output', item.get('answer')),
            'metadata': {k: v for k, v in item.items() if k not in ['input', 'output', 'question', 'answer']}
        })

    print(json.dumps(items))
    """

    # Execute Python code
    {output, 0} = System.cmd("python3", ["-c", python_code])

    items = Jason.decode!(output)

    Dataset.new(
      repo_name,
      "1.0",
      items,
      %{source: "huggingface:#{repo_name}", config: config, split: split}
    )
  end
end

# Usage
dataset = HuggingFaceLoader.load_from_huggingface(
  "cais/mmlu",
  config: "abstract_algebra",
  split: "test"
)
```

### Custom Dataset Validator

```elixir
defmodule DatasetValidator do
  @doc """
  Validate custom dataset structure.
  """
  def validate(dataset) do
    with :ok <- validate_required_fields(dataset),
         :ok <- validate_items(dataset.items),
         :ok <- validate_unique_ids(dataset.items) do
      {:ok, dataset}
    end
  end

  defp validate_required_fields(%{name: _, version: _, items: _}), do: :ok
  defp validate_required_fields(_), do: {:error, :missing_required_fields}

  defp validate_items(items) when is_list(items) do
    invalid_items =
      items
      |> Enum.with_index()
      |> Enum.reject(fn {item, _idx} ->
        is_map(item) and
          Map.has_key?(item, :id) and
          Map.has_key?(item, :input) and
          Map.has_key?(item, :expected)
      end)

    if Enum.empty?(invalid_items) do
      :ok
    else
      {:error, {:invalid_items, Enum.map(invalid_items, &elem(&1, 1))}}
    end
  end

  defp validate_items(_), do: {:error, :items_must_be_list}

  defp validate_unique_ids(items) do
    ids = Enum.map(items, & &1.id)
    unique_ids = Enum.uniq(ids)

    if length(ids) == length(unique_ids) do
      :ok
    else
      duplicates = ids -- unique_ids
      {:error, {:duplicate_ids, duplicates}}
    end
  end
end

# Usage
case DatasetValidator.validate(my_dataset) do
  {:ok, dataset} ->
    IO.puts("Dataset is valid!")

  {:error, {:invalid_items, indices}} ->
    IO.puts("Invalid items at indices: #{inspect(indices)}")

  {:error, {:duplicate_ids, ids}} ->
    IO.puts("Duplicate IDs found: #{inspect(ids)}")

  {:error, reason} ->
    IO.puts("Validation failed: #{inspect(reason)}")
end
```

---

## Data Preprocessing

### Text Normalization

```elixir
defmodule TextNormalizer do
  @doc """
  Normalize text for consistent comparison.
  """
  def normalize(text, opts \\ []) do
    text
    |> maybe_lowercase(opts)
    |> remove_extra_whitespace()
    |> maybe_remove_punctuation(opts)
    |> maybe_remove_articles(opts)
  end

  defp maybe_lowercase(text, opts) do
    if Keyword.get(opts, :lowercase, true) do
      String.downcase(text)
    else
      text
    end
  end

  defp remove_extra_whitespace(text) do
    text
    |> String.trim()
    |> String.replace(~r/\s+/, " ")
  end

  defp maybe_remove_punctuation(text, opts) do
    if Keyword.get(opts, :remove_punctuation, true) do
      String.replace(text, ~r/[^\w\s]/, "")
    else
      text
    end
  end

  defp maybe_remove_articles(text, opts) do
    if Keyword.get(opts, :remove_articles, false) do
      String.replace(text, ~r/\b(a|an|the)\b/i, "")
      |> String.trim()
    else
      text
    end
  end
end

# Apply to dataset
defmodule DatasetPreprocessor do
  def normalize_answers(dataset) do
    normalized_items =
      Enum.map(dataset.items, fn item ->
        normalized_expected =
          if is_binary(item.expected) do
            TextNormalizer.normalize(item.expected)
          else
            item.expected
          end

        %{item | expected: normalized_expected}
      end)

    %{dataset | items: normalized_items}
  end
end
```

### Feature Extraction

```elixir
defmodule FeatureExtractor do
  @doc """
  Extract features from dataset items for analysis.
  """
  def extract_features(dataset) do
    Enum.map(dataset.items, fn item ->
      %{
        id: item.id,
        input_length: count_tokens(item.input),
        expected_length: count_tokens(item.expected),
        has_numbers: contains_numbers?(item.input),
        has_code: contains_code?(item.input),
        complexity: estimate_complexity(item),
        metadata: item.metadata
      }
    end)
  end

  defp count_tokens(text) when is_binary(text) do
    text
    |> String.split(~r/\s+/)
    |> length()
  end

  defp count_tokens(_), do: 0

  defp contains_numbers?(text) when is_binary(text) do
    String.match?(text, ~r/\d+/)
  end

  defp contains_numbers?(_), do: false

  defp contains_code?(text) when is_binary(text) do
    # Heuristic: Check for common code patterns
    String.match?(text, ~r/(def |function |class |import |return )/)
  end

  defp contains_code?(_), do: false

  defp estimate_complexity(item) do
    # Simple heuristic based on input/output length
    input_len = count_tokens(item.input)
    expected_len = count_tokens(item.expected)

    cond do
      input_len + expected_len < 50 -> :low
      input_len + expected_len < 200 -> :medium
      true -> :high
    end
  end
end

# Analyze dataset
features = FeatureExtractor.extract_features(dataset)

# Compute statistics
avg_input_length =
  features
  |> Enum.map(& &1.input_length)
  |> Enum.sum()
  |> Kernel./(length(features))

IO.puts("Average input length: #{avg_input_length} tokens")

complexity_dist =
  features
  |> Enum.group_by(& &1.complexity)
  |> Enum.map(fn {complexity, items} ->
    {complexity, length(items)}
  end)
  |> Map.new()

IO.inspect(complexity_dist)
# => %{low: 45, medium: 32, high: 23}
```

### Data Augmentation

```elixir
defmodule DataAugmenter do
  @doc """
  Augment dataset with variations.
  """
  def augment(dataset, strategies) do
    augmented_items =
      Enum.flat_map(dataset.items, fn item ->
        variations = generate_variations(item, strategies)
        [item | variations]
      end)

    %{dataset |
      items: augmented_items,
      metadata: Map.put(dataset.metadata, :augmented, true)
    }
  end

  defp generate_variations(item, strategies) do
    Enum.flat_map(strategies, fn strategy ->
      apply_strategy(item, strategy)
    end)
  end

  defp apply_strategy(item, :paraphrase) do
    # Generate paraphrases of the input
    paraphrases = generate_paraphrases(item.input)

    Enum.map(paraphrases, fn paraphrase ->
      %{item |
        id: "#{item.id}_para_#{:rand.uniform(1000)}",
        input: paraphrase,
        metadata: Map.put(item.metadata, :augmented, :paraphrase)
      }
    end)
  end

  defp apply_strategy(item, :back_translation) do
    # Back-translate through another language
    translated = back_translate(item.input, through: :french)

    [%{item |
      id: "#{item.id}_bt",
      input: translated,
      metadata: Map.put(item.metadata, :augmented, :back_translation)
    }]
  end

  defp apply_strategy(item, :typo_injection) do
    # Introduce realistic typos
    typo_text = inject_typos(item.input, rate: 0.05)

    [%{item |
      id: "#{item.id}_typo",
      input: typo_text,
      metadata: Map.put(item.metadata, :augmented, :typo)
    }]
  end

  defp generate_paraphrases(text) do
    # Use LLM or rule-based paraphrasing
    # Placeholder implementation
    [text]
  end

  defp back_translate(text, through: lang) do
    # Translate to intermediate language and back
    # Placeholder implementation
    text
  end

  defp inject_typos(text, rate: rate) do
    # Randomly inject typos
    text
    |> String.graphemes()
    |> Enum.map(fn char ->
      if :rand.uniform() < rate do
        typo_char(char)
      else
        char
      end
    end)
    |> Enum.join()
  end

  defp typo_char(char) do
    # Common typo patterns
    typos = %{
      "a" => ["s", "q", "z"],
      "e" => ["r", "w", "d"],
      "i" => ["o", "u", "k"],
      "o" => ["i", "p", "l"],
      "u" => ["y", "i", "j"]
    }

    alternatives = Map.get(typos, String.downcase(char), [char])
    Enum.random(alternatives)
  end
end

# Usage
augmented_dataset = DataAugmenter.augment(dataset, [
  :paraphrase,
  :back_translation,
  :typo_injection
])

IO.puts("Original size: #{length(dataset.items)}")
IO.puts("Augmented size: #{length(augmented_dataset.items)}")
```

---

## Advanced Usage

### Ensemble Evaluation

```elixir
defmodule EnsembleEvaluator do
  @doc """
  Evaluate ensemble of models with voting.
  """
  def evaluate_ensemble(models, dataset, voting_strategy: strategy) do
    # Generate predictions from each model
    all_predictions =
      Enum.map(models, fn {name, model} ->
        predictions = generate_predictions(model, dataset)
        {name, predictions}
      end)

    # Combine predictions using voting
    ensemble_predictions = combine_predictions(all_predictions, strategy)

    # Evaluate ensemble
    {:ok, results} = DatasetManager.evaluate(
      ensemble_predictions,
      dataset: dataset,
      metrics: [:exact_match, :f1]
    )

    results
  end

  defp generate_predictions(model, dataset) do
    Enum.map(dataset.items, fn item ->
      result = model.predict(item.input)

      %{
        id: item.id,
        predicted: result.answer,
        confidence: result.confidence,
        metadata: result.metadata
      }
    end)
  end

  defp combine_predictions(all_predictions, :majority_vote) do
    # Group predictions by item ID
    by_item =
      all_predictions
      |> Enum.flat_map(fn {_name, preds} -> preds end)
      |> Enum.group_by(& &1.id)

    # Majority vote for each item
    Enum.map(by_item, fn {id, item_preds} ->
      voted_answer = majority_vote(Enum.map(item_preds, & &1.predicted))

      %{
        id: id,
        predicted: voted_answer,
        metadata: %{
          ensemble_size: length(item_preds),
          vote_counts: count_votes(Enum.map(item_preds, & &1.predicted))
        }
      }
    end)
  end

  defp combine_predictions(all_predictions, :weighted_vote) do
    # Weight votes by model confidence
    by_item =
      all_predictions
      |> Enum.flat_map(fn {_name, preds} -> preds end)
      |> Enum.group_by(& &1.id)

    Enum.map(by_item, fn {id, item_preds} ->
      weighted_answer = weighted_vote(
        Enum.map(item_preds, &{&1.predicted, &1.confidence})
      )

      %{
        id: id,
        predicted: weighted_answer,
        metadata: %{ensemble_size: length(item_preds)}
      }
    end)
  end

  defp majority_vote(answers) do
    answers
    |> Enum.frequencies()
    |> Enum.max_by(fn {_answer, count} -> count end)
    |> elem(0)
  end

  defp weighted_vote(answer_confidence_pairs) do
    answer_confidence_pairs
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {answer, confidences} ->
      {answer, Enum.sum(confidences)}
    end)
    |> Enum.max_by(&elem(&1, 1))
    |> elem(0)
  end

  defp count_votes(answers) do
    Enum.frequencies(answers)
  end
end
```

### Confidence Calibration

```elixir
defmodule ConfidenceCalibrator do
  @doc """
  Calibrate model confidence scores.

  Uses temperature scaling or Platt scaling.
  """
  def calibrate(predictions, ground_truth, method: :temperature_scaling) do
    # Find optimal temperature that minimizes calibration error
    optimal_temp = find_optimal_temperature(predictions, ground_truth)

    # Apply temperature scaling
    Enum.map(predictions, fn pred ->
      calibrated_confidence = apply_temperature(pred.confidence, optimal_temp)
      %{pred | confidence: calibrated_confidence}
    end)
  end

  defp find_optimal_temperature(predictions, ground_truth) do
    # Grid search for best temperature
    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    temperatures
    |> Enum.map(fn temp ->
      error = calibration_error(predictions, ground_truth, temp)
      {temp, error}
    end)
    |> Enum.min_by(&elem(&1, 1))
    |> elem(0)
  end

  defp calibration_error(predictions, ground_truth, temperature) do
    # Expected Calibration Error (ECE)
    bins = create_confidence_bins(10)

    bin_errors =
      Enum.map(bins, fn {bin_min, bin_max} ->
        bin_predictions =
          Enum.filter(predictions, fn pred ->
            scaled_conf = apply_temperature(pred.confidence, temperature)
            scaled_conf >= bin_min and scaled_conf < bin_max
          end)

        if Enum.empty?(bin_predictions) do
          0.0
        else
          avg_confidence =
            bin_predictions
            |> Enum.map(&apply_temperature(&1.confidence, temperature))
            |> average()

          accuracy = calculate_bin_accuracy(bin_predictions, ground_truth)

          abs(avg_confidence - accuracy) * length(bin_predictions) / length(predictions)
        end
      end)

    Enum.sum(bin_errors)
  end

  defp apply_temperature(confidence, temperature) do
    # Softmax temperature scaling
    1 / (1 + :math.exp(-confidence / temperature))
  end

  defp create_confidence_bins(n) do
    step = 1.0 / n
    Enum.map(0..(n-1), fn i ->
      {i * step, (i + 1) * step}
    end)
  end

  defp calculate_bin_accuracy(predictions, ground_truth) do
    correct =
      Enum.count(predictions, fn pred ->
        expected = Enum.find(ground_truth, &(&1.id == pred.id)).expected
        pred.predicted == expected
      end)

    correct / length(predictions)
  end

  defp average(list), do: Enum.sum(list) / length(list)
end
```

### Error Analysis

```elixir
defmodule ErrorAnalyzer do
  @doc """
  Perform detailed error analysis on evaluation results.
  """
  def analyze(results, dataset) do
    incorrect_items = Enum.reject(results.items, & &1.correct)

    %{
      total_errors: length(incorrect_items),
      error_rate: length(incorrect_items) / length(results.items),
      error_by_category: analyze_by_category(incorrect_items, dataset),
      error_by_difficulty: analyze_by_difficulty(incorrect_items, dataset),
      common_failure_modes: identify_failure_modes(incorrect_items, dataset),
      hardest_items: find_hardest_items(results.items, limit: 10)
    }
  end

  defp analyze_by_category(errors, dataset) do
    errors
    |> Enum.map(fn error ->
      item = Enum.find(dataset.items, &(&1.id == error.id))
      get_in(item.metadata, [:category]) || "unknown"
    end)
    |> Enum.frequencies()
    |> Enum.sort_by(&elem(&1, 1), :desc)
  end

  defp analyze_by_difficulty(errors, dataset) do
    errors
    |> Enum.map(fn error ->
      item = Enum.find(dataset.items, &(&1.id == error.id))
      get_in(item.metadata, [:difficulty]) || "unknown"
    end)
    |> Enum.frequencies()
  end

  defp identify_failure_modes(errors, dataset) do
    # Cluster errors by type
    errors
    |> Enum.map(fn error ->
      item = Enum.find(dataset.items, &(&1.id == error.id))
      classify_error_type(error, item)
    end)
    |> Enum.frequencies()
    |> Enum.sort_by(&elem(&1, 1), :desc)
  end

  defp classify_error_type(error, item) do
    cond do
      is_nil(error.predicted) ->
        :no_answer

      is_binary(error.predicted) and String.length(error.predicted) == 0 ->
        :empty_answer

      error.predicted == item.expected ->
        :correct

      similar?(error.predicted, item.expected) ->
        :near_miss

      true ->
        :wrong_answer
    end
  end

  defp similar?(pred, expected) when is_binary(pred) and is_binary(expected) do
    String.jaro_distance(pred, expected) > 0.8
  end

  defp similar?(_, _), do: false

  defp find_hardest_items(items, limit: limit) do
    items
    |> Enum.sort_by(& &1.score)
    |> Enum.take(limit)
  end
end

# Usage
{:ok, results} = DatasetManager.evaluate(predictions, dataset: dataset)
error_analysis = ErrorAnalyzer.analyze(results, dataset)

IO.inspect(error_analysis)
# => %{
#   total_errors: 23,
#   error_rate: 0.23,
#   error_by_category: [{"math", 12}, {"reading", 8}, {"science", 3}],
#   error_by_difficulty: %{"easy" => 3, "medium" => 10, "hard" => 10},
#   common_failure_modes: [
#     {:near_miss, 8},
#     {:wrong_answer, 12},
#     {:no_answer, 3}
#   ],
#   hardest_items: [...]
# }
```

---

## Performance Optimization

### Caching Strategies

```elixir
defmodule CachingStrategy do
  @doc """
  Implement multi-level caching for dataset operations.
  """

  # L1: Process-local ETS cache
  defp get_from_l1_cache(key) do
    case :ets.lookup(:dataset_cache_l1, key) do
      [{^key, value}] -> {:ok, value}
      [] -> :miss
    end
  end

  defp put_in_l1_cache(key, value) do
    :ets.insert(:dataset_cache_l1, {key, value})
  end

  # L2: Disk cache
  defp get_from_l2_cache(key) do
    cache_file = cache_path(key)

    if File.exists?(cache_file) do
      {:ok, binary} = File.read(cache_file)
      {:ok, :erlang.binary_to_term(binary)}
    else
      :miss
    end
  end

  defp put_in_l2_cache(key, value) do
    cache_file = cache_path(key)
    binary = :erlang.term_to_binary(value)
    File.write!(cache_file, binary)
  end

  # Hierarchical cache lookup
  def get_cached(key) do
    case get_from_l1_cache(key) do
      {:ok, value} ->
        {:ok, value}

      :miss ->
        case get_from_l2_cache(key) do
          {:ok, value} ->
            put_in_l1_cache(key, value)
            {:ok, value}

          :miss ->
            :miss
        end
    end
  end

  defp cache_path(key) do
    cache_dir = Application.get_env(:dataset_manager, :cache_dir)
    Path.join([cache_dir, "datasets", "#{key}.etf"])
  end
end
```

### Lazy Loading

```elixir
defmodule LazyDataset do
  @moduledoc """
  Lazy-loading dataset for memory efficiency.
  """

  defstruct [:name, :version, :metadata, :item_loader, :total_items]

  def new(name, version, item_loader, total_items, metadata \\ %{}) do
    %__MODULE__{
      name: name,
      version: version,
      item_loader: item_loader,
      total_items: total_items,
      metadata: metadata
    }
  end

  @doc """
  Stream dataset items without loading all into memory.
  """
  def stream(%__MODULE__{} = dataset) do
    Stream.resource(
      fn -> 0 end,
      fn index ->
        if index < dataset.total_items do
          item = dataset.item_loader.(index)
          {[item], index + 1}
        else
          {:halt, index}
        end
      end,
      fn _ -> :ok end
    )
  end

  @doc """
  Get specific item by index.
  """
  def get_item(%__MODULE__{} = dataset, index) do
    if index >= 0 and index < dataset.total_items do
      {:ok, dataset.item_loader.(index)}
    else
      {:error, :index_out_of_bounds}
    end
  end
end

# Example: Create lazy-loading MMLU dataset
defmodule MMLULazy do
  def load do
    item_loader = fn index ->
      # Load item from file/database only when needed
      load_mmlu_item(index)
    end

    LazyDataset.new("mmlu", "1.0", item_loader, 14042)
  end

  defp load_mmlu_item(index) do
    # Load specific item (from file, DB, etc.)
    %{
      id: "mmlu_#{index}",
      input: "...",
      expected: "..."
    }
  end
end

# Usage
lazy_dataset = MMLULazy.load()

# Process in batches without loading all data
LazyDataset.stream(lazy_dataset)
|> Stream.chunk_every(100)
|> Enum.each(fn batch ->
  process_batch(batch)
end)
```

### Parallel Processing

```elixir
defmodule ParallelProcessor do
  @doc """
  Process dataset items in parallel using Flow.
  """
  def process_parallel(dataset, processor_fn, opts \\ []) do
    stages = Keyword.get(opts, :stages, System.schedulers_online())
    max_demand = Keyword.get(opts, :max_demand, 50)

    dataset.items
    |> Flow.from_enumerable(stages: stages, max_demand: max_demand)
    |> Flow.map(processor_fn)
    |> Enum.to_list()
  end
end

# Example: Parallel evaluation
defmodule ParallelEval do
  def evaluate_parallel(model, dataset) do
    ParallelProcessor.process_parallel(
      dataset,
      fn item ->
        prediction = model.predict(item.input)
        correct = prediction == item.expected

        %{
          id: item.id,
          predicted: prediction,
          expected: item.expected,
          correct: correct
        }
      end,
      stages: 8,
      max_demand: 10
    )
  end
end
```

---

## Research Best Practices

### Reproducibility

```elixir
defmodule ReproducibleExperiment do
  @doc """
  Ensure experiment reproducibility with comprehensive logging.
  """
  def run_experiment(config) do
    # Set random seeds
    :rand.seed(:exsss, {config.seed, config.seed, config.seed})

    # Log configuration
    experiment_id = generate_experiment_id()
    log_config(experiment_id, config)

    # Load dataset with version pinning
    {:ok, dataset} = DatasetManager.load(
      config.dataset_name,
      version: config.dataset_version,
      seed: config.seed
    )

    # Run evaluation
    {:ok, results} = DatasetManager.evaluate(
      config.predictions,
      dataset: dataset,
      metrics: config.metrics
    )

    # Log results with full provenance
    log_results(experiment_id, results, %{
      dataset_version: dataset.version,
      dataset_size: length(dataset.items),
      timestamp: DateTime.utc_now(),
      elixir_version: System.version(),
      otp_version: System.otp_release()
    })

    {:ok, experiment_id, results}
  end

  defp generate_experiment_id do
    "exp_#{DateTime.utc_now() |> DateTime.to_unix()}_#{:rand.uniform(10000)}"
  end

  defp log_config(experiment_id, config) do
    log_file = "experiments/#{experiment_id}/config.json"
    File.mkdir_p!(Path.dirname(log_file))

    config_json = Jason.encode!(config, pretty: true)
    File.write!(log_file, config_json)
  end

  defp log_results(experiment_id, results, metadata) do
    log_file = "experiments/#{experiment_id}/results.json"

    output = %{
      results: results,
      metadata: metadata
    }

    results_json = Jason.encode!(output, pretty: true)
    File.write!(log_file, results_json)
  end
end
```

### Statistical Significance Testing

```elixir
defmodule StatisticalTesting do
  @doc """
  Perform statistical significance tests between models.
  """
  def compare_models(results_a, results_b) do
    # Paired t-test
    differences =
      Enum.zip(results_a.items, results_b.items)
      |> Enum.map(fn {a, b} ->
        a.score - b.score
      end)

    {t_stat, p_value} = paired_t_test(differences)

    # Effect size (Cohen's d)
    effect_size = cohens_d(
      Enum.map(results_a.items, & &1.score),
      Enum.map(results_b.items, & &1.score)
    )

    %{
      t_statistic: t_stat,
      p_value: p_value,
      significant: p_value < 0.05,
      effect_size: effect_size,
      interpretation: interpret_effect_size(effect_size)
    }
  end

  defp paired_t_test(differences) do
    n = length(differences)
    mean_diff = Enum.sum(differences) / n

    variance =
      differences
      |> Enum.map(&:math.pow(&1 - mean_diff, 2))
      |> Enum.sum()
      |> Kernel./(n - 1)

    std_error = :math.sqrt(variance / n)
    t_stat = mean_diff / std_error

    # Calculate p-value (two-tailed)
    df = n - 1
    p_value = 2 * (1 - Statistex.Distributions.T.cdf(abs(t_stat), df))

    {t_stat, p_value}
  end

  defp cohens_d(sample_a, sample_b) do
    mean_a = Enum.sum(sample_a) / length(sample_a)
    mean_b = Enum.sum(sample_b) / length(sample_b)

    var_a = variance(sample_a, mean_a)
    var_b = variance(sample_b, mean_b)

    pooled_std = :math.sqrt((var_a + var_b) / 2)

    (mean_a - mean_b) / pooled_std
  end

  defp variance(sample, mean) do
    sample
    |> Enum.map(&:math.pow(&1 - mean, 2))
    |> Enum.sum()
    |> Kernel./(length(sample) - 1)
  end

  defp interpret_effect_size(d) do
    abs_d = abs(d)

    cond do
      abs_d < 0.2 -> "negligible"
      abs_d < 0.5 -> "small"
      abs_d < 0.8 -> "medium"
      true -> "large"
    end
  end
end
```

### Experiment Tracking

```elixir
defmodule ExperimentTracker do
  @doc """
  Track experiments with MLflow-style interface.
  """
  def start_run(experiment_name) do
    run_id = generate_run_id()

    run = %{
      run_id: run_id,
      experiment_name: experiment_name,
      start_time: DateTime.utc_now(),
      params: %{},
      metrics: %{},
      artifacts: []
    }

    :ets.insert(:experiment_runs, {run_id, run})
    run_id
  end

  def log_param(run_id, key, value) do
    update_run(run_id, fn run ->
      %{run | params: Map.put(run.params, key, value)}
    end)
  end

  def log_metric(run_id, key, value, step \\ 0) do
    update_run(run_id, fn run ->
      metrics = Map.get(run.metrics, key, [])
      updated_metrics = Map.put(run.metrics, key, metrics ++ [{step, value}])
      %{run | metrics: updated_metrics}
    end)
  end

  def log_artifact(run_id, artifact_path) do
    update_run(run_id, fn run ->
      %{run | artifacts: run.artifacts ++ [artifact_path]}
    end)
  end

  def end_run(run_id) do
    update_run(run_id, fn run ->
      %{run | end_time: DateTime.utc_now()}
    end)

    # Save to disk
    [{^run_id, run}] = :ets.lookup(:experiment_runs, run_id)
    save_run(run)
  end

  defp update_run(run_id, update_fn) do
    [{^run_id, run}] = :ets.lookup(:experiment_runs, run_id)
    updated_run = update_fn.(run)
    :ets.insert(:experiment_runs, {run_id, updated_run})
    :ok
  end

  defp save_run(run) do
    dir = "experiments/#{run.experiment_name}/#{run.run_id}"
    File.mkdir_p!(dir)

    File.write!(
      Path.join(dir, "run.json"),
      Jason.encode!(run, pretty: true)
    )
  end

  defp generate_run_id do
    "run_#{:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)}"
  end
end

# Usage
run_id = ExperimentTracker.start_run("mmlu_benchmark")

ExperimentTracker.log_param(run_id, "model", "gpt-4")
ExperimentTracker.log_param(run_id, "temperature", 0.7)
ExperimentTracker.log_param(run_id, "dataset_size", 500)

{:ok, results} = DatasetManager.evaluate(predictions, dataset: dataset)

ExperimentTracker.log_metric(run_id, "accuracy", results.metrics.exact_match)
ExperimentTracker.log_metric(run_id, "f1", results.metrics.f1)

ExperimentTracker.end_run(run_id)
```

---

## API Reference

### DatasetManager

**Main module for dataset operations**

#### Functions

##### `load/2`

Load a dataset by name.

```elixir
@spec load(atom() | String.t(), keyword()) :: {:ok, Dataset.t()} | {:error, term()}
```

**Options:**
- `:version` - Dataset version (default: "1.0")
- `:cache` - Use cache (default: true)
- `:sample_size` - Limit items (default: all)
- `:source` - Custom source path
- `:seed` - Random seed

##### `evaluate/2`

Evaluate predictions against dataset.

```elixir
@spec evaluate([prediction()], keyword()) :: {:ok, EvaluationResult.t()} | {:error, term()}
```

**Options:**
- `:dataset` - Dataset name or struct (required)
- `:metrics` - List of metrics (default: [:exact_match, :f1])
- `:model_name` - Model identifier

##### `evaluate_batch/2`

Batch evaluate multiple models.

```elixir
@spec evaluate_batch([{String.t(), [prediction()]}], keyword()) ::
  {:ok, [EvaluationResult.t()]} | {:error, term()}
```

##### `random_sample/2`

Create random sample.

```elixir
@spec random_sample(Dataset.t(), keyword()) :: {:ok, Dataset.t()}
```

##### `stratified_sample/2`

Create stratified sample.

```elixir
@spec stratified_sample(Dataset.t(), keyword()) :: {:ok, Dataset.t()} | {:error, term()}
```

##### `k_fold/2`

Create k-fold splits.

```elixir
@spec k_fold(Dataset.t(), keyword()) :: {:ok, [{Dataset.t(), Dataset.t()}]}
```

##### `train_test_split/2`

Split into train/test.

```elixir
@spec train_test_split(Dataset.t(), keyword()) :: {:ok, {Dataset.t(), Dataset.t()}}
```

### Dataset

**Dataset structure and operations**

#### Struct

```elixir
%Dataset{
  name: String.t(),
  version: String.t(),
  items: [map()],
  metadata: map()
}
```

#### Functions

##### `new/4`

Create new dataset.

```elixir
@spec new(String.t(), String.t(), [map()], map()) :: t()
```

##### `validate/1`

Validate dataset structure.

```elixir
@spec validate(t()) :: {:ok, t()} | {:error, term()}
```

### EvaluationResult

**Evaluation result structure**

#### Struct

```elixir
%EvaluationResult{
  dataset_name: String.t(),
  dataset_version: String.t(),
  model_name: String.t(),
  items: [map()],
  metrics: map(),
  duration_ms: integer()
}
```

---

## Conclusion

This guide provides comprehensive coverage of dataset management in the Elixir AI Research framework. Key takeaways:

1. **Standardization**: Use consistent dataset formats and evaluation metrics
2. **Reproducibility**: Pin versions, use seeds, log everything
3. **Efficiency**: Leverage caching and parallel processing
4. **Rigor**: Apply proper sampling and statistical testing
5. **Extensibility**: Easy integration of custom datasets and metrics

For questions or contributions, see the project repository or contact the research team.

---

**References**

1. Hendrycks, D., et al. (2021). "Measuring Massive Multitask Language Understanding." ICLR.
2. Chen, M., et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374.
3. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.
4. Dietterich, T. G. (1998). "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation.
5. Japkowicz, N., & Shah, M. (2011). "Evaluating Learning Algorithms: A Classification Perspective." Cambridge University Press.
