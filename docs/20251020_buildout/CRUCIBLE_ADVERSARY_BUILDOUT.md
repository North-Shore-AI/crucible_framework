# Crucible Adversary - Complete Implementation Prompt

**Version**: 0.1.0
**Date**: 2025-10-20
**Target**: Phase 1 Foundation Complete Implementation

---

## Table of Contents

1. [Project Context](#project-context)
2. [Architecture Overview](#architecture-overview)
3. [Module Structure & File Organization](#module-structure--file-organization)
4. [Implementation Requirements by Module](#implementation-requirements-by-module)
5. [TDD Process & Workflow](#tdd-process--workflow)
6. [Integration Points](#integration-points)
7. [Quality Gates & Success Criteria](#quality-gates--success-criteria)
8. [Example Usage Patterns](#example-usage-patterns)
9. [Performance Requirements](#performance-requirements)
10. [Testing Requirements](#testing-requirements)

---

## Project Context

### What is CrucibleAdversary?

CrucibleAdversary is a comprehensive adversarial testing framework designed for AI/ML systems in Elixir. It provides advanced attack generation, robustness evaluation, security vulnerability scanning, and stress testing capabilities for AI models integrated with the Crucible framework.

### Design Principles

1. **Security-First**: Identify vulnerabilities before they become exploits
2. **Comprehensive Coverage**: Multi-layered attack strategies across all vectors
3. **Measurable Robustness**: Quantifiable metrics for model resilience
4. **Production-Ready**: Real-world attack simulations for deployment confidence
5. **Research-Oriented**: Support for adversarial ML research and experimentation

### Core Features

- **Text Perturbations**: Character-level, word-level, and semantic perturbations
- **Prompt Attacks**: Injection attacks, context manipulation, delimiter attacks
- **Jailbreak Techniques**: Role-playing, context switching, encoding tricks
- **Robustness Testing**: Stress testing under adversarial conditions
- **Security Scanning**: Automated vulnerability detection and exploitation
- **Metrics & Analysis**: Comprehensive robustness metrics and reporting
- **Integration**: Seamless integration with Crucible framework components

### Dependencies

From `mix.exs`:
- Elixir: ~> 1.14
- OTP: 25+
- ex_doc: ~> 0.31 (dev only)

---

## Architecture Overview

### System Architecture

CrucibleAdversary follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  (Main API, CLI Interface, Pipeline Integration)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      Attack Layer                           │
│  Perturbations | Injection | Jailbreak | Extraction         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   Evaluation Layer                          │
│  Robustness | Security | Stress | Metrics                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    Defense Layer                            │
│  Detection | Filtering | Sanitization                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   Core Services                             │
│  Generators | Mutation Engine | Reporting | Storage         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → API → Attack Generator → Model Inference
                                              ↓
                                         Evaluator
                                              ↓
                                      Metrics Calculation
                                              ↓
                                      Report Generation
```

### Key Design Patterns

1. **Strategy Pattern**: For different attack types
2. **Pipeline Pattern**: For evaluation workflows
3. **Observer Pattern**: For monitoring and events
4. **Behavior Pattern**: For pluggable attack implementations

---

## Module Structure & File Organization

### Directory Structure

```
lib/crucible_adversary/
├── adversary.ex                      # Main API and coordination
├── config.ex                         # Configuration management
│
├── attacks/                          # Attack implementations
│   ├── attack.ex                    # Base attack behavior
│   ├── injection.ex                 # Prompt injection attacks
│   ├── jailbreak.ex                 # Jailbreak techniques
│   ├── extraction.ex                # Data extraction attacks
│   └── bias.ex                      # Bias exploitation
│
├── perturbations/                   # Text perturbation strategies
│   ├── character.ex                 # Character-level perturbations
│   ├── word.ex                      # Word-level perturbations
│   └── semantic.ex                  # Semantic-level perturbations
│
├── evaluation/                      # Evaluation systems
│   ├── robustness.ex                # Robustness evaluation
│   ├── security.ex                  # Security scanning
│   └── stress.ex                    # Stress testing
│
├── defenses/                        # Defense mechanisms
│   ├── detection.ex                 # Attack detection
│   ├── filtering.ex                 # Input filtering
│   └── sanitization.ex              # Input sanitization
│
├── generators/                      # Attack generation
│   ├── text_generator.ex            # Adversarial text generation
│   ├── prompt_generator.ex          # Attack prompt generation
│   └── mutation_engine.ex           # Mutation strategies
│
├── metrics/                         # Robustness metrics
│   ├── accuracy.ex                  # Accuracy-based metrics
│   ├── consistency.ex               # Consistency metrics
│   ├── certified.ex                 # Certified robustness
│   └── asr.ex                       # Attack success rate
│
└── reports/                         # Reporting and export
    ├── robustness_report.ex         # Robustness reports
    ├── security_report.ex           # Security reports
    └── export.ex                    # Export utilities

test/crucible_adversary/
├── perturbations/
│   ├── character_test.exs
│   ├── word_test.exs
│   └── semantic_test.exs
├── attacks/
│   ├── injection_test.exs
│   ├── jailbreak_test.exs
│   ├── extraction_test.exs
│   └── bias_test.exs
├── evaluation/
│   ├── robustness_test.exs
│   ├── security_test.exs
│   └── stress_test.exs
├── metrics/
│   ├── accuracy_test.exs
│   ├── consistency_test.exs
│   └── asr_test.exs
└── integration/
    ├── pipeline_test.exs
    └── end_to_end_test.exs
```

---

## Implementation Requirements by Module

### Phase 1 (v0.1.0): Foundation - PRIORITY IMPLEMENTATION

This is the core foundation that must be built first.

---

### Module 1: Core Data Structures

**File**: `lib/crucible_adversary/core.ex`

**Purpose**: Define core data structures used throughout the system

**Required Structs**:

```elixir
defmodule CrucibleAdversary.AttackResult do
  @moduledoc """
  Represents the result of an adversarial attack.
  """

  @type t :: %__MODULE__{
    original: String.t(),
    attacked: String.t(),
    attack_type: atom(),
    success: boolean(),
    metadata: map(),
    timestamp: DateTime.t()
  }

  defstruct [
    :original,
    :attacked,
    :attack_type,
    success: false,
    metadata: %{},
    timestamp: nil
  ]
end

defmodule CrucibleAdversary.EvaluationResult do
  @moduledoc """
  Represents the result of a robustness evaluation.
  """

  @type t :: %__MODULE__{
    model: atom() | String.t(),
    test_set_size: non_neg_integer(),
    attack_types: list(atom()),
    metrics: map(),
    vulnerabilities: list(map()),
    timestamp: DateTime.t()
  }

  defstruct [
    :model,
    :test_set_size,
    attack_types: [],
    metrics: %{},
    vulnerabilities: [],
    timestamp: nil
  ]
end

defmodule CrucibleAdversary.Config do
  @moduledoc """
  Configuration management for CrucibleAdversary.
  """

  @type t :: %__MODULE__{
    default_attack_rate: float(),
    max_perturbation_rate: float(),
    random_seed: integer() | nil,
    logging_level: atom(),
    cache_enabled: boolean()
  }

  defstruct [
    default_attack_rate: 0.1,
    max_perturbation_rate: 0.3,
    random_seed: nil,
    logging_level: :info,
    cache_enabled: true
  ]

  def default, do: %__MODULE__{}

  def validate(%__MODULE__{} = config) do
    # Validation logic
    :ok
  end
end
```

**Tests Required**:
- Struct initialization with defaults
- Field validation
- Config validation with valid/invalid values

---

### Module 2: Character-Level Perturbations

**File**: `lib/crucible_adversary/perturbations/character.ex`

**Purpose**: Implement character-level adversarial text perturbations

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Perturbations.Character do
  @moduledoc """
  Character-level perturbation attacks for adversarial text generation.

  Implements:
  - Character swapping (typos)
  - Character deletion
  - Character insertion
  - Homoglyph substitution
  - Keyboard-based typo injection
  """

  alias CrucibleAdversary.AttackResult

  @doc """
  Randomly swaps adjacent characters to simulate typos.

  ## Options
    * `:rate` - Float between 0.0 and 1.0, percentage of characters to swap (default: 0.1)
    * `:seed` - Integer, random seed for reproducibility (default: nil)

  ## Examples

      iex> Character.swap("hello world", rate: 0.2)
      {:ok, %AttackResult{original: "hello world", attacked: "helo wlord", ...}}
  """
  @spec swap(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def swap(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Deletes random characters from text.

  ## Options
    * `:rate` - Float, percentage of characters to delete (default: 0.1)
    * `:preserve_spaces` - Boolean, whether to preserve space characters (default: true)
  """
  @spec delete(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def delete(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Inserts random characters into text.

  ## Options
    * `:rate` - Float, percentage of insertion positions (default: 0.1)
    * `:char_pool` - List of characters to insert from (default: a-z)
  """
  @spec insert(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def insert(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Substitutes characters with visually similar Unicode characters (homoglyphs).

  ## Options
    * `:rate` - Float, percentage of characters to substitute (default: 0.1)
    * `:charset` - Atom, character set to use (:cyrillic, :greek, :all) (default: :all)

  ## Examples

      iex> Character.homoglyph("administrator", charset: :cyrillic)
      {:ok, %AttackResult{attacked: "аdministrator", ...}}  # Cyrillic 'а'
  """
  @spec homoglyph(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def homoglyph(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Injects realistic typos based on keyboard layout.

  ## Options
    * `:rate` - Float, percentage of typo injection (default: 0.1)
    * `:layout` - Atom, keyboard layout (:qwerty, :dvorak) (default: :qwerty)
    * `:typo_types` - List of atoms, types to include [:substitution, :insertion, :deletion, :transposition]
  """
  @spec keyboard_typo(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def keyboard_typo(text, opts \\ []) do
    # Implementation
  end

  # Private helper functions

  defp adjacent_keys(key, :qwerty) do
    # QWERTY keyboard adjacency map
  end

  defp homoglyph_map(charset) do
    # Homoglyph mapping for different charsets
  end

  defp apply_rate(text, rate, apply_fn) do
    # Apply transformation at specified rate
  end
end
```

**Tests Required**:
- `swap/2`: Basic swapping, rate validation, empty string, single character
- `delete/2`: Basic deletion, preserve spaces, rate boundaries
- `insert/2`: Basic insertion, custom char pool, rate validation
- `homoglyph/2`: Cyrillic substitution, Greek substitution, mixed charsets
- `keyboard_typo/2`: QWERTY layout, different typo types, combined typos
- Property tests: Semantic preservation, length constraints, reversibility
- Integration: Chain multiple perturbations

---

### Module 3: Word-Level Perturbations

**File**: `lib/crucible_adversary/perturbations/word.ex`

**Purpose**: Implement word-level adversarial text perturbations

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Perturbations.Word do
  @moduledoc """
  Word-level perturbation attacks.

  Implements:
  - Word deletion
  - Word insertion
  - Synonym replacement
  - Word order shuffling
  """

  alias CrucibleAdversary.AttackResult

  @doc """
  Randomly deletes words from text.

  ## Options
    * `:rate` - Float, percentage of words to delete (default: 0.2)
    * `:strategy` - Atom, deletion strategy (:random, :importance_based) (default: :random)
    * `:preserve_stopwords` - Boolean (default: false)
  """
  @spec delete(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def delete(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Inserts random words into text.

  ## Options
    * `:rate` - Float, percentage of insertion positions (default: 0.2)
    * `:noise_type` - Atom, type of noise (:random_words, :adversarial) (default: :random_words)
    * `:dictionary` - List of words to insert from (default: common English words)
  """
  @spec insert(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def insert(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Replaces words with synonyms while preserving semantic meaning.

  ## Options
    * `:rate` - Float, percentage of words to replace (default: 0.3)
    * `:dictionary` - Atom, dictionary source (:simple, :wordnet) (default: :simple)
    * `:preserve_pos` - Boolean, preserve part of speech (default: true)

  ## Examples

      iex> Word.synonym_replace("The quick brown fox", rate: 0.5)
      {:ok, %AttackResult{attacked: "The rapid brown fox", ...}}
  """
  @spec synonym_replace(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def synonym_replace(text, opts \\ []) do
    # Implementation
  end

  @doc """
  Shuffles word order while attempting to maintain some coherence.

  ## Options
    * `:shuffle_type` - Atom, shuffle strategy (:random, :adjacent_only) (default: :adjacent_only)
    * `:rate` - Float, percentage of words to shuffle (default: 0.2)
  """
  @spec shuffle(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def shuffle(text, opts \\ []) do
    # Implementation
  end

  # Private helpers

  defp tokenize(text) do
    # Split text into words while preserving punctuation context
  end

  defp simple_synonym_map do
    %{
      "quick" => ["fast", "rapid", "swift"],
      "dangerous" => ["hazardous", "risky", "unsafe"],
      # ... more mappings
    }
  end

  defp common_words do
    ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]
  end
end
```

**Tests Required**:
- `delete/2`: Random deletion, stopword preservation, boundary cases
- `insert/2`: Random word insertion, custom dictionary, noise types
- `synonym_replace/2`: Simple dictionary, multiple synonyms, no synonyms available
- `shuffle/2`: Adjacent shuffling, random shuffling, single word
- Property tests: Word count constraints, semantic similarity
- Integration: Combine with character perturbations

---

### Module 4: Metrics - Accuracy

**File**: `lib/crucible_adversary/metrics/accuracy.ex`

**Purpose**: Calculate accuracy-based robustness metrics

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Metrics.Accuracy do
  @moduledoc """
  Accuracy-based robustness metrics.

  Calculates:
  - Accuracy drop under adversarial attacks
  - Robust accuracy
  - Relative accuracy degradation
  """

  @doc """
  Calculates accuracy drop between clean and adversarial results.

  ## Parameters
    * `original_results` - List of tuples {prediction, label, correct?}
    * `attacked_results` - List of tuples {prediction, label, correct?}

  ## Returns
    Map containing:
    * `:original_accuracy` - Accuracy on clean inputs
    * `:attacked_accuracy` - Accuracy on adversarial inputs
    * `:absolute_drop` - Absolute difference
    * `:relative_drop` - Relative percentage drop
    * `:severity` - Atom (:low, :moderate, :high, :critical)

  ## Examples

      iex> Accuracy.drop(original, attacked)
      %{
        original_accuracy: 0.95,
        attacked_accuracy: 0.78,
        absolute_drop: 0.17,
        relative_drop: 0.179,
        severity: :moderate
      }
  """
  @spec drop(list(tuple()), list(tuple())) :: map()
  def drop(original_results, attacked_results) do
    # Implementation
  end

  @doc """
  Calculates robust accuracy (accuracy on adversarial examples only).

  ## Parameters
    * `predictions` - List of predicted values
    * `ground_truth` - List of true labels

  ## Returns
    Float between 0.0 and 1.0
  """
  @spec robust_accuracy(list(), list()) :: float()
  def robust_accuracy(predictions, ground_truth) do
    # Implementation
  end

  # Private helpers

  defp calculate_accuracy(results) do
    # Calculate accuracy from results list
  end

  defp determine_severity(relative_drop) do
    cond do
      relative_drop < 0.05 -> :low
      relative_drop < 0.15 -> :moderate
      relative_drop < 0.30 -> :high
      true -> :critical
    end
  end
end
```

**Tests Required**:
- `drop/2`: Perfect accuracy, zero accuracy, partial degradation
- `robust_accuracy/2`: All correct, all incorrect, mixed results
- Severity classification edge cases
- Empty input handling
- Mismatched list lengths

---

### Module 5: Metrics - Attack Success Rate

**File**: `lib/crucible_adversary/metrics/asr.ex`

**Purpose**: Calculate attack success rates

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Metrics.ASR do
  @moduledoc """
  Attack Success Rate (ASR) metrics.

  Measures the effectiveness of adversarial attacks.
  """

  alias CrucibleAdversary.AttackResult

  @doc """
  Calculates attack success rate.

  ## Parameters
    * `attack_results` - List of AttackResult structs
    * `success_fn` - Function that determines if attack succeeded

  ## Returns
    Map containing:
    * `:overall_asr` - Overall success rate
    * `:by_attack_type` - Success rate by attack type
    * `:total_attacks` - Total number of attacks
    * `:successful_attacks` - Number of successful attacks

  ## Examples

      iex> success_fn = fn result -> result.success end
      iex> ASR.calculate(attack_results, success_fn)
      %{
        overall_asr: 0.23,
        by_attack_type: %{
          character_swap: 0.15,
          word_deletion: 0.31
        },
        total_attacks: 100,
        successful_attacks: 23
      }
  """
  @spec calculate(list(AttackResult.t()), function()) :: map()
  def calculate(attack_results, success_fn) do
    # Implementation
  end

  @doc """
  Calculates query efficiency (success per query).

  ## Parameters
    * `attack_results` - List of attack results with query counts
    * `total_queries` - Total number of queries made

  ## Returns
    Map with efficiency metrics
  """
  @spec query_efficiency(list(map()), non_neg_integer()) :: map()
  def query_efficiency(attack_results, total_queries) do
    # Implementation
  end

  # Private helpers

  defp group_by_type(results) do
    # Group results by attack type
  end

  defp calculate_rate(successful, total) when total > 0 do
    successful / total
  end
  defp calculate_rate(_successful, 0), do: 0.0
end
```

**Tests Required**:
- `calculate/2`: All successful, none successful, mixed
- By attack type grouping
- Custom success functions
- Empty results handling
- `query_efficiency/2`: Various efficiency scenarios

---

### Module 6: Metrics - Semantic Similarity

**File**: `lib/crucible_adversary/metrics/consistency.ex`

**Purpose**: Measure semantic similarity and output consistency

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Metrics.Consistency do
  @moduledoc """
  Consistency and semantic similarity metrics.

  Measures how similar outputs are between original and perturbed inputs.
  """

  @doc """
  Calculates semantic similarity between two texts.

  ## Options
    * `:method` - Atom, similarity method (:jaccard, :cosine, :edit_distance) (default: :jaccard)

  ## Returns
    Float between 0.0 (completely different) and 1.0 (identical)

  ## Examples

      iex> Consistency.semantic_similarity("the cat sat", "the feline sat", method: :jaccard)
      0.67
  """
  @spec semantic_similarity(String.t(), String.t(), keyword()) :: float()
  def semantic_similarity(text1, text2, opts \\ []) do
    # Implementation
  end

  @doc """
  Calculates output consistency across original and perturbed inputs.

  ## Parameters
    * `original_outputs` - List of original model outputs
    * `perturbed_outputs` - List of perturbed model outputs
    * `opts` - Options including :method

  ## Returns
    Map containing:
    * `:mean_consistency` - Average consistency score
    * `:median_consistency` - Median consistency score
    * `:std_consistency` - Standard deviation
    * `:min` - Minimum consistency
    * `:max` - Maximum consistency
  """
  @spec consistency(list(String.t()), list(String.t()), keyword()) :: map()
  def consistency(original_outputs, perturbed_outputs, opts \\ []) do
    # Implementation
  end

  # Private helpers

  defp jaccard_similarity(text1, text2) do
    # Jaccard similarity: |A ∩ B| / |A ∪ B|
  end

  defp edit_distance(text1, text2) do
    # Levenshtein distance
  end

  defp calculate_stats(similarities) do
    # Calculate mean, median, std, min, max
  end
end
```

**Tests Required**:
- `semantic_similarity/3`: Identical texts, completely different, partial similarity
- Jaccard similarity calculation
- Edit distance calculation
- `consistency/3`: Perfect consistency, no consistency, partial
- Statistical calculations: mean, median, std
- Empty input handling

---

### Module 7: Robustness Evaluation

**File**: `lib/crucible_adversary/evaluation/robustness.ex`

**Purpose**: Core robustness evaluation orchestration

**Required Functions**:

```elixir
defmodule CrucibleAdversary.Evaluation.Robustness do
  @moduledoc """
  Core robustness evaluation framework.

  Orchestrates adversarial attacks and evaluation metrics.
  """

  alias CrucibleAdversary.{AttackResult, EvaluationResult}
  alias CrucibleAdversary.Perturbations.{Character, Word}
  alias CrucibleAdversary.Metrics.{Accuracy, ASR, Consistency}

  @doc """
  Evaluates model robustness across multiple attack types.

  ## Parameters
    * `model` - Model module or function that takes input and returns output
    * `test_set` - List of {input, expected_output} tuples
    * `opts` - Options including:
      * `:attacks` - List of attack types to test (default: [:character_swap, :word_deletion])
      * `:metrics` - List of metrics to calculate (default: [:accuracy_drop, :asr])
      * `:attack_opts` - Keyword list of options for attacks

  ## Returns
    {:ok, %EvaluationResult{}} or {:error, reason}

  ## Examples

      iex> Robustness.evaluate(MyModel, test_set, attacks: [:character_swap, :word_deletion])
      {:ok, %EvaluationResult{
        metrics: %{
          accuracy_drop: %{absolute_drop: 0.17, ...},
          asr: %{overall_asr: 0.23, ...}
        },
        vulnerabilities: [...]
      }}
  """
  @spec evaluate(module() | function(), list(tuple()), keyword()) ::
    {:ok, EvaluationResult.t()} | {:error, term()}
  def evaluate(model, test_set, opts \\ []) do
    # Implementation:
    # 1. Generate attacks for each type
    # 2. Run model on original and attacked inputs
    # 3. Calculate requested metrics
    # 4. Identify vulnerabilities
    # 5. Return evaluation result
  end

  @doc """
  Evaluates a single input with multiple attacks.

  ## Returns
    List of AttackResult structs
  """
  @spec evaluate_single(module() | function(), {String.t(), any()}, keyword()) ::
    list(AttackResult.t())
  def evaluate_single(model, {input, expected}, opts \\ []) do
    # Implementation
  end

  # Private helpers

  defp generate_attacks(input, attack_types, opts) do
    # Generate attacks based on specified types
  end

  defp run_model(model, input) when is_function(model) do
    model.(input)
  end
  defp run_model(model, input) when is_atom(model) do
    apply(model, :predict, [input])
  end

  defp identify_vulnerabilities(metrics, threshold) do
    # Identify areas where model is vulnerable
  end

  defp attack_module(:character_swap), do: {Character, :swap}
  defp attack_module(:character_delete), do: {Character, :delete}
  defp attack_module(:word_deletion), do: {Word, :delete}
  defp attack_module(:word_insertion), do: {Word, :insert}
  defp attack_module(:synonym_replacement), do: {Word, :synonym_replace}
end
```

**Tests Required**:
- `evaluate/3`: Full evaluation with multiple attacks
- `evaluate_single/3`: Single input evaluation
- Different model types: function, module
- Attack generation for each type
- Metric calculation integration
- Vulnerability identification
- Error handling: invalid model, empty test set
- Integration: End-to-end evaluation pipeline

---

### Module 8: Main API

**File**: `lib/crucible_adversary.ex` (Replace skeleton)

**Purpose**: Main public API and entry point

**Required Functions**:

```elixir
defmodule CrucibleAdversary do
  @moduledoc """
  CrucibleAdversary - Adversarial Testing and Robustness Evaluation Framework

  Main API for adversarial testing of AI models. Provides a comprehensive suite
  of attacks, evaluation metrics, and robustness testing capabilities.

  ## Quick Start

      # Character-level attack
      {:ok, result} = CrucibleAdversary.attack("Hello world", type: :character_swap)

      # Evaluate model robustness
      {:ok, eval} = CrucibleAdversary.evaluate(MyModel, test_set,
        attacks: [:character_swap, :word_deletion],
        metrics: [:accuracy_drop, :asr]
      )

  ## Modules

  - `CrucibleAdversary.Perturbations.Character` - Character-level attacks
  - `CrucibleAdversary.Perturbations.Word` - Word-level attacks
  - `CrucibleAdversary.Evaluation.Robustness` - Robustness evaluation
  - `CrucibleAdversary.Metrics.*` - Various robustness metrics
  """

  alias CrucibleAdversary.{AttackResult, EvaluationResult, Config}
  alias CrucibleAdversary.Perturbations.{Character, Word}
  alias CrucibleAdversary.Evaluation.Robustness

  @doc """
  Performs a single adversarial attack on input text.

  ## Parameters
    * `input` - Text to attack
    * `opts` - Options including:
      * `:type` - Attack type (required)
      * Other attack-specific options

  ## Returns
    {:ok, %AttackResult{}} or {:error, reason}

  ## Examples

      iex> CrucibleAdversary.attack("Hello world", type: :character_swap, rate: 0.2)
      {:ok, %AttackResult{original: "Hello world", attacked: "Hlelo wrold", ...}}
  """
  @spec attack(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def attack(input, opts) do
    attack_type = Keyword.fetch!(opts, :type)

    case attack_type do
      :character_swap -> Character.swap(input, opts)
      :character_delete -> Character.delete(input, opts)
      :character_insert -> Character.insert(input, opts)
      :homoglyph -> Character.homoglyph(input, opts)
      :keyboard_typo -> Character.keyboard_typo(input, opts)
      :word_deletion -> Word.delete(input, opts)
      :word_insertion -> Word.insert(input, opts)
      :synonym_replacement -> Word.synonym_replace(input, opts)
      :word_shuffle -> Word.shuffle(input, opts)
      _ -> {:error, {:unknown_attack_type, attack_type}}
    end
  end

  @doc """
  Performs multiple attacks on a batch of inputs.

  ## Parameters
    * `inputs` - List of texts to attack
    * `opts` - Options including:
      * `:types` - List of attack types (default: [:character_swap])

  ## Returns
    {:ok, list of AttackResult structs} or {:error, reason}
  """
  @spec attack_batch(list(String.t()), keyword()) ::
    {:ok, list(AttackResult.t())} | {:error, term()}
  def attack_batch(inputs, opts \\ []) do
    # Implementation
  end

  @doc """
  Evaluates model robustness against adversarial attacks.

  Delegates to CrucibleAdversary.Evaluation.Robustness.evaluate/3

  ## Parameters
    * `model` - Model module or function
    * `test_set` - List of {input, expected_output} tuples
    * `opts` - Evaluation options

  ## Returns
    {:ok, %EvaluationResult{}} or {:error, reason}
  """
  @spec evaluate(module() | function(), list(tuple()), keyword()) ::
    {:ok, EvaluationResult.t()} | {:error, term()}
  def evaluate(model, test_set, opts \\ []) do
    Robustness.evaluate(model, test_set, opts)
  end

  @doc """
  Returns the current configuration.
  """
  @spec config() :: Config.t()
  def config do
    Application.get_env(:crucible_adversary, :config, Config.default())
  end

  @doc """
  Sets configuration.
  """
  @spec configure(Config.t() | keyword()) :: :ok
  def configure(%Config{} = config) do
    Application.put_env(:crucible_adversary, :config, config)
  end
  def configure(opts) when is_list(opts) do
    config = struct(Config.default(), opts)
    configure(config)
  end

  @doc """
  Returns version information.
  """
  @spec version() :: String.t()
  def version do
    Application.spec(:crucible_adversary, :vsn) |> to_string()
  end
end
```

**Tests Required**:
- `attack/2`: All attack types, error handling
- `attack_batch/2`: Multiple inputs, mixed success/failure
- `evaluate/3`: Full integration test
- `config/0` and `configure/1`: Configuration management
- `version/0`: Version string

---

## TDD Process & Workflow

### Red-Green-Refactor Cycle

For **EVERY** function implementation, follow this strict process:

#### 1. RED: Write Failing Tests First

```elixir
# test/crucible_adversary/perturbations/character_test.exs

defmodule CrucibleAdversary.Perturbations.CharacterTest do
  use ExUnit.Case, async: true

  alias CrucibleAdversary.Perturbations.Character
  alias CrucibleAdversary.AttackResult

  describe "swap/2" do
    test "swaps adjacent characters at specified rate" do
      # Given
      input = "hello"

      # When
      {:ok, result} = Character.swap(input, rate: 0.4, seed: 42)

      # Then
      assert %AttackResult{} = result
      assert result.original == "hello"
      assert result.attacked != "hello"
      assert result.attack_type == :character_swap
      assert String.length(result.attacked) == String.length(input)
    end

    test "returns error for invalid rate" do
      assert {:error, :invalid_rate} = Character.swap("test", rate: 1.5)
      assert {:error, :invalid_rate} = Character.swap("test", rate: -0.1)
    end

    test "handles empty string" do
      {:ok, result} = Character.swap("", rate: 0.1)
      assert result.attacked == ""
    end

    test "handles single character" do
      {:ok, result} = Character.swap("a", rate: 0.1)
      assert result.attacked == "a"
    end
  end
end
```

**Run**: `mix test` - **Tests should FAIL** (Red)

#### 2. GREEN: Implement Minimal Code to Pass

```elixir
defmodule CrucibleAdversary.Perturbations.Character do
  alias CrucibleAdversary.AttackResult

  def swap(text, opts \\ []) do
    rate = Keyword.get(opts, :rate, 0.1)
    seed = Keyword.get(opts, :seed)

    with :ok <- validate_rate(rate),
         attacked <- apply_swap(text, rate, seed) do
      {:ok, %AttackResult{
        original: text,
        attacked: attacked,
        attack_type: :character_swap,
        success: attacked != text,
        metadata: %{rate: rate},
        timestamp: DateTime.utc_now()
      }}
    end
  end

  defp validate_rate(rate) when rate >= 0.0 and rate <= 1.0, do: :ok
  defp validate_rate(_), do: {:error, :invalid_rate}

  defp apply_swap("", _rate, _seed), do: ""
  defp apply_swap(text, rate, seed) when byte_size(text) == 1, do: text
  defp apply_swap(text, rate, seed) do
    if seed, do: :rand.seed(:exsplus, {seed, seed, seed})

    text
    |> String.graphemes()
    |> swap_pairs(rate)
    |> Enum.join()
  end

  defp swap_pairs(chars, rate) do
    # Implementation
  end
end
```

**Run**: `mix test` - **Tests should PASS** (Green)

#### 3. REFACTOR: Clean Up

```elixir
# Refactor for clarity and efficiency
defp swap_pairs(chars, rate) do
  chars
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.map(fn pair -> maybe_swap(pair, rate) end)
  |> List.flatten()
  |> handle_last_char(chars)
end

defp maybe_swap([a, b], rate) do
  if :rand.uniform() < rate, do: [b, a], else: [a, b]
end

defp handle_last_char(result, original) do
  # Handle odd-length strings
end
```

**Run**: `mix test` - **Tests should still PASS**

#### 4. VERIFY: Check Quality Gates

```bash
# Run all quality checks
mix test                    # All tests pass
mix compile --warnings-as-errors  # Zero warnings
mix dialyzer                # Zero dialyzer errors
mix format --check-formatted      # Code formatted
mix credo --strict          # Code quality (optional)
```

**ONLY PROCEED** when all checks pass.

### Test Categories Required

#### Unit Tests
- Test each function in isolation
- Test edge cases and error conditions
- Test boundary values
- Test type validation

#### Property-Based Tests

```elixir
use ExUnitProperties

property "swap preserves string length" do
  check all text <- string(:alphanumeric, min_length: 1),
            rate <- float(min: 0.0, max: 1.0) do
    {:ok, result} = Character.swap(text, rate: rate, seed: 123)
    assert String.length(result.attacked) == String.length(text)
  end
end

property "semantic similarity above threshold" do
  check all text <- string(:alphanumeric, min_length: 5),
            rate <- float(min: 0.0, max: 0.3) do
    {:ok, result} = Character.swap(text, rate: rate)
    similarity = calculate_similarity(result.original, result.attacked)
    assert similarity >= 0.7
  end
end
```

#### Integration Tests

```elixir
# test/crucible_adversary/integration/pipeline_test.exs

defmodule CrucibleAdversary.Integration.PipelineTest do
  use ExUnit.Case

  test "full evaluation pipeline" do
    # Setup mock model
    model = fn input -> String.upcase(input) end

    test_set = [
      {"hello world", "HELLO WORLD"},
      {"test input", "TEST INPUT"}
    ]

    # Execute full evaluation
    {:ok, result} = CrucibleAdversary.evaluate(
      model,
      test_set,
      attacks: [:character_swap, :word_deletion],
      metrics: [:accuracy_drop, :asr]
    )

    # Verify result structure
    assert %EvaluationResult{} = result
    assert Map.has_key?(result.metrics, :accuracy_drop)
    assert Map.has_key?(result.metrics, :asr)
  end
end
```

---

## Integration Points

### With CrucibleBench

```elixir
# Future integration for statistical comparison
defmodule CrucibleAdversary.Integration.Bench do
  def compare_robustness(models, test_set, opts \\ []) do
    # Evaluate each model
    results = Enum.map(models, fn model ->
      CrucibleAdversary.evaluate(model, test_set, opts)
    end)

    # Extract robustness scores
    scores = Enum.map(results, & &1.metrics.robustness_score)

    # Use CrucibleBench for statistical comparison
    CrucibleBench.compare_multiple(scores, labels: model_names)
  end
end
```

### With Crucible Core

```elixir
# Future pipeline integration
defmodule CrucibleAdversary.Integration.Pipeline do
  def adversarial_pipeline(test_data, opts \\ []) do
    Crucible.Pipeline.new()
    |> Crucible.Pipeline.add_stage(:attack_generation, fn batch ->
      CrucibleAdversary.attack_batch(batch, opts)
    end)
    |> Crucible.Pipeline.add_stage(:robustness_eval, fn attacked ->
      CrucibleAdversary.Evaluation.Robustness.evaluate(attacked)
    end)
    |> Crucible.Pipeline.run(test_data)
  end
end
```

### External Model Adapters (Future)

```elixir
defmodule CrucibleAdversary.Adapters.OpenAI do
  @behaviour CrucibleAdversary.ModelAdapter

  def predict(input, opts \\ []) do
    # OpenAI API call
  end
end
```

---

## Quality Gates & Success Criteria

### Compilation Quality Gates

**ZERO TOLERANCE** for:
- Compilation warnings
- Dialyzer errors
- Formatting issues

```bash
# Must all pass before commit
mix compile --warnings-as-errors
mix dialyzer
mix format --check-formatted
```

### Test Quality Gates

**Required Coverage**: 80% minimum (target 90%+)

```bash
mix test --cover
mix test.coverage
```

**Test Requirements**:
- All unit tests passing
- All property tests passing
- All integration tests passing
- All edge cases covered
- All error paths tested

### Documentation Quality Gates

**Required**:
- Module docstrings with overview and examples
- Function docstrings with @spec, parameters, returns, examples
- README with quick start and usage
- CHANGELOG updated for changes

```elixir
# Example complete documentation
defmodule CrucibleAdversary.Perturbations.Character do
  @moduledoc """
  Character-level perturbation attacks.

  Provides various character-level transformations for adversarial
  testing including swaps, deletions, insertions, and homoglyphs.

  ## Examples

      iex> Character.swap("hello", rate: 0.2)
      {:ok, %AttackResult{attacked: "hlelo", ...}}
  """

  @doc """
  Swaps adjacent characters to simulate typos.

  ## Parameters
    * `text` - String to perturb
    * `opts` - Options:
      * `:rate` - Float 0.0-1.0, percentage to swap (default: 0.1)
      * `:seed` - Integer for reproducibility (optional)

  ## Returns
    * `{:ok, %AttackResult{}}` - Success with attack result
    * `{:error, reason}` - Error with reason

  ## Examples

      iex> Character.swap("hello world", rate: 0.2, seed: 42)
      {:ok, %AttackResult{original: "hello world", attacked: "hlelo wrold"}}

      iex> Character.swap("test", rate: 1.5)
      {:error, :invalid_rate}
  """
  @spec swap(String.t(), keyword()) :: {:ok, AttackResult.t()} | {:error, term()}
  def swap(text, opts \\ []) do
    # ...
  end
end
```

### Performance Quality Gates

**Target Benchmarks** (Phase 1):
- Single attack generation: < 10ms
- Batch of 100 attacks: < 500ms
- Full evaluation (100 samples, 3 attacks): < 5s

```elixir
# Benchmarking with Benchee (add to dev deps if needed)
Benchee.run(%{
  "character_swap" => fn -> Character.swap(@sample_text, rate: 0.1) end,
  "word_deletion" => fn -> Word.delete(@sample_text, rate: 0.2) end
})
```

### Code Quality Gates

**Credo** (optional but recommended):

```bash
mix credo --strict
```

**Quality Standards**:
- No code duplication
- Functions < 20 lines (guideline, not hard rule)
- Modules focused on single responsibility
- Descriptive variable names
- No magic numbers (use module attributes)

### Pre-Commit Checklist

Before committing code, verify:

- [ ] All tests pass (`mix test`)
- [ ] No compilation warnings (`mix compile --warnings-as-errors`)
- [ ] No dialyzer errors (`mix dialyzer`)
- [ ] Code formatted (`mix format`)
- [ ] Test coverage ≥ 80% (`mix test --cover`)
- [ ] Documentation complete (module and function docs)
- [ ] CHANGELOG updated (if applicable)
- [ ] No `IO.inspect` or debug statements left in code
- [ ] All TODOs addressed or documented

---

## Example Usage Patterns

### Basic Attack Usage

```elixir
# Single character-level attack
{:ok, result} = CrucibleAdversary.attack(
  "The quick brown fox",
  type: :character_swap,
  rate: 0.15
)

IO.puts("Original: #{result.original}")
IO.puts("Attacked: #{result.attacked}")
# => Original: The quick brown fox
# => Attacked: The qiuck borwn fox

# Word-level attack
{:ok, result} = CrucibleAdversary.attack(
  "This is a dangerous action",
  type: :synonym_replacement,
  rate: 0.5
)

IO.puts("Attacked: #{result.attacked}")
# => Attacked: This is a hazardous action
```

### Batch Attack Usage

```elixir
inputs = [
  "Hello world",
  "Test input",
  "Another example"
]

{:ok, results} = CrucibleAdversary.attack_batch(
  inputs,
  types: [:character_swap, :word_deletion],
  rate: 0.2
)

Enum.each(results, fn result ->
  IO.puts("#{result.attack_type}: #{result.attacked}")
end)
```

### Model Robustness Evaluation

```elixir
# Define a simple model (function or module)
defmodule SimpleClassifier do
  def predict(input) do
    if String.contains?(input, "positive") do
      :positive
    else
      :negative
    end
  end
end

# Create test set
test_set = [
  {"This is positive", :positive},
  {"This is negative", :negative},
  {"Another positive example", :positive}
]

# Evaluate robustness
{:ok, evaluation} = CrucibleAdversary.evaluate(
  SimpleClassifier,
  test_set,
  attacks: [:character_swap, :word_deletion, :synonym_replacement],
  metrics: [:accuracy_drop, :asr, :consistency]
)

# Inspect results
IO.inspect(evaluation.metrics.accuracy_drop)
# => %{
#   original_accuracy: 1.0,
#   attacked_accuracy: 0.67,
#   absolute_drop: 0.33,
#   relative_drop: 0.33,
#   severity: :high
# }

IO.inspect(evaluation.metrics.asr)
# => %{
#   overall_asr: 0.33,
#   by_attack_type: %{
#     character_swap: 0.20,
#     word_deletion: 0.40,
#     synonym_replacement: 0.33
#   }
# }
```

### Configuration

```elixir
# Set global configuration
CrucibleAdversary.configure(
  default_attack_rate: 0.15,
  max_perturbation_rate: 0.4,
  random_seed: 42,
  logging_level: :debug
)

# Use in attacks
{:ok, result} = CrucibleAdversary.attack("test", type: :character_swap)
# Uses default_attack_rate: 0.15
```

---

## Performance Requirements

### Phase 1 Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Single character attack | < 10ms | 95th percentile |
| Single word attack | < 15ms | 95th percentile |
| Batch 100 attacks | < 500ms | Mean |
| Metric calculation | < 5ms | Mean |
| Full evaluation (100 samples, 3 attacks) | < 5s | Mean |

### Optimization Strategies

1. **Avoid Unnecessary String Copying**
   - Use string slicing where possible
   - Consider binaries for large texts

2. **Batch Operations**
   - Process multiple attacks in parallel where safe
   - Use `Task.async_stream` for independent operations

3. **Caching**
   - Cache homoglyph mappings
   - Cache synonym dictionaries
   - Use ETS for shared lookup tables

4. **Randomness**
   - Seed once per batch, not per operation
   - Use efficient PRNG (`:rand` module)

### Performance Testing

```elixir
# Add to test suite
defmodule CrucibleAdversary.PerformanceTest do
  use ExUnit.Case

  @tag :performance
  test "character swap performance" do
    text = String.duplicate("hello world ", 100)

    {time, {:ok, _result}} = :timer.tc(fn ->
      CrucibleAdversary.attack(text, type: :character_swap)
    end)

    # Assert < 10ms (10_000 microseconds)
    assert time < 10_000, "Attack took #{time}μs, expected < 10,000μs"
  end
end

# Run with: mix test --only performance
```

---

## Testing Requirements

### Unit Test Structure

```elixir
defmodule CrucibleAdversary.ModuleNameTest do
  use ExUnit.Case, async: true

  alias CrucibleAdversary.ModuleName

  describe "function_name/arity" do
    test "basic functionality" do
      # Given (setup)
      # When (execute)
      # Then (assert)
    end

    test "edge case: empty input" do
      # ...
    end

    test "error handling: invalid parameters" do
      # ...
    end
  end
end
```

### Property-Based Test Structure

```elixir
defmodule CrucibleAdversary.Properties.CharacterTest do
  use ExUnit.Case
  use ExUnitProperties

  alias CrucibleAdversary.Perturbations.Character

  property "swap preserves string length" do
    check all text <- string(:alphanumeric, min_length: 1),
              rate <- float(min: 0.0, max: 1.0) do
      {:ok, result} = Character.swap(text, rate: rate, seed: 123)
      assert String.length(result.attacked) == String.length(text)
    end
  end

  property "swap changes at most rate% of characters" do
    check all text <- string(:alphanumeric, min_length: 10),
              rate <- float(min: 0.0, max: 0.5) do
      {:ok, result} = Character.swap(text, rate: rate, seed: 123)

      changed = count_differences(result.original, result.attacked)
      max_changes = ceil(String.length(text) * rate)

      assert changed <= max_changes
    end
  end
end
```

### Integration Test Structure

```elixir
defmodule CrucibleAdversary.Integration.EvaluationTest do
  use ExUnit.Case

  @moduletag :integration

  setup do
    # Setup mock model
    model = fn input ->
      cond do
        String.contains?(input, "positive") -> :positive
        String.contains?(input, "negative") -> :negative
        true -> :neutral
      end
    end

    test_set = [
      {"This is positive", :positive},
      {"This is negative", :negative},
      {"Neutral statement", :neutral}
    ]

    {:ok, model: model, test_set: test_set}
  end

  test "full evaluation pipeline", %{model: model, test_set: test_set} do
    {:ok, result} = CrucibleAdversary.evaluate(
      model,
      test_set,
      attacks: [:character_swap, :word_deletion],
      metrics: [:accuracy_drop, :asr]
    )

    assert %EvaluationResult{} = result
    assert result.test_set_size == 3
    assert length(result.attack_types) == 2
    assert Map.has_key?(result.metrics, :accuracy_drop)
    assert Map.has_key?(result.metrics, :asr)
  end
end
```

### Test Coverage Requirements

**Minimum Coverage**: 80% for Phase 1

**Priority Coverage**:
1. **Public API Functions**: 100% coverage
2. **Core Logic**: 95%+ coverage
3. **Error Paths**: 90%+ coverage
4. **Edge Cases**: Comprehensive coverage

**Coverage Report**:
```bash
mix test --cover
mix test.coverage
```

### Test Organization

```
test/
├── test_helper.exs                      # Test configuration
├── support/                              # Test helpers and fixtures
│   ├── fixtures.ex                      # Test data
│   └── mock_models.ex                   # Mock models for testing
├── crucible_adversary/
│   ├── perturbations/
│   │   ├── character_test.exs
│   │   ├── word_test.exs
│   │   └── semantic_test.exs
│   ├── metrics/
│   │   ├── accuracy_test.exs
│   │   ├── asr_test.exs
│   │   └── consistency_test.exs
│   ├── evaluation/
│   │   └── robustness_test.exs
│   └── core_test.exs
├── properties/                           # Property-based tests
│   ├── character_properties_test.exs
│   └── word_properties_test.exs
└── integration/                          # Integration tests
    ├── pipeline_test.exs
    └── end_to_end_test.exs
```

---

## Implementation Checklist

### Week 1-2: Core Infrastructure

- [ ] Project structure finalized
- [ ] Core data structures implemented
  - [ ] `AttackResult` struct
  - [ ] `EvaluationResult` struct
  - [ ] `Config` struct
- [ ] Configuration system
  - [ ] Default config
  - [ ] Config validation
  - [ ] Runtime config updates
- [ ] Error handling framework
  - [ ] Common error types defined
  - [ ] Error formatting helpers
- [ ] Logging infrastructure
  - [ ] Logger configuration
  - [ ] Log levels
- [ ] Test infrastructure
  - [ ] `test_helper.exs` configured
  - [ ] Test support modules
  - [ ] Fixtures and mock data

### Week 2-3: Character & Word Perturbations

**Character Module**:
- [ ] `Character.swap/2` - TDD complete
- [ ] `Character.delete/2` - TDD complete
- [ ] `Character.insert/2` - TDD complete
- [ ] `Character.homoglyph/2` - TDD complete
  - [ ] Homoglyph mappings (Cyrillic, Greek)
- [ ] `Character.keyboard_typo/2` - TDD complete
  - [ ] QWERTY adjacency map
- [ ] Character module: 100% test coverage
- [ ] Character module: Zero warnings
- [ ] Character module: Zero dialyzer errors

**Word Module**:
- [ ] `Word.delete/2` - TDD complete
- [ ] `Word.insert/2` - TDD complete
  - [ ] Common word dictionary
- [ ] `Word.synonym_replace/2` - TDD complete
  - [ ] Simple synonym dictionary
- [ ] `Word.shuffle/2` - TDD complete
- [ ] Word module: 100% test coverage
- [ ] Word module: Zero warnings
- [ ] Word module: Zero dialyzer errors

### Week 3: Metrics Implementation

**Accuracy Metrics**:
- [ ] `Accuracy.drop/2` - TDD complete
- [ ] `Accuracy.robust_accuracy/2` - TDD complete
- [ ] Accuracy module: 100% test coverage

**ASR Metrics**:
- [ ] `ASR.calculate/2` - TDD complete
- [ ] `ASR.query_efficiency/2` - TDD complete
- [ ] ASR module: 100% test coverage

**Consistency Metrics**:
- [ ] `Consistency.semantic_similarity/3` - TDD complete
  - [ ] Jaccard similarity
  - [ ] Edit distance
- [ ] `Consistency.consistency/3` - TDD complete
- [ ] Consistency module: 100% test coverage

### Week 4: Evaluation & API

**Robustness Evaluation**:
- [ ] `Robustness.evaluate/3` - TDD complete
- [ ] `Robustness.evaluate_single/3` - TDD complete
- [ ] Attack type routing
- [ ] Vulnerability identification
- [ ] Robustness module: 100% test coverage

**Main API**:
- [ ] `CrucibleAdversary.attack/2` - TDD complete
- [ ] `CrucibleAdversary.attack_batch/2` - TDD complete
- [ ] `CrucibleAdversary.evaluate/3` - TDD complete
- [ ] `CrucibleAdversary.config/0` - TDD complete
- [ ] `CrucibleAdversary.configure/1` - TDD complete
- [ ] `CrucibleAdversary.version/0` - TDD complete
- [ ] Main module: 100% test coverage

### Week 4: Documentation & Polish

- [ ] README.md complete with:
  - [ ] Quick start guide
  - [ ] Installation instructions
  - [ ] Basic usage examples
  - [ ] API overview
- [ ] ExDoc documentation:
  - [ ] All modules documented
  - [ ] All public functions documented
  - [ ] Examples in docstrings
  - [ ] Generated docs reviewed
- [ ] CHANGELOG.md for v0.1.0
- [ ] Integration tests complete
- [ ] Property-based tests added
- [ ] Performance benchmarks run

### Week 4: Final Quality Gates

- [ ] **All tests passing** (100%)
- [ ] **Test coverage ≥ 80%** (target 90%+)
- [ ] **Zero compilation warnings**
- [ ] **Zero dialyzer errors**
- [ ] **Code formatted** (`mix format`)
- [ ] **Credo checks pass** (if using)
- [ ] **Documentation complete**
- [ ] **Performance benchmarks meet targets**
- [ ] **CI/CD pipeline green** (if configured)
- [ ] **Ready for v0.1.0 release**

---

## Success Criteria Summary

### Functional Success Criteria

**Phase 1 (v0.1.0) is complete when**:

1. ✅ **Core perturbations working**:
   - Character-level: swap, delete, insert, homoglyph, keyboard typo
   - Word-level: delete, insert, synonym replace, shuffle

2. ✅ **Metrics implemented**:
   - Accuracy drop calculation
   - Attack success rate (ASR)
   - Semantic similarity (Jaccard, edit distance)
   - Output consistency

3. ✅ **Evaluation framework**:
   - Single input evaluation
   - Batch evaluation
   - Multi-attack evaluation
   - Vulnerability identification

4. ✅ **Main API functional**:
   - `attack/2` for single attacks
   - `attack_batch/2` for batch attacks
   - `evaluate/3` for robustness evaluation
   - Configuration management

### Technical Success Criteria

1. ✅ **Test Coverage**: ≥ 80% (target 90%+)
2. ✅ **Compilation**: Zero warnings
3. ✅ **Dialyzer**: Zero type errors
4. ✅ **Code Quality**: Clean, readable, well-documented
5. ✅ **Performance**: Meets benchmark targets
6. ✅ **Documentation**: Complete and accurate

### Release Readiness Criteria

1. ✅ **README**: Quick start, examples, feature list
2. ✅ **CHANGELOG**: v0.1.0 changes documented
3. ✅ **ExDoc**: Generated documentation reviewed
4. ✅ **Examples**: Working code examples provided
5. ✅ **Tests**: All passing, comprehensive coverage
6. ✅ **CI**: Pipeline configured and green (if applicable)

---

## Notes & Reminders

### Development Philosophy

- **TDD First**: No production code without failing tests first
- **Quality Over Speed**: Take time to get it right
- **Simplicity**: Start simple, refactor to complex only when needed
- **Documentation**: Code is read more than written
- **Performance**: Measure before optimizing

### Common Pitfalls to Avoid

1. **Skipping Tests**: Always write tests first
2. **Ignoring Warnings**: Fix all warnings immediately
3. **Poor Error Handling**: Always return `{:ok, result}` or `{:error, reason}`
4. **Magic Numbers**: Use module attributes for constants
5. **Over-Engineering**: Build what's needed, not what might be needed

### When Stuck

1. **Review Documentation**: Re-read architecture and attack library docs
2. **Check Examples**: Look at usage patterns in this prompt
3. **Write Tests**: Often clarifies what needs to be built
4. **Start Simple**: Implement basic version, refactor later
5. **Ask Questions**: Better to clarify than assume

### Resources

- **Elixir Docs**: https://hexdocs.pm/elixir/
- **ExUnit Guide**: https://hexdocs.pm/ex_unit/
- **Dialyzer**: https://hexdocs.pm/dialyxir/
- **Property Testing**: https://hexdocs.pm/stream_data/

---

## Final Checklist Before Submission

Before considering Phase 1 complete:

- [ ] Run full test suite: `mix test`
- [ ] Check coverage: `mix test --cover`
- [ ] Compile with warnings: `mix compile --warnings-as-errors`
- [ ] Run dialyzer: `mix dialyzer`
- [ ] Format code: `mix format`
- [ ] Generate docs: `mix docs`
- [ ] Review README for accuracy
- [ ] Update CHANGELOG with v0.1.0 changes
- [ ] Tag version in git: `git tag v0.1.0`
- [ ] Create release notes
- [ ] Celebrate! 🎉

---

**END OF CRUCIBLE ADVERSARY BUILDOUT PROMPT v0.1.0**

This prompt contains everything needed to implement Phase 1 of CrucibleAdversary following TDD principles with strict quality gates. Use this as your complete implementation guide.
