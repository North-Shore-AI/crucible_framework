# Contributing to Elixir AI Research Framework

Thank you for your interest in contributing to the Elixir AI Research Framework! This document provides comprehensive guidelines for contributing code, documentation, research, and community engagement.

**Table of Contents**

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Review Guidelines](#review-guidelines)
- [Community Norms](#community-norms)
- [Research Collaboration](#research-collaboration)
- [Library-Specific Guidelines](#library-specific-guidelines)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)
- [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of:

- Age, body size, disability, ethnicity, gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behaviors include:**

- Using welcoming and inclusive language
- Respecting differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**

- Trolling, insulting/derogatory comments, personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate action in response to unacceptable behavior.

**Contact:** research@example.com

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Elixir 1.14+** and **Erlang/OTP 25+** installed
2. **Git** for version control
3. **PostgreSQL 14+** (optional, for testing persistent storage)
4. **Text editor or IDE** with Elixir support (VS Code, IntelliJ, Emacs, Vim)

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/elixir_ai_research.git
cd elixir_ai_research

# Add upstream remote
git remote add upstream https://github.com/nshkrdotcom/elixir_ai_research.git

# Verify remotes
git remote -v
```

### Install Dependencies

```bash
# Install all dependencies
mix deps.get

# Compile all apps
mix compile

# Run tests to verify setup
mix test
```

### Development Environment Setup

**Recommended VS Code Extensions:**

```json
{
  "recommendations": [
    "jakebecker.elixir-ls",
    "phoenixframework.phoenix",
    "bungcip.better-toml",
    "redhat.vscode-yaml"
  ]
}
```

**Configure ElixirLS:**

Create `.vscode/settings.json`:

```json
{
  "elixirLS.dialyzerEnabled": true,
  "elixirLS.fetchDeps": false,
  "elixirLS.suggestSpecs": true
}
```

---

## Ways to Contribute

### 1. Code Contributions

**Areas needing help:**

- **New voting strategies** in Ensemble library
- **Additional hedging strategies** (e.g., multi-armed bandit)
- **More statistical tests** in Bench library
- **New dataset integrations** (BigBench, HELM, etc.)
- **Performance optimizations**
- **Bug fixes**

### 2. Documentation

- **Improve existing guides** with examples and clarifications
- **Create tutorials** for common use cases
- **Write blog posts** about your research using the framework
- **Translate documentation** to other languages
- **Fix typos and improve clarity**

### 3. Testing

- **Add test cases** to improve coverage
- **Write property-based tests** using StreamData
- **Create integration tests** for multi-library interactions
- **Performance benchmarks** and regression tests

### 4. Research Contributions

- **Share experimental results** from using the framework
- **Propose new hypotheses** for testing
- **Validate published results** (replication studies)
- **Contribute datasets** or preprocessing pipelines

### 5. Community Support

- **Answer questions** in GitHub Discussions
- **Review pull requests** from other contributors
- **Report bugs** with detailed reproduction steps
- **Share use cases** and success stories

---

## Development Workflow

### Branching Strategy

We use **Git Flow** with the following branches:

- `main` - Production-ready code, tagged releases
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes
- `research/*` - Experimental research code

### Creating a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git add .
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature
```

### Commit Messages

Follow the **Conventional Commits** specification:

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process or tooling changes

**Examples:**

```bash
# Feature commit
git commit -m "feat(ensemble): add confidence-weighted voting strategy

Implements weighted voting where votes are scaled by model confidence
scores. Improves accuracy on ambiguous queries by 5%.

Closes #42"

# Bug fix commit
git commit -m "fix(hedging): correct P95 percentile calculation

Previous calculation used P50 instead of P95, causing incorrect
hedge timing. Now properly calculates P95 from historical latencies.

Fixes #67"

# Documentation commit
git commit -m "docs(contributing): add commit message guidelines

Added section on conventional commits with examples."
```

### Keeping Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch
git checkout feature/amazing-feature
git rebase upstream/main

# Resolve any conflicts, then:
git rebase --continue

# Force push to your fork (if already pushed)
git push origin feature/amazing-feature --force-with-lease
```

---

## Code Standards

### Elixir Style Guide

We follow the [Elixir Style Guide](https://github.com/christopheradams/elixir_style_guide) with some additions:

**Formatting:**

```elixir
# Use mix format
mix format

# Check formatting in CI
mix format --check-formatted
```

**Configuration:**

`.formatter.exs` is already configured for the project.

### Code Organization

**Module structure:**

```elixir
defmodule MyApp.MyModule do
  @moduledoc """
  Brief description of module purpose.

  More detailed explanation if needed.

  ## Examples

      iex> MyModule.function(arg)
      :ok
  """

  # Module attributes
  @default_timeout 5000

  # Type specs
  @type t :: %__MODULE__{
    field: String.t(),
    count: non_neg_integer()
  }

  # Struct definition
  defstruct [:field, count: 0]

  # Public API (documented)
  @doc """
  Function documentation.

  ## Parameters

  - `arg1` - Description
  - `arg2` - Description

  ## Returns

  - `{:ok, result}` on success
  - `{:error, reason}` on failure

  ## Examples

      iex> function(1, 2)
      {:ok, 3}
  """
  @spec function(integer(), integer()) :: {:ok, integer()} | {:error, term()}
  def function(arg1, arg2) do
    # Implementation
  end

  # Private functions
  defp private_helper(arg) do
    # Implementation
  end
end
```

### Naming Conventions

**Modules:**
```elixir
# PascalCase for modules
Ensemble.VotingStrategies.Majority
Bench.StatisticalTests.TTest
```

**Functions:**
```elixir
# snake_case for functions and variables
def calculate_consensus(votes) do
  total_votes = Enum.count(votes)
  # ...
end
```

**Private functions:**
```elixir
# Leading underscore for intentionally unused variables
def process({:ok, _result}), do: :ok
def process({:error, _reason}), do: :error
```

### Pattern Matching

**Prefer pattern matching over conditional logic:**

```elixir
# Good
def handle_result({:ok, data}), do: process(data)
def handle_result({:error, reason}), do: log_error(reason)

# Avoid
def handle_result(result) do
  if elem(result, 0) == :ok do
    process(elem(result, 1))
  else
    log_error(elem(result, 1))
  end
end
```

### Pipeline Operator

**Use pipelines for transformations:**

```elixir
# Good
def process_votes(votes) do
  votes
  |> Enum.filter(&valid?/1)
  |> Enum.map(&normalize/1)
  |> Enum.frequencies()
  |> Map.to_list()
  |> Enum.sort_by(fn {_vote, count} -> count end, :desc)
end

# Avoid nested function calls
def process_votes(votes) do
  Enum.sort_by(
    Map.to_list(
      Enum.frequencies(
        Enum.map(
          Enum.filter(votes, &valid?/1),
          &normalize/1
        )
      )
    ),
    fn {_vote, count} -> count end,
    :desc
  )
end
```

### Error Handling

**Use tagged tuples for expected errors:**

```elixir
@spec divide(number(), number()) :: {:ok, float()} | {:error, :division_by_zero}
def divide(_num, 0), do: {:error, :division_by_zero}
def divide(num, denom), do: {:ok, num / denom}
```

**Raise exceptions for programmer errors:**

```elixir
def calculate(nil), do: raise ArgumentError, "argument cannot be nil"
def calculate(data) when is_list(data), do: process(data)
```

**Use with for sequential operations:**

```elixir
def run_experiment(config) do
  with {:ok, dataset} <- load_dataset(config.dataset),
       {:ok, models} <- initialize_models(config.models),
       {:ok, results} <- execute_queries(dataset, models),
       {:ok, analysis} <- analyze_results(results) do
    {:ok, analysis}
  else
    {:error, reason} -> {:error, reason}
  end
end
```

### Documentation

**All public functions must have @doc:**

```elixir
@doc """
Brief one-line summary.

More detailed explanation of function behavior, edge cases,
and important implementation details.

## Parameters

- `param1` - Description including type and constraints
- `param2` - Description

## Returns

- `{:ok, result}` - Description of success case
- `{:error, reason}` - Description of error cases

## Examples

    iex> function(arg)
    {:ok, result}

    iex> function(invalid_arg)
    {:error, :invalid_input}

## Notes

Any additional context, performance characteristics, or caveats.
"""
@spec function(type1(), type2()) :: {:ok, result()} | {:error, reason()}
def function(param1, param2) do
  # Implementation
end
```

### Type Specifications

**Add @spec to all public functions:**

```elixir
@type vote :: String.t()
@type votes :: [vote()]
@type strategy :: :majority | :unanimous | :weighted

@spec count_votes(votes()) :: %{vote() => non_neg_integer()}
def count_votes(votes) do
  Enum.frequencies(votes)
end

@spec apply_strategy(votes(), strategy()) :: {:ok, vote()} | {:error, :no_consensus}
def apply_strategy(votes, strategy) do
  # Implementation
end
```

---

## Testing Requirements

### Test Coverage

**Minimum coverage requirements:**

- **Unit tests:** 90% coverage for all modules
- **Integration tests:** Cover all library interactions
- **Property tests:** For algorithms with invariants
- **Performance tests:** For latency-sensitive code

### Running Tests

```bash
# Run all tests
mix test

# Run specific test file
mix test test/ensemble/voting_strategies_test.exs

# Run with coverage
mix test --cover

# Run only slow tests (tagged)
mix test --only slow

# Run except slow tests
mix test --exclude slow
```

### Writing Tests

**Basic test structure:**

```elixir
defmodule Ensemble.VotingStrategies.MajorityTest do
  use ExUnit.Case, async: true

  alias Ensemble.VotingStrategies.Majority

  describe "determine_winner/1" do
    test "returns answer with most votes" do
      votes = ["A", "A", "B", "A", "C"]

      assert {:ok, "A"} = Majority.determine_winner(votes)
    end

    test "returns error when no clear majority" do
      votes = ["A", "B", "C"]

      assert {:error, :no_consensus} = Majority.determine_winner(votes)
    end

    test "handles empty votes list" do
      assert {:error, :empty_votes} = Majority.determine_winner([])
    end
  end
end
```

**Property-based testing:**

```elixir
defmodule Bench.Statistics.PropertiesTest do
  use ExUnit.Case
  use ExUnitProperties

  describe "percentile/2 properties" do
    property "percentile is within data range" do
      check all data <- list_of(float(), min_length: 1),
                p <- float(min: 0.0, max: 1.0) do
        result = Bench.Statistics.percentile(data, p)
        min_val = Enum.min(data)
        max_val = Enum.max(data)

        assert result >= min_val
        assert result <= max_val
      end
    end

    property "50th percentile is approximately the median" do
      check all data <- list_of(integer(), min_length: 10) do
        p50 = Bench.Statistics.percentile(data, 0.5)
        median = Enum.sort(data) |> Enum.at(div(length(data), 2))

        assert_in_delta p50, median, 1.0
      end
    end
  end
end
```

**Async tests:**

```elixir
# Use async: true for independent tests
use ExUnit.Case, async: true

# Use async: false for tests with shared state
use ExUnit.Case, async: false
```

**Test tags:**

```elixir
@tag :slow
@tag timeout: 60_000
test "runs expensive computation" do
  # Long-running test
end

@tag :integration
test "tests multiple libraries together" do
  # Integration test
end
```

### Mocking and Stubbing

**Use Mox for mocks:**

```elixir
# Define mock in test_helper.exs
Mox.defmock(Ensemble.LLMClientMock, for: Ensemble.LLMClient.Behaviour)

# Use in tests
defmodule Ensemble.PredictTest do
  use ExUnit.Case

  import Mox

  setup :verify_on_exit!

  test "calls LLM client with correct parameters" do
    expect(Ensemble.LLMClientMock, :query, fn model, prompt ->
      assert model == :gpt4
      assert prompt =~ "What is"
      {:ok, "Answer"}
    end)

    Ensemble.predict("What is 2+2?", models: [:gpt4])
  end
end
```

### Performance Testing

**Benchmark critical paths:**

```elixir
defmodule Bench.PerformanceTest do
  use ExUnit.Case

  @tag :benchmark
  test "percentile calculation performance" do
    data = Enum.to_list(1..10_000)

    {time_us, _result} = :timer.tc(fn ->
      Bench.Statistics.percentile(data, 0.99)
    end)

    # Should complete in under 1ms
    assert time_us < 1_000
  end
end
```

---

## Documentation Standards

### Module Documentation

**Every module must have @moduledoc:**

```elixir
defmodule Ensemble.VotingStrategies.Majority do
  @moduledoc """
  Majority voting strategy for ensemble predictions.

  Selects the answer that receives the most votes from the ensemble.
  In case of ties, returns an error indicating no consensus was reached.

  ## Algorithm

  1. Count frequency of each unique answer
  2. Find answer(s) with maximum count
  3. If exactly one winner, return it
  4. If tie, return error

  ## Examples

      iex> Majority.determine_winner(["A", "A", "B"])
      {:ok, "A"}

      iex> Majority.determine_winner(["A", "B"])
      {:error, :no_consensus}

  ## Performance

  - Time complexity: O(n) where n is number of votes
  - Space complexity: O(k) where k is number of unique answers
  """
end
```

### Function Documentation

**Include all sections:**

```elixir
@doc """
Calculates the weighted average of predictions.

Weights each prediction by its associated confidence score,
then returns the weighted average. Used for numerical predictions
where averaging makes sense.

## Parameters

- `predictions` - List of {value, confidence} tuples where:
  - `value` is a number
  - `confidence` is a float between 0.0 and 1.0
- `opts` - Keyword list of options:
  - `:normalize` - Whether to normalize confidences to sum to 1.0 (default: true)
  - `:min_confidence` - Minimum confidence threshold (default: 0.1)

## Returns

- `{:ok, weighted_avg}` - Weighted average as float
- `{:error, :empty_predictions}` - When predictions list is empty
- `{:error, :invalid_confidence}` - When confidence values are invalid

## Examples

    iex> predictions = [{10, 0.9}, {20, 0.3}, {15, 0.7}]
    iex> weighted_average(predictions)
    {:ok, 13.42}

    iex> weighted_average([])
    {:error, :empty_predictions}

## Performance

- Time: O(n) where n is number of predictions
- Space: O(1) additional memory

## See Also

- `Ensemble.VotingStrategies.Median` - For robust central tendency
"""
@spec weighted_average([{number(), float()}], keyword()) ::
        {:ok, float()} | {:error, atom()}
def weighted_average(predictions, opts \\ []) do
  # Implementation
end
```

### README Files

**Each library should have comprehensive README:**

```markdown
# Library Name

Brief one-paragraph description.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

\`\`\`elixir
def deps do
  [
    {:library_name, "~> 0.1.0"}
  ]
end
\`\`\`

## Quick Start

[Minimal example showing basic usage]

## Documentation

Full documentation: https://hexdocs.pm/library_name

## Examples

[2-3 comprehensive examples]

## Configuration

[Configuration options if any]

## Testing

\`\`\`bash
mix test
\`\`\`

## License

MIT
```

### Inline Comments

**Use sparingly for non-obvious code:**

```elixir
def calculate_hedge_delay(latencies) do
  # Use P95 rather than P99 to avoid over-hedging
  # on outliers while still catching most tail latency
  p95 = Statistics.percentile(latencies, 0.95)

  # Google's research shows 90% of P95 is optimal
  # balance between cost and latency reduction
  p95 * 0.9
end
```

---

## Pull Request Process

### Before Submitting

**Checklist:**

- [ ] Code follows style guidelines
- [ ] All tests pass (`mix test`)
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Ran `mix format`
- [ ] Ran `mix credo` (if available)
- [ ] Ran `mix dialyzer` for type checking
- [ ] Updated CHANGELOG.md
- [ ] Rebased on latest main

### PR Title and Description

**Title format:**

```
[Type] Brief description (closes #issue)

Examples:
- [Feature] Add confidence-weighted voting (closes #42)
- [Bug] Fix P95 percentile calculation (closes #67)
- [Docs] Improve ensemble guide with examples
```

**Description template:**

```markdown
## Summary

Brief description of what this PR does and why.

## Changes

- Change 1
- Change 2
- Change 3

## Testing

Describe how you tested these changes:

- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Integration tests pass
- [ ] Performance impact measured

## Documentation

- [ ] README updated
- [ ] API docs updated
- [ ] Examples added/updated
- [ ] CHANGELOG updated

## Related Issues

Closes #42
Related to #67

## Screenshots (if applicable)

[Add screenshots for UI changes]

## Checklist

- [ ] Code follows project style
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Breaking Changes

If this PR contains breaking changes, describe:

1. What breaks
2. How to migrate
3. Why the breaking change is necessary

## Performance Impact

If applicable, describe performance impact:

- Latency: +5ms average (acceptable for feature)
- Memory: +2MB for caching (configurable)
- Throughput: No change

## Additional Context

Any other relevant information.
```

### Submitting the PR

```bash
# Push your branch
git push origin feature/amazing-feature

# Create PR on GitHub
# Fill out the PR template
# Link related issues
# Request reviewers
```

---

## Review Guidelines

### For PR Authors

**Responding to feedback:**

- Be open to suggestions and constructive criticism
- Respond to all comments, even if just to acknowledge
- Make requested changes in new commits (don't force push until approved)
- Ask for clarification if feedback is unclear
- Thank reviewers for their time

**Common review feedback:**

- "Add tests for this case" - Write the test
- "Extract this to a function" - Refactor as suggested
- "Add documentation" - Document the code
- "This could be simplified" - Consider the suggestion
- "Performance concern" - Benchmark and optimize if needed

### For Reviewers

**What to look for:**

1. **Correctness:** Does the code do what it claims?
2. **Tests:** Are there adequate tests? Do they cover edge cases?
3. **Documentation:** Is the code well-documented?
4. **Style:** Does it follow project conventions?
5. **Performance:** Any obvious performance issues?
6. **Security:** Any security concerns?
7. **Maintainability:** Is the code readable and maintainable?

**How to give feedback:**

```markdown
# Good feedback (specific, actionable, kind)
This function could benefit from error handling for the empty list case.
Consider adding:

\`\`\`elixir
def process([]), do: {:error, :empty_list}
def process(items), do: # existing logic
\`\`\`

# Avoid (vague, critical, not helpful)
This doesn't look right.
```

**Review checklist:**

- [ ] Code is correct and solves the stated problem
- [ ] Tests are comprehensive and pass
- [ ] Documentation is clear and complete
- [ ] No obvious performance issues
- [ ] Follows project style
- [ ] No security vulnerabilities
- [ ] Breaking changes are documented
- [ ] CHANGELOG is updated

**Approval guidelines:**

- Approve when satisfied with quality
- Request changes for significant issues
- Comment for minor suggestions that don't block approval
- Re-review after changes are made

---

## Community Norms

### Communication

**Channels:**

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas, general discussion
- **Pull Requests:** Code review and discussion
- **Email:** research@example.com for private matters

**Response times:**

- Issues: Within 1-2 business days
- PRs: Initial review within 3-5 business days
- Discussion: Best effort, usually within 1 week

### Inclusivity

**We strive to be welcoming to all contributors:**

- Use inclusive language (they/them when gender unknown)
- Be patient with newcomers
- Assume good intent
- Provide constructive feedback
- Celebrate contributions of all sizes

### Attribution

**We recognize all contributions:**

- Code authors in git history
- Contributors in CHANGELOG.md
- Major contributors in README.md
- Research collaborators in PUBLICATIONS.md

### Acknowledgment

**Ways we say thank you:**

- Public thanks in PR comments
- Contributor badge in README
- Co-authorship on papers when appropriate
- Conference presentation opportunities

---

## Research Collaboration

### Sharing Results

**We encourage sharing:**

- **Experimental results** using the framework
- **Replication studies** of published research
- **Novel hypotheses** for community testing
- **Dataset contributions** for benchmarking

### Pre-publication Review

**Before publishing research using this framework:**

1. **Verify reproducibility** of your results
2. **Share experimental artifacts** (code, data, configs)
3. **Request community review** via GitHub Discussions
4. **Cite the framework** appropriately (see PUBLICATIONS.md)

### Collaboration Protocol

**For joint research projects:**

1. **Propose the research** in GitHub Discussions
2. **Define roles and authorship** upfront
3. **Use research/* branches** for experimental code
4. **Document methodology** thoroughly
5. **Share interim results** with collaborators
6. **Coordinate publication** submission

### Data Sharing

**Guidelines for sharing datasets:**

1. **Licensing:** Ensure you have rights to share
2. **Privacy:** Remove any PII or sensitive information
3. **Format:** Use standard formats (JSONL, CSV, Parquet)
4. **Documentation:** Include schema, collection methodology
5. **Versioning:** Use semantic versioning for datasets
6. **Citation:** Provide citation information

**Dataset contribution process:**

```bash
# Add dataset to DatasetManager
# 1. Create dataset module
# apps/dataset_manager/lib/dataset_manager/datasets/my_dataset.ex

# 2. Implement Loader behavior
defmodule DatasetManager.Datasets.MyDataset do
  @behaviour DatasetManager.Loader

  @impl true
  def load(opts) do
    # Implementation
  end

  # ... other callbacks
end

# 3. Add tests
# test/dataset_manager/datasets/my_dataset_test.exs

# 4. Update documentation
# apps/dataset_manager/README.md

# 5. Submit PR with dataset
```

---

## Library-Specific Guidelines

### Ensemble Library

**Adding voting strategies:**

1. Implement `Ensemble.VotingStrategy` behaviour
2. Add comprehensive tests including edge cases
3. Document algorithm complexity
4. Benchmark performance vs existing strategies
5. Add examples to README

**Example:**

```elixir
defmodule Ensemble.VotingStrategies.MyStrategy do
  @behaviour Ensemble.VotingStrategy

  @impl true
  def determine_winner(votes, _opts) do
    # Implementation
  end

  @impl true
  def requires_confidence?, do: false
end
```

### Hedging Library

**Adding hedging strategies:**

1. Implement `Hedging.Strategy` behaviour
2. Test with simulated latency distributions
3. Measure cost vs latency tradeoff
4. Compare against baseline strategies
5. Document when to use this strategy

### Bench Library

**Adding statistical tests:**

1. Implement test function with clear interface
2. Include assumption checking
3. Calculate effect sizes
4. Provide interpretation text
5. Add references to statistical literature
6. Validate against R or Python implementations

### TelemetryResearch Library

**Adding metrics:**

1. Define metric schema
2. Implement calculation efficiently
3. Add aggregation functions
4. Support multiple export formats
5. Document metric interpretation

### DatasetManager Library

**Adding datasets:**

1. Implement Loader behaviour
2. Handle caching properly
3. Support sampling strategies
4. Provide evaluation metrics
5. Document dataset source and license

---

## Performance Considerations

### Profiling

**Before optimizing, profile:**

```bash
# Use :fprof for detailed profiling
:fprof.apply(&MyModule.slow_function/1, [arg])
:fprof.profile()
:fprof.analyse()

# Use :eprof for function call counting
:eprof.start()
:eprof.profile(fn -> MyModule.function() end)
:eprof.analyse()

# Use Benchee for microbenchmarks
Benchee.run(%{
  "original" => fn -> original_implementation() end,
  "optimized" => fn -> optimized_implementation() end
})
```

### Common Optimizations

**Avoid creating unnecessary lists:**

```elixir
# Good - use streams for large datasets
dataset
|> Stream.filter(&valid?/1)
|> Stream.map(&process/1)
|> Enum.take(100)

# Avoid - materializes entire list
dataset
|> Enum.filter(&valid?/1)
|> Enum.map(&process/1)
|> Enum.take(100)
```

**Use ETS for caching:**

```elixir
# Fast lookups for frequently accessed data
:ets.new(:cache, [:set, :public, :named_table])
:ets.insert(:cache, {key, value})
:ets.lookup(:cache, key)
```

**Parallelize independent operations:**

```elixir
# Good - parallel queries
results =
  queries
  |> Task.async_stream(&call_api/1, max_concurrency: 10)
  |> Enum.map(fn {:ok, result} -> result end)

# Avoid - sequential
results = Enum.map(queries, &call_api/1)
```

### Memory Management

**Be conscious of memory usage:**

- Use streaming for large datasets
- Clean up ETS tables when done
- Avoid keeping unnecessary data in process state
- Use `:binary.copy/1` for large binaries from external sources

---

## Security Guidelines

### API Keys

**Never commit API keys:**

```bash
# Use environment variables
export OPENAI_API_KEY="sk-..."

# Or use .env file (add to .gitignore)
OPENAI_API_KEY=sk-...
```

**In code:**

```elixir
# Good
api_key = System.get_env("OPENAI_API_KEY")

# Never do this
api_key = "sk-proj-actual-key-here"  # WRONG!
```

### Input Validation

**Always validate external input:**

```elixir
def process_query(query) when is_binary(query) and byte_size(query) > 0 do
  # Safe to process
end

def process_query(_invalid) do
  {:error, :invalid_query}
end
```

### Dependency Security

**Keep dependencies updated:**

```bash
# Check for security advisories
mix deps.audit

# Update dependencies
mix deps.update --all
```

### Rate Limiting

**Implement rate limiting for external APIs:**

```elixir
# Use token bucket or similar
defmodule RateLimiter do
  def check_rate_limit(api_key) do
    # Implementation
  end
end
```

---

## Release Process

### Versioning

**We use Semantic Versioning:**

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

### Changelog

**Update CHANGELOG.md for each release:**

```markdown
# Changelog

## [Unreleased]

### Added
- New feature description

### Changed
- Changed behavior description

### Fixed
- Bug fix description

### Deprecated
- Deprecated feature (will be removed in next major version)

## [0.2.0] - 2025-02-01

### Added
- Confidence-weighted voting strategy
- PostgreSQL storage backend for telemetry

### Fixed
- P95 percentile calculation in hedging
```

### Release Checklist

**Before releasing:**

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG is updated
- [ ] Version numbers bumped in mix.exs
- [ ] Git tag created
- [ ] Release notes written
- [ ] Artifacts built and published

### Creating a Release

```bash
# Update version in mix.exs files
vim mix.exs apps/*/mix.exs

# Update CHANGELOG
vim CHANGELOG.md

# Commit version bump
git add .
git commit -m "Bump version to 0.2.0"

# Create tag
git tag v0.2.0

# Push tag
git push origin v0.2.0

# Publish to Hex (if applicable)
mix hex.publish
```

---

## Getting Help

### Documentation

- **Framework docs:** https://hexdocs.pm/elixir_ai_research
- **Elixir guides:** https://elixir-lang.org/getting-started/introduction.html
- **OTP docs:** https://www.erlang.org/doc/

### Community Support

- **GitHub Discussions:** Ask questions, share ideas
- **GitHub Issues:** Report bugs, request features
- **Email:** research@example.com

### Debugging Tips

**Common issues:**

1. **Mix compile errors:**
   ```bash
   mix deps.clean --all
   mix deps.get
   mix compile
   ```

2. **Test failures:**
   ```bash
   mix test --trace  # See each test as it runs
   mix test --seed 0  # Reproduce random failures
   ```

3. **Dependencies issues:**
   ```bash
   mix deps.update --all
   mix deps.clean --unused
   ```

---

## Recognition

### Contributors

We recognize contributors in several ways:

- **Git commits:** Your name in commit history
- **CHANGELOG:** Credited for significant contributions
- **README:** Listed as contributor
- **Papers:** Co-authorship for research contributions
- **Releases:** Mentioned in release notes

### Hall of Fame

Outstanding contributors may be featured in our Hall of Fame for:

- Significant code contributions
- Exceptional documentation
- Research collaborations
- Community leadership

---

## Final Notes

### Thank You

Thank you for contributing to the Elixir AI Research Framework! Your contributions help advance research in LLM reliability and performance.

### Questions?

If anything in this guide is unclear, please:

1. Open a Discussion on GitHub
2. Submit a PR to improve this document
3. Email research@example.com

### Stay Updated

- Watch the repository for updates
- Subscribe to releases for notifications
- Join discussions for community engagement

---

**Last Updated:** 2025-10-08
**Version:** 1.0.0
**Maintainers:** Research Infrastructure Team

---

Built with ❤️ by researchers, for researchers.
