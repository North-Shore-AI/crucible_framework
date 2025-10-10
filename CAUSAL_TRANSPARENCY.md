# Causal Transparency and Debugging Guide

**Comprehensive guide to understanding, analyzing, and debugging AI reasoning chains through causal event tracing.**

Version: 1.0
Last Updated: 2025-10-08

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Transparency Matters](#why-transparency-matters)
3. [Core Concepts](#core-concepts)
4. [Event Types Explained](#event-types-explained)
5. [Creating and Managing Chains](#creating-and-managing-chains)
6. [HTML Visualization](#html-visualization)
7. [Analysis Techniques](#analysis-techniques)
8. [User Study Protocols](#user-study-protocols)
9. [Integration with Code Generation](#integration-with-code-generation)
10. [Advanced Debugging](#advanced-debugging)
11. [Research Applications](#research-applications)
12. [API Reference](#api-reference)

---

## Introduction

The CausalTrace library provides a framework for capturing, visualizing, and analyzing the decision-making process of AI systems during code generation. It creates an audit trail of reasoning steps, alternatives considered, and confidence levels throughout the generation process.

### Key Features

- **Event-Based Tracing**: Capture discrete decision points in the reasoning process
- **Alternative Tracking**: Record paths not taken and why
- **Confidence Monitoring**: Track uncertainty throughout generation
- **Interactive Visualization**: HTML-based exploration of reasoning chains
- **Pattern Analysis**: Identify common reasoning patterns and failure modes
- **Human Studies**: Protocols for evaluating transparency benefits

### What Makes This Different

Traditional LLM observability focuses on inputs and outputs. Causal transparency goes deeper:

**Traditional Logging**:
```
Input: "Implement a cache"
Output: [generated code]
```

**Causal Transparency**:
```
1. Hypothesis Formed: Use LRU eviction strategy
   Alternatives: [FIFO, Random, LFU]
   Reasoning: LRU balances recency and frequency
   Confidence: 0.85

2. Alternative Rejected: GenServer for thread safety
   Chosen: ETS for better performance
   Reasoning: ETS provides O(1) lookups without GenServer overhead
   Confidence: 0.92

3. Constraint Evaluated: Memory limits
   Decision: Add size-based eviction
   Reasoning: Prevent unbounded memory growth
   Confidence: 0.95
```

This granular view enables debugging, validation, and understanding of AI reasoning.

---

## Why Transparency Matters

### Research Background

The importance of AI transparency has been extensively studied:

**Interpretability Research**:

1. **Lipton (2016)** - "The Mythos of Model Interpretability"
   - Distinguishes between transparency (understanding mechanism) and post-hoc interpretability
   - Argues for building inherently interpretable systems
   - Citation: *arXiv:1606.03490*

2. **Doshi-Velez & Kim (2017)** - "Towards A Rigorous Science of Interpretable Machine Learning"
   - Establishes taxonomy of interpretability evaluation
   - Proposes human-grounded experiments for transparency claims
   - Citation: *arXiv:1702.08608*

3. **Miller (2019)** - "Explanation in Artificial Intelligence: Insights from the Social Sciences"
   - Reviews 250+ papers on explanation theory from social sciences
   - Key finding: Good explanations are contrastive (explain why X not Y)
   - Citation: *Artificial Intelligence, 267:1-38*

**Code Generation Specific**:

4. **Chen et al. (2021)** - "Evaluating Large Language Models Trained on Code"
   - Introduced HumanEval but noted lack of reasoning transparency
   - Recommendation: "Future work should explore making the reasoning process explicit"
   - Citation: *arXiv:2107.03374*

5. **Peng et al. (2023)** - "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot"
   - Found 55.8% faster task completion but 41% wanted "better explanations"
   - Developers spend significant time debugging generated code
   - Citation: *arXiv:2302.06590*

### Benefits of Causal Transparency

**1. Debugging and Error Analysis**

Traditional approach:
```elixir
# Code doesn't work - why?
# Black box output with no insight into reasoning
```

With causal transparency:
```elixir
# Chain shows:
# - Hypothesis: Use recursion (confidence: 0.6)
# - Alternative rejected: Iteration (too low confidence!)
# - Pattern applied: Tail recursion (confidence: 0.5)
#
# LOW CONFIDENCE throughout recursive solution!
# Switch to iteration which was rejected.
```

**2. Trust and Adoption**

Research shows transparency increases trust:

- **Ribeiro et al. (2016)** - "Why Should I Trust You?"
  - Model-agnostic explanations increased trust by 24%
  - Citation: *KDD 2016*

- **Springer et al. (2017)** - "Transparency Yields Better Collaboration"
  - Transparency increased human-AI task performance by 35%
  - Citation: *CSCW 2017*

**3. Learning and Skill Development**

Causal traces serve as educational tools:

```elixir
# Novice sees:
Event 1: Hypothesis Formed
  Decision: "Use GenServer for state management"
  Reasoning: "Need to handle concurrent access safely"
  Alternatives: ["Agent", "ETS", "Process dictionary"]
  Code Section: "defmodule Cache"

# Novice learns:
# - Why GenServer over alternatives
# - When to consider each option
# - Confidence in architectural decisions
```

**4. Regulatory Compliance**

GDPR Article 22 requires "right to explanation" for automated decisions. Causal traces provide:

- **Auditability**: Complete record of decision process
- **Justification**: Reasoning for each choice
- **Accountability**: Attribution of decisions

**5. Research and Model Improvement**

Aggregate analysis reveals:

- Common failure modes
- Low-confidence decision patterns
- Alternative paths that should have been taken
- Domain-specific reasoning patterns

---

## Core Concepts

### Causal Chains

A **chain** represents the complete reasoning trace for a single task:

```elixir
%CausalTrace.Chain{
  id: "chain_abc123",
  name: "Implement Binary Search Tree",
  description: "Generate BST with insert, search, and delete operations",
  events: [
    # Event 1: Initial hypothesis
    %Event{...},
    # Event 2: Constraint evaluation
    %Event{...},
    # Event 3: Pattern application
    %Event{...}
  ],
  metadata: %{
    task_type: "code_generation",
    language: "elixir",
    complexity: "medium"
  },
  created_at: ~U[2025-10-08 10:30:00Z],
  updated_at: ~U[2025-10-08 10:30:45Z]
}
```

### Events

An **event** captures a single decision point:

```elixir
%CausalTrace.Event{
  id: "event_xyz789",
  timestamp: ~U[2025-10-08 10:30:15Z],
  type: :hypothesis_formed,
  decision: "Use recursive implementation for tree traversal",
  alternatives: [
    "Iterative with stack",
    "Iterative with queue",
    "Continuation-passing style"
  ],
  reasoning: "Recursion naturally mirrors tree structure and is more readable for this use case",
  confidence: 0.87,
  code_section: "def traverse(node)",
  spec_reference: "BST traversal specification section 2.3",
  metadata: %{
    subtask: "implement_traversal",
    depth: 2
  }
}
```

### Event Lifecycle

Events flow through a chain chronologically:

```
Task Start
    ↓
Hypothesis Formed ──→ Alternative Rejected
    ↓                      ↓
Constraint Evaluated      ↓
    ↓                      ↓
Pattern Applied ←──────────┘
    ↓
Confidence Updated
    ↓
More Events...
    ↓
Task Complete
```

### Confidence Tracking

Confidence values (0.0 to 1.0) indicate certainty:

- **0.9-1.0**: High confidence (well-established patterns)
- **0.7-0.9**: Medium-high confidence (reasonable approach)
- **0.5-0.7**: Medium confidence (uncertain, may need review)
- **0.3-0.5**: Low confidence (likely needs revision)
- **0.0-0.3**: Very low confidence (flag for immediate review)

**Example**:

```elixir
chain = CausalTrace.Chain.new("API Endpoint")

# High confidence - standard pattern
chain = CausalTrace.Chain.add_event(chain,
  CausalTrace.Event.new(
    :pattern_applied,
    "Use Plug for HTTP handling",
    "Plug is standard for web servers in Elixir",
    confidence: 0.95
  )
)

# Low confidence - unusual requirement
chain = CausalTrace.Chain.add_event(chain,
  CausalTrace.Event.new(
    :ambiguity_flagged,
    "Rate limiting strategy unclear",
    "Requirements don't specify rate limit algorithm or window",
    confidence: 0.45,
    alternatives: ["Token bucket", "Sliding window", "Fixed window"]
  )
)
```

---

## Event Types Explained

### 1. Hypothesis Formed

**Purpose**: Record an initial approach or solution strategy.

**When to Use**:
- Beginning a new subtask
- Proposing an architecture
- Selecting an algorithm

**Example**:

```elixir
event = CausalTrace.Event.new(
  :hypothesis_formed,
  "Implement caching layer with Redis",
  """
  Redis provides:
  - Distributed caching across nodes
  - Built-in TTL support
  - Pub/sub for cache invalidation
  - Better performance than in-memory for large datasets
  """,
  alternatives: [
    "ETS (in-memory)",
    "Memcached",
    "PostgreSQL materialized views"
  ],
  confidence: 0.82,
  code_section: "defmodule MyApp.Cache"
)
```

**Research Note**: Miller (2019) emphasizes contrastive explanations - always include alternatives to show what was NOT chosen and why.

### 2. Alternative Rejected

**Purpose**: Document why a specific alternative was NOT selected.

**When to Use**:
- After evaluating multiple approaches
- When deviating from a common pattern
- When a seemingly obvious choice is rejected

**Example**:

```elixir
event = CausalTrace.Event.new(
  :alternative_rejected,
  "Rejected GenServer for high-throughput cache",
  """
  GenServer would work but:
  - Message passing overhead for each lookup
  - Single process bottleneck under high load
  - ETS provides O(1) concurrent reads without serialization

  GenServer is better for:
  - When write consistency is critical
  - When you need complex coordination
  - When the bottleneck is NOT cache access
  """,
  alternatives: ["ETS with periodic cleanup"],
  confidence: 0.91,
  metadata: %{
    rejected_approach: "GenServer",
    reason_category: "performance"
  }
)
```

**Best Practice**: Always explain the context where the rejected alternative WOULD be appropriate. This aids learning.

### 3. Constraint Evaluated

**Purpose**: Record evaluation of requirements, limitations, or constraints.

**When to Use**:
- Checking system requirements
- Validating assumptions
- Evaluating trade-offs

**Example**:

```elixir
event = CausalTrace.Event.new(
  :constraint_evaluated,
  "Memory constraint: Cache must not exceed 1GB",
  """
  Current design implications:
  - Need size-based eviction policy (not just TTL)
  - Must track memory usage per entry
  - LRU suitable for size-constrained cache

  Implementation approach:
  - Track byte size with each cache entry
  - Maintain running total
  - Evict LRU items when approaching limit
  """,
  confidence: 0.88,
  spec_reference: "Requirements doc section 3.2: Resource limits",
  metadata: %{
    constraint_type: "memory",
    limit: "1GB",
    enforcement: "soft"
  }
)
```

**Pattern**: Always link to requirements/specs when evaluating constraints.

### 4. Pattern Applied

**Purpose**: Record application of a known pattern or best practice.

**When to Use**:
- Applying design patterns
- Following language idioms
- Using established best practices

**Example**:

```elixir
event = CausalTrace.Event.new(
  :pattern_applied,
  "Apply Supervisor pattern for fault tolerance",
  """
  Using OTP Supervisor to manage cache processes:
  - one_for_one strategy: Restart only failed process
  - max_restarts: 3, max_seconds: 5
  - Child spec for CacheWorker

  This provides:
  - Automatic restart on crashes
  - Isolation of failures
  - System stability

  Pattern reference: "Let it crash" philosophy
  """,
  confidence: 0.93,
  code_section: "defmodule MyApp.CacheSupervisor",
  metadata: %{
    pattern_name: "supervisor",
    pattern_source: "OTP",
    strategy: "one_for_one"
  }
)
```

**Educational Value**: Explicit pattern identification helps developers learn by seeing patterns in context.

### 5. Ambiguity Flagged

**Purpose**: Highlight unclear requirements or uncertain decisions.

**When to Use**:
- Requirements are ambiguous or contradictory
- Multiple valid interpretations exist
- Need human clarification

**Example**:

```elixir
event = CausalTrace.Event.new(
  :ambiguity_flagged,
  "Unclear: Should cache persist across application restarts?",
  """
  Requirements state "caching layer" but don't specify:
  - Persistence requirements
  - Cold start behavior
  - Data durability needs

  Possible interpretations:
  1. In-memory only (ETS) - fast, volatile
  2. Redis - persisted, survives restarts
  3. Hybrid - ETS with periodic Redis backup

  NEEDS CLARIFICATION from product team.

  Proceeding with assumption: In-memory only (most common case).
  If wrong, easy migration path to Redis exists.
  """,
  confidence: 0.52,  # Low confidence due to ambiguity
  alternatives: [
    "ETS (assumption)",
    "Redis with persistence",
    "Hybrid ETS+Redis"
  ],
  metadata: %{
    requires_clarification: true,
    assumption_made: "volatile cache acceptable",
    migration_path: "ets_to_redis_guide.md"
  }
)
```

**Critical Feature**: Ambiguity flagging prevents silent failures from misunderstood requirements.

### 6. Confidence Updated

**Purpose**: Record changes in confidence as more information becomes available.

**When to Use**:
- After evaluating additional constraints
- When discovering new information
- After implementing and testing

**Example**:

```elixir
# Initial decision
event1 = CausalTrace.Event.new(
  :hypothesis_formed,
  "Use JSON for cache serialization",
  "JSON is human-readable and widely supported",
  confidence: 0.75
)

# Later, after testing
event2 = CausalTrace.Event.new(
  :confidence_updated,
  "Reduced confidence in JSON serialization",
  """
  Performance testing revealed:
  - JSON encoding/decoding adds 15ms overhead per operation
  - Target is 5ms total cache access time
  - JSON overhead is 75% of total latency

  New recommendation: Use ETF (Erlang Term Format)
  - Native Elixir serialization
  - 10x faster than JSON
  - Still supports complex data types

  CONFIDENCE LOWERED from 0.75 to 0.35 for JSON.
  """,
  confidence: 0.35,
  metadata: %{
    original_confidence: 0.75,
    reason_for_update: "performance_testing",
    new_recommendation: "etf"
  }
)
```

**Pattern**: Confidence updates show learning and adaptation during implementation.

---

## Creating and Managing Chains

### Basic Chain Creation

```elixir
# Create a new chain
chain = CausalTrace.Chain.new(
  "Implement User Authentication",
  description: "Add JWT-based authentication to API",
  metadata: %{
    task_id: "AUTH-123",
    assigned_to: "ai_model_gpt4",
    priority: "high"
  }
)

# Add events as decisions are made
chain =
  chain
  |> CausalTrace.Chain.add_event(
    CausalTrace.Event.new(
      :hypothesis_formed,
      "Use JWT for stateless authentication",
      "JWT enables stateless auth, reducing database load",
      confidence: 0.88
    )
  )
  |> CausalTrace.Chain.add_event(
    CausalTrace.Event.new(
      :constraint_evaluated,
      "Tokens must expire in 1 hour",
      "Security requirement: Short-lived tokens reduce exposure",
      confidence: 0.95,
      spec_reference: "Security Policy Section 4.2"
    )
  )
  |> CausalTrace.Chain.add_event(
    CausalTrace.Event.new(
      :pattern_applied,
      "Apply Guardian library for JWT handling",
      "Guardian is the standard Elixir JWT library with strong community support",
      confidence: 0.92,
      code_section: "defmodule MyApp.Guardian"
    )
  )

# Inspect chain
IO.inspect(CausalTrace.Chain.statistics(chain))
# => %{
#   total_events: 3,
#   event_type_counts: %{
#     hypothesis_formed: 1,
#     constraint_evaluated: 1,
#     pattern_applied: 1
#   },
#   avg_confidence: 0.92,
#   duration_seconds: 45
# }
```

### Querying Chains

```elixir
# Get events by type
hypotheses = CausalTrace.Chain.get_events_by_type(chain, :hypothesis_formed)

# Find low confidence decisions
risky_decisions = CausalTrace.Chain.find_low_confidence(chain, 0.7)

Enum.each(risky_decisions, fn event ->
  IO.puts("⚠ Low confidence: #{event.decision} (#{event.confidence})")
  IO.puts("   Reasoning: #{event.reasoning}")
  IO.puts("   Alternatives: #{inspect(event.alternatives)}")
end)

# Get decision points (where alternatives were considered)
decision_points = CausalTrace.Chain.find_decision_points(chain)

Enum.each(decision_points, fn dp ->
  IO.puts("Decision: #{dp.decision}")
  IO.puts("Alternatives considered:")
  Enum.each(dp.alternatives, fn alt ->
    IO.puts("  - #{alt}")
  end)
end)
```

### Chain Analysis

```elixir
defmodule ChainAnalyzer do
  @doc """
  Analyze a chain for potential issues.
  """
  def analyze(chain) do
    stats = CausalTrace.Chain.statistics(chain)

    issues = []

    # Check for low average confidence
    issues =
      if stats.avg_confidence < 0.7 do
        [
          %{
            type: :low_confidence,
            severity: :warning,
            message: "Average confidence (#{stats.avg_confidence}) is below recommended threshold (0.7)"
          }
          | issues
        ]
      else
        issues
      end

    # Check for unresolved ambiguities
    ambiguities = CausalTrace.Chain.get_events_by_type(chain, :ambiguity_flagged)

    issues =
      if length(ambiguities) > 0 do
        [
          %{
            type: :unresolved_ambiguity,
            severity: :error,
            message: "#{length(ambiguities)} ambiguities require clarification",
            details: Enum.map(ambiguities, & &1.decision)
          }
          | issues
        ]
      else
        issues
      end

    # Check for contradictions
    issues =
      case find_contradictions(chain) do
        [] -> issues
        contradictions -> [
          %{
            type: :contradiction,
            severity: :error,
            message: "Found contradictory decisions",
            details: contradictions
          }
          | issues
        ]
      end

    %{
      chain_id: chain.id,
      total_events: stats.total_events,
      avg_confidence: stats.avg_confidence,
      issues: issues,
      recommendations: generate_recommendations(issues)
    }
  end

  defp find_contradictions(chain) do
    # Look for events that contradict each other
    # E.g., "Use Redis" followed by "Use ETS"
    chain.events
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [event1, event2] ->
      contradicts?(event1, event2)
    end)
    |> Enum.map(fn [event1, event2] ->
      %{
        event1: event1.decision,
        event2: event2.decision,
        conflict: describe_conflict(event1, event2)
      }
    end)
  end

  defp contradicts?(event1, event2) do
    # Simplified contradiction detection
    # In practice, would use semantic similarity
    String.contains?(event1.decision, "Redis") and
      String.contains?(event2.decision, "ETS")
  end

  defp describe_conflict(event1, event2) do
    "First chooses #{extract_technology(event1.decision)}, " <>
    "then switches to #{extract_technology(event2.decision)}"
  end

  defp extract_technology(text) do
    # Extract technology name from decision text
    text
  end

  defp generate_recommendations(issues) do
    Enum.map(issues, fn issue ->
      case issue.type do
        :low_confidence ->
          "Review low-confidence decisions with domain expert. Consider alternative approaches."

        :unresolved_ambiguity ->
          "Clarify requirements before proceeding. Flag for product owner review."

        :contradiction ->
          "Resolve contradictory decisions. May indicate changing requirements or errors."

        _ ->
          "Manual review recommended"
      end
    end)
  end
end

# Usage
analysis = ChainAnalyzer.analyze(chain)

if length(analysis.issues) > 0 do
  IO.puts("⚠ Chain Analysis: #{length(analysis.issues)} issues found")

  Enum.each(analysis.issues, fn issue ->
    severity_icon =
      case issue.severity do
        :error -> "❌"
        :warning -> "⚠️"
        :info -> "ℹ️"
      end

    IO.puts("#{severity_icon} #{issue.message}")
  end)

  IO.puts("\nRecommendations:")
  Enum.each(analysis.recommendations, fn rec ->
    IO.puts("  • #{rec}")
  end)
else
  IO.puts("✅ Chain analysis: No issues found")
end
```

### Chain Persistence

```elixir
defmodule ChainStorage do
  @doc """
  Save chain to disk for later analysis.
  """
  def save(chain, path) do
    json = chain |> CausalTrace.Chain.to_map() |> Jason.encode!(pretty: true)
    File.write!(path, json)
  end

  @doc """
  Load chain from disk.
  """
  def load(path) do
    path
    |> File.read!()
    |> Jason.decode!()
    |> CausalTrace.Chain.from_map()
  end

  @doc """
  Save chain to database.
  """
  def save_to_db(chain, repo) do
    attrs = %{
      chain_id: chain.id,
      name: chain.name,
      description: chain.description,
      events: Jason.encode!(Enum.map(chain.events, &CausalTrace.Event.to_map/1)),
      metadata: Jason.encode!(chain.metadata),
      statistics: Jason.encode!(CausalTrace.Chain.statistics(chain)),
      created_at: chain.created_at,
      updated_at: chain.updated_at
    }

    %ChainRecord{}
    |> ChainRecord.changeset(attrs)
    |> repo.insert()
  end
end

# Example
chain = CausalTrace.Chain.new("Implement Feature X")
# ... add events ...

# Save to file
ChainStorage.save(chain, "traces/feature_x_chain.json")

# Later, load from file
loaded_chain = ChainStorage.load("traces/feature_x_chain.json")

# Or save to database for querying
{:ok, _record} = ChainStorage.save_to_db(chain, MyApp.Repo)
```

---

## HTML Visualization

### Generating HTML Reports

The CausalTrace.Viewer module generates interactive HTML visualizations:

```elixir
# Generate HTML
html = CausalTrace.Viewer.generate_html(chain,
  title: "User Authentication Implementation",
  style: :light,
  include_statistics: true,
  include_timeline: true
)

# Save to file
{:ok, path} = CausalTrace.Viewer.save_html(chain, "reports/auth_chain.html")

# Open in browser
{:ok, _path} = CausalTrace.Viewer.open_in_browser(chain)
```

### Visualization Features

**1. Event List with Filtering**

```html
<!-- Users can filter by: -->
- Event type (hypothesis_formed, alternative_rejected, etc.)
- Confidence level (slider: 0.0 to 1.0)
- Code section
- Time range
```

**2. Timeline View**

Visual timeline showing event distribution over time:

```
|----H----A----C----P----A----C----|
0s   5s   10s  15s  20s  25s  30s

H = Hypothesis Formed
A = Alternative Rejected
C = Constraint Evaluated
P = Pattern Applied
```

**3. Statistics Dashboard**

```
┌─────────────────────────────────────┐
│ Chain Statistics                    │
├─────────────────────────────────────┤
│ Total Events: 15                    │
│ Average Confidence: 0.84            │
│ Duration: 45 seconds                │
│                                     │
│ Event Type Breakdown:               │
│ - Hypothesis Formed: 4              │
│ - Alternative Rejected: 3           │
│ - Constraint Evaluated: 5           │
│ - Pattern Applied: 2                │
│ - Confidence Updated: 1             │
└─────────────────────────────────────┘
```

### Customizing Visualizations

```elixir
defmodule CustomViewer do
  @doc """
  Generate custom HTML with additional analysis.
  """
  def generate_with_analysis(chain) do
    # Standard HTML
    base_html = CausalTrace.Viewer.generate_html(chain)

    # Add custom analysis section
    analysis = ChainAnalyzer.analyze(chain)
    analysis_html = generate_analysis_section(analysis)

    # Combine
    insert_before_closing_body(base_html, analysis_html)
  end

  defp generate_analysis_section(analysis) do
    """
    <section class="custom-analysis">
      <h2>Automated Analysis</h2>
      <div class="analysis-summary">
        <h3>Overall Assessment</h3>
        <p>Average Confidence: #{Float.round(analysis.avg_confidence, 2)}</p>
        <p>Issues Found: #{length(analysis.issues)}</p>
      </div>

      #{if length(analysis.issues) > 0 do
        """
        <div class="issues-list">
          <h3>Issues Requiring Attention</h3>
          <ul>
            #{Enum.map(analysis.issues, &issue_item/1) |> Enum.join("\n")}
          </ul>
        </div>
        """
      else
        "<p class=\"success\">✅ No issues detected</p>"
      end}

      <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
          #{Enum.map(analysis.recommendations, &recommendation_item/1) |> Enum.join("\n")}
        </ul>
      </div>
    </section>
    """
  end

  defp issue_item(issue) do
    """
    <li class="issue #{issue.severity}">
      <strong>#{issue.type}</strong>: #{issue.message}
    </li>
    """
  end

  defp recommendation_item(rec) do
    "<li>#{rec}</li>"
  end

  defp insert_before_closing_body(html, content) do
    String.replace(html, "</body>", "#{content}\n</body>")
  end
end

# Usage
html = CustomViewer.generate_with_analysis(chain)
File.write!("custom_report.html", html)
```

### Embedding in Web Applications

```elixir
defmodule MyAppWeb.ChainController do
  use MyAppWeb, :controller

  def show(conn, %{"id" => chain_id}) do
    chain = ChainStorage.load_from_db(chain_id)

    # Generate HTML fragment (without <html>, <head>, <body>)
    html_fragment = generate_embeddable_html(chain)

    render(conn, "show.html",
      chain: chain,
      visualization_html: html_fragment
    )
  end

  defp generate_embeddable_html(chain) do
    # Extract just the content, not full HTML page
    CausalTrace.Viewer.generate_html(chain)
    |> extract_body_content()
  end

  defp extract_body_content(html) do
    # Parse HTML and extract <body> contents
    html
    |> Floki.parse_document!()
    |> Floki.find("body")
    |> Floki.children()
    |> Floki.raw_html()
  end
end
```

### Interactive Features

The HTML visualization includes JavaScript for interactivity:

**1. Event Filtering**

```javascript
function filterEvents() {
  const typeFilter = document.getElementById('typeFilter').value;
  const confidenceThreshold = document.getElementById('confidenceSlider').value / 100;
  const events = document.querySelectorAll('.event');

  events.forEach(event => {
    const eventType = event.dataset.type;
    const eventConfidence = parseFloat(event.dataset.confidence);

    const typeMatch = typeFilter === 'all' || eventType === typeFilter;
    const confidenceMatch = eventConfidence >= confidenceThreshold;

    if (typeMatch && confidenceMatch) {
      event.classList.remove('hidden');
    } else {
      event.classList.add('hidden');
    }
  });
}
```

**2. Event Highlighting**

```javascript
// Highlight related events
function highlightRelatedEvents(eventId) {
  const event = document.querySelector(`[data-id="${eventId}"]`);
  const codeSection = event.dataset.codeSection;

  // Highlight all events related to same code section
  document.querySelectorAll(`[data-code-section="${codeSection}"]`).forEach(el => {
    el.classList.add('highlighted');
  });
}
```

**3. Export Functionality**

```javascript
function exportAsJSON() {
  const chainData = JSON.parse(document.getElementById('chain-data').textContent);
  const blob = new Blob([JSON.stringify(chainData, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `chain_${chainData.id}.json`;
  a.click();
}
```

---

## Analysis Techniques

### Pattern Mining

```elixir
defmodule PatternMiner do
  @doc """
  Mine common reasoning patterns from multiple chains.
  """
  def mine_patterns(chains) do
    # Extract event sequences
    sequences =
      Enum.map(chains, fn chain ->
        Enum.map(chain.events, & &1.type)
      end)

    # Find frequent subsequences
    frequent_patterns = find_frequent_subsequences(sequences, min_support: 0.3)

    # Analyze pattern success rates
    pattern_analysis =
      Enum.map(frequent_patterns, fn pattern ->
        chains_with_pattern = filter_chains_with_pattern(chains, pattern)
        success_rate = calculate_success_rate(chains_with_pattern)

        %{
          pattern: pattern,
          frequency: length(chains_with_pattern),
          success_rate: success_rate,
          avg_confidence: average_confidence(chains_with_pattern)
        }
      end)

    # Sort by success rate
    Enum.sort_by(pattern_analysis, & &1.success_rate, :desc)
  end

  defp find_frequent_subsequences(sequences, opts) do
    min_support = Keyword.get(opts, :min_support, 0.3)

    # Generate candidate patterns (length 2-5)
    candidates =
      for len <- 2..5,
          seq <- sequences,
          subseq <- subsequences(seq, len),
          uniq: true,
          do: subseq

    # Count support for each pattern
    pattern_counts =
      Enum.reduce(candidates, %{}, fn pattern, acc ->
        count =
          Enum.count(sequences, fn seq ->
            contains_subsequence?(seq, pattern)
          end)

        Map.put(acc, pattern, count)
      end)

    # Filter by minimum support
    total_sequences = length(sequences)

    pattern_counts
    |> Enum.filter(fn {_pattern, count} ->
      count / total_sequences >= min_support
    end)
    |> Enum.map(&elem(&1, 0))
  end

  defp subsequences(list, len) do
    list
    |> Enum.chunk_every(len, 1, :discard)
  end

  defp contains_subsequence?(sequence, subsequence) do
    sequence
    |> Enum.chunk_every(length(subsequence), 1, :discard)
    |> Enum.any?(&(&1 == subsequence))
  end

  defp filter_chains_with_pattern(chains, pattern) do
    Enum.filter(chains, fn chain ->
      event_types = Enum.map(chain.events, & &1.type)
      contains_subsequence?(event_types, pattern)
    end)
  end

  defp calculate_success_rate(chains) do
    # Assume chains have success field
    successful = Enum.count(chains, & &1.metadata[:successful])
    successful / length(chains)
  end

  defp average_confidence(chains) do
    all_events = Enum.flat_map(chains, & &1.events)
    confidences = Enum.map(all_events, & &1.confidence)
    Enum.sum(confidences) / length(confidences)
  end
end

# Usage
chains = load_all_chains()  # Load historical chains

patterns = PatternMiner.mine_patterns(chains)

IO.puts("Top Successful Patterns:")
Enum.take(patterns, 5)
|> Enum.each(fn pattern ->
  IO.puts("Pattern: #{inspect(pattern.pattern)}")
  IO.puts("  Frequency: #{pattern.frequency}")
  IO.puts("  Success Rate: #{Float.round(pattern.success_rate * 100, 1)}%")
  IO.puts("  Avg Confidence: #{Float.round(pattern.avg_confidence, 2)}")
end)

# Example output:
# Pattern: [:hypothesis_formed, :constraint_evaluated, :pattern_applied]
#   Frequency: 45
#   Success Rate: 87.3%
#   Avg Confidence: 0.89
```

### Failure Mode Analysis

```elixir
defmodule FailureAnalyzer do
  @doc """
  Analyze chains that led to failures to identify common failure modes.
  """
  def analyze_failures(failed_chains) do
    %{
      low_confidence_failures: analyze_low_confidence(failed_chains),
      ambiguity_failures: analyze_unresolved_ambiguities(failed_chains),
      contradiction_failures: analyze_contradictions(failed_chains),
      missing_patterns: analyze_missing_patterns(failed_chains)
    }
  end

  defp analyze_low_confidence(chains) do
    # Find chains where low confidence correlated with failure
    low_conf_chains =
      Enum.filter(chains, fn chain ->
        avg_conf = CausalTrace.Chain.statistics(chain).avg_confidence
        avg_conf < 0.7
      end)

    %{
      count: length(low_conf_chains),
      percentage: length(low_conf_chains) / length(chains) * 100,
      examples: Enum.take(low_conf_chains, 3) |> Enum.map(&summarize_chain/1)
    }
  end

  defp analyze_unresolved_ambiguities(chains) do
    # Find chains with unresolved ambiguities
    ambiguous_chains =
      Enum.filter(chains, fn chain ->
        ambiguities = CausalTrace.Chain.get_events_by_type(chain, :ambiguity_flagged)
        length(ambiguities) > 0
      end)

    common_ambiguities =
      ambiguous_chains
      |> Enum.flat_map(fn chain ->
        CausalTrace.Chain.get_events_by_type(chain, :ambiguity_flagged)
      end)
      |> Enum.map(& &1.decision)
      |> Enum.frequencies()
      |> Enum.sort_by(&elem(&1, 1), :desc)
      |> Enum.take(10)

    %{
      count: length(ambiguous_chains),
      percentage: length(ambiguous_chains) / length(chains) * 100,
      common_ambiguities: common_ambiguities
    }
  end

  defp analyze_contradictions(chains) do
    # Find contradictory decisions in failed chains
    chains_with_contradictions =
      Enum.filter(chains, fn chain ->
        length(find_contradictions(chain)) > 0
      end)

    %{
      count: length(chains_with_contradictions),
      percentage: length(chains_with_contradictions) / length(chains) * 100,
      examples: Enum.take(chains_with_contradictions, 3) |> Enum.map(&summarize_contradictions/1)
    }
  end

  defp analyze_missing_patterns(chains) do
    # Compare failed chains to successful patterns
    successful_patterns = PatternMiner.mine_patterns(load_successful_chains())

    missing_pattern_chains =
      Enum.filter(chains, fn chain ->
        event_sequence = Enum.map(chain.events, & &1.type)

        # Check if chain lacks common successful patterns
        Enum.all?(successful_patterns, fn pattern ->
          not contains_subsequence?(event_sequence, pattern.pattern)
        end)
      end)

    %{
      count: length(missing_pattern_chains),
      percentage: length(missing_pattern_chains) / length(chains) * 100
    }
  end

  defp summarize_chain(chain) do
    %{
      id: chain.id,
      name: chain.name,
      avg_confidence: CausalTrace.Chain.statistics(chain).avg_confidence
    }
  end

  defp summarize_contradictions(chain) do
    contradictions = find_contradictions(chain)

    %{
      id: chain.id,
      name: chain.name,
      contradictions: Enum.map(contradictions, &"#{&1.event1} vs #{&1.event2}")
    }
  end

  defp find_contradictions(chain) do
    # Simplified contradiction detection
    []
  end

  defp load_successful_chains do
    # Load chains that were successful
    []
  end

  defp contains_subsequence?(sequence, subsequence) do
    sequence
    |> Enum.chunk_every(length(subsequence), 1, :discard)
    |> Enum.any?(&(&1 == subsequence))
  end
end

# Usage
failed_chains = load_failed_chains()
analysis = FailureAnalyzer.analyze_failures(failed_chains)

IO.puts("Failure Mode Analysis:")
IO.puts("Low Confidence: #{analysis.low_confidence_failures.percentage}%")
IO.puts("Unresolved Ambiguities: #{analysis.ambiguity_failures.percentage}%")
IO.puts("Contradictions: #{analysis.contradiction_failures.percentage}%")
IO.puts("Missing Successful Patterns: #{analysis.missing_patterns.percentage}%")
```

### Confidence Calibration

```elixir
defmodule ConfidenceCalibration do
  @doc """
  Analyze if confidence scores are calibrated (do they match actual success rates?).
  """
  def analyze_calibration(chains) do
    # Group chains by confidence bins
    bins = create_bins(10)

    bin_data =
      Enum.map(bins, fn {min, max} ->
        chains_in_bin =
          Enum.filter(chains, fn chain ->
            avg_conf = CausalTrace.Chain.statistics(chain).avg_confidence
            avg_conf >= min and avg_conf < max
          end)

        if Enum.empty?(chains_in_bin) do
          nil
        else
          avg_confidence =
            chains_in_bin
            |> Enum.map(&CausalTrace.Chain.statistics(&1).avg_confidence)
            |> average()

          success_rate =
            chains_in_bin
            |> Enum.count(& &1.metadata[:successful])
            |> Kernel./(length(chains_in_bin))

          %{
            bin: {min, max},
            avg_confidence: avg_confidence,
            success_rate: success_rate,
            count: length(chains_in_bin),
            calibration_error: abs(avg_confidence - success_rate)
          }
        end
      end)
      |> Enum.reject(&is_nil/1)

    expected_calibration_error =
      bin_data
      |> Enum.map(&(&1.calibration_error * &1.count))
      |> Enum.sum()
      |> Kernel./(Enum.map(bin_data, & &1.count) |> Enum.sum())

    %{
      bins: bin_data,
      expected_calibration_error: expected_calibration_error,
      is_well_calibrated: expected_calibration_error < 0.1
    }
  end

  defp create_bins(n) do
    step = 1.0 / n

    Enum.map(0..(n - 1), fn i ->
      {i * step, (i + 1) * step}
    end)
  end

  defp average(list) do
    Enum.sum(list) / length(list)
  end

  @doc """
  Visualize calibration plot.
  """
  def plot_calibration(calibration_data) do
    points =
      calibration_data.bins
      |> Enum.map(fn bin ->
        {bin.avg_confidence, bin.success_rate}
      end)

    # Generate ASCII plot
    generate_ascii_plot(points)
  end

  defp generate_ascii_plot(points) do
    width = 50
    height = 20

    # Create grid
    grid =
      for y <- 0..(height - 1) do
        for x <- 0..(width - 1), do: " "
      end

    # Plot perfect calibration line
    grid =
      Enum.reduce(0..(width - 1), grid, fn x, acc ->
        y = height - 1 - round(x / width * height)
        put_in(acc, [Access.at(y), Access.at(x)], "·")
      end)

    # Plot actual points
    grid =
      Enum.reduce(points, grid, fn {conf, success}, acc ->
        x = round(conf * width)
        y = height - 1 - round(success * height)
        x = min(x, width - 1)
        y = max(0, min(y, height - 1))
        put_in(acc, [Access.at(y), Access.at(x)], "●")
      end)

    # Convert to string
    Enum.map(grid, &Enum.join/1) |> Enum.join("\n")
  end
end

# Usage
chains = load_all_chains()
calibration = ConfidenceCalibration.analyze_calibration(chains)

IO.puts("Calibration Analysis:")
IO.puts("Expected Calibration Error: #{Float.round(calibration.expected_calibration_error, 3)}")
IO.puts("Well Calibrated: #{calibration.is_well_calibrated}")
IO.puts("\nCalibration Plot:")
IO.puts(ConfidenceCalibration.plot_calibration(calibration))
IO.puts("(● = actual, · = perfect calibration)")
```

---

## User Study Protocols

### H5: Transparency Improves Debugging Speed

**Hypothesis**: Developers debug AI-generated code 30% faster with causal traces vs. without.

**Study Design**:

```elixir
defmodule TransparencyStudy.H5 do
  @doc """
  Protocol for H5 user study on debugging speed.
  """

  # Study parameters
  @num_participants 30
  @num_tasks 6  # 3 with traces, 3 without (counterbalanced)
  @task_categories [:authentication, :caching, :data_processing]

  defstruct [
    :participant_id,
    :group,  # :control or :treatment
    :tasks,
    :results,
    :pre_survey,
    :post_survey
  ]

  @doc """
  Run study for a single participant.
  """
  def run_participant_session(participant_id) do
    participant = %__MODULE__{
      participant_id: participant_id,
      group: assign_group(participant_id),
      tasks: generate_tasks(participant_id)
    }

    # Pre-survey (demographics, experience)
    participant = %{participant | pre_survey: conduct_pre_survey()}

    # Training (15 minutes)
    conduct_training(participant.group)

    # Task execution
    results =
      Enum.map(participant.tasks, fn task ->
        execute_task(participant, task)
      end)

    participant = %{participant | results: results}

    # Post-survey (perceived usefulness, workload)
    participant = %{participant | post_survey: conduct_post_survey(participant)}

    # Save data
    save_participant_data(participant)

    participant
  end

  defp assign_group(participant_id) do
    # Balanced random assignment
    if rem(participant_id, 2) == 0, do: :treatment, else: :control
  end

  defp generate_tasks(participant_id) do
    # Generate 6 tasks (3 with bugs, 3 without)
    # Counterbalance order and trace availability

    seed = participant_id
    :rand.seed(:exsss, {seed, seed, seed})

    @task_categories
    |> Enum.flat_map(fn category ->
      [
        %{
          category: category,
          code: generate_buggy_code(category),
          has_trace: Enum.random([true, false]),
          trace: generate_trace(category)
        },
        %{
          category: category,
          code: generate_correct_code(category),
          has_trace: Enum.random([true, false]),
          trace: nil
        }
      ]
    end)
    |> Enum.shuffle()
  end

  defp execute_task(participant, task) do
    IO.puts("\n=== Task #{task.category} ===")
    IO.puts("Code:\n#{task.code}\n")

    if task.has_trace and participant.group == :treatment do
      IO.puts("Causal Trace Available: Yes")
      display_trace(task.trace)
    end

    start_time = System.monotonic_time(:millisecond)

    # Participant debugs the code
    IO.puts("\nInstructions:")
    IO.puts("1. Review the code")
    IO.puts("2. Identify any bugs")
    IO.puts("3. Explain the bug and propose a fix")
    IO.puts("\nPress Enter when ready to submit...")
    IO.gets("")

    end_time = System.monotonic_time(:millisecond)
    duration_ms = end_time - start_time

    # Collect participant's bug report
    bug_report = collect_bug_report()

    # Evaluate correctness
    correctness = evaluate_bug_report(bug_report, task)

    %{
      task: task,
      duration_ms: duration_ms,
      bug_report: bug_report,
      correctness: correctness,
      had_trace: task.has_trace and participant.group == :treatment
    }
  end

  defp conduct_pre_survey do
    %{
      years_experience: ask_integer("Years of programming experience: "),
      familiar_with_elixir: ask_yes_no("Familiar with Elixir?"),
      familiar_with_ai_code_gen: ask_yes_no("Used AI code generation tools?"),
      confidence_debugging: ask_scale("Confidence in debugging (1-5): ", 1..5)
    }
  end

  defp conduct_training(group) do
    IO.puts("\n=== Training Session ===")

    case group do
      :treatment ->
        IO.puts("In this study, you will debug code with access to causal traces.")
        IO.puts("Causal traces show:")
        IO.puts("- Decisions made during code generation")
        IO.puts("- Alternatives considered")
        IO.puts("- Confidence levels")
        IO.puts("\nExample trace:")
        display_example_trace()

      :control ->
        IO.puts("In this study, you will debug code using standard techniques.")
        IO.puts("You will have access to:")
        IO.puts("- The generated code")
        IO.puts("- Test cases")
        IO.puts("- Documentation")
    end

    IO.puts("\nPress Enter when ready to begin...")
    IO.gets("")
  end

  defp conduct_post_survey(participant) do
    IO.puts("\n=== Post-Study Survey ===")

    base_questions = %{
      difficulty: ask_scale("Overall difficulty of tasks (1-5): ", 1..5),
      mental_workload: ask_scale("Mental workload (1-7, NASA-TLX): ", 1..7),
      confidence_in_answers: ask_scale("Confidence in your bug reports (1-5): ", 1..5)
    }

    treatment_questions =
      if participant.group == :treatment do
        %{
          trace_usefulness: ask_scale("How useful were the causal traces? (1-5): ", 1..5),
          trace_understandability: ask_scale("How easy to understand were the traces? (1-5): ", 1..5),
          would_use_again: ask_yes_no("Would you want causal traces in your daily work?"),
          suggestions: ask_text("Suggestions for improving traces:")
        }
      else
        %{}
      end

    Map.merge(base_questions, treatment_questions)
  end

  defp generate_buggy_code(:authentication) do
    """
    defmodule Auth do
      def verify_token(token) do
        case JWT.verify(token) do
          {:ok, claims} ->
            # BUG: No expiration check!
            {:ok, claims["user_id"]}
          {:error, reason} ->
            {:error, reason}
        end
      end
    end
    """
  end

  defp generate_buggy_code(:caching) do
    """
    defmodule Cache do
      def get_or_compute(key, compute_fn) do
        case :ets.lookup(:cache, key) do
          [{^key, value}] ->
            value
          [] ->
            value = compute_fn.()
            # BUG: No error handling for compute_fn!
            :ets.insert(:cache, {key, value})
            value
        end
      end
    end
    """
  end

  defp generate_buggy_code(:data_processing) do
    """
    defmodule DataProcessor do
      def process_batch(items) do
        items
        |> Enum.map(&parse_item/1)
        |> Enum.filter(&valid?/1)
        # BUG: Division by zero if no valid items!
        |> Enum.reduce(0, &+/2)
        |> Kernel./(length(items))
      end
    end
    """
  end

  defp generate_trace(:authentication) do
    chain = CausalTrace.Chain.new("Token Verification")

    chain
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Verify JWT token signature",
        "JWT.verify checks signature validity",
        confidence: 0.92
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :ambiguity_flagged,
        "Unclear: Should check token expiration?",
        "Requirements don't specify expiration handling",
        confidence: 0.55,
        alternatives: ["Check expiration", "Skip expiration check"]
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Skip expiration check",
        "Assuming tokens don't expire (based on common pattern)",
        confidence: 0.62
      )
    )
  end

  defp display_trace(chain) do
    IO.puts("\n--- Causal Trace ---")

    Enum.each(chain.events, fn event ->
      IO.puts("\nEvent: #{event.type}")
      IO.puts("Decision: #{event.decision}")
      IO.puts("Reasoning: #{event.reasoning}")
      IO.puts("Confidence: #{event.confidence}")

      if length(event.alternatives) > 0 do
        IO.puts("Alternatives: #{inspect(event.alternatives)}")
      end
    end)

    IO.puts("-------------------\n")
  end

  defp collect_bug_report do
    %{
      found_bug: ask_yes_no("\nDid you find a bug?"),
      bug_location: ask_text("Where is the bug? (line/function): "),
      bug_description: ask_text("Describe the bug: "),
      proposed_fix: ask_text("How would you fix it? ")
    }
  end

  defp evaluate_bug_report(report, task) do
    # Manual evaluation by researcher
    # In real study, would have ground truth
    %{
      correct_identification: true,  # Did they find the real bug?
      correct_diagnosis: true,       # Did they understand why it's a bug?
      correct_fix: true              # Did they propose a correct fix?
    }
  end

  # Helper functions for user input
  defp ask_integer(prompt) do
    IO.gets(prompt) |> String.trim() |> String.to_integer()
  end

  defp ask_yes_no(prompt) do
    response = IO.gets("#{prompt} (y/n): ") |> String.trim() |> String.downcase()
    response == "y"
  end

  defp ask_scale(prompt, range) do
    value = ask_integer(prompt)

    if value in range do
      value
    else
      IO.puts("Invalid input. Please enter a value in range #{inspect(range)}")
      ask_scale(prompt, range)
    end
  end

  defp ask_text(prompt) do
    IO.gets(prompt) |> String.trim()
  end

  defp display_example_trace do
    example = """
    Example Trace:
    1. Hypothesis Formed: Use GenServer for state
       Reasoning: Need thread-safe state management
       Confidence: 0.85

    2. Alternative Rejected: Use Agent
       Reasoning: GenServer provides more control over state updates
       Confidence: 0.78

    3. Pattern Applied: OTP GenServer pattern
       Reasoning: Standard Elixir pattern for stateful processes
       Confidence: 0.92
    """

    IO.puts(example)
  end

  defp save_participant_data(participant) do
    filename = "study_data/h5/participant_#{participant.participant_id}.json"
    json = Jason.encode!(participant, pretty: true)
    File.write!(filename, json)
  end

  @doc """
  Analyze aggregated results from all participants.
  """
  def analyze_results(participant_files) do
    participants =
      Enum.map(participant_files, fn file ->
        file |> File.read!() |> Jason.decode!(keys: :atoms)
      end)

    # Split by group
    treatment_group = Enum.filter(participants, &(&1.group == :treatment))
    control_group = Enum.filter(participants, &(&1.group == :control))

    # Calculate metrics
    treatment_stats = calculate_group_stats(treatment_group)
    control_stats = calculate_group_stats(control_group)

    # Statistical test
    {t_stat, p_value} = paired_t_test(
      treatment_stats.avg_debug_time,
      control_stats.avg_debug_time,
      treatment_stats.std_debug_time,
      control_stats.std_debug_time,
      length(treatment_group)
    )

    %{
      treatment: treatment_stats,
      control: control_stats,
      t_statistic: t_stat,
      p_value: p_value,
      significant: p_value < 0.05,
      effect_size: cohens_d(treatment_stats, control_stats)
    }
  end

  defp calculate_group_stats(participants) do
    debug_times =
      participants
      |> Enum.flat_map(& &1.results)
      |> Enum.map(& &1.duration_ms)

    correctness_scores =
      participants
      |> Enum.flat_map(& &1.results)
      |> Enum.map(fn result ->
        score =
          (if result.correctness.correct_identification, do: 1, else: 0) +
          (if result.correctness.correct_diagnosis, do: 1, else: 0) +
          (if result.correctness.correct_fix, do: 1, else: 0)
        score / 3
      end)

    %{
      avg_debug_time: average(debug_times),
      std_debug_time: std_dev(debug_times),
      avg_correctness: average(correctness_scores),
      median_debug_time: median(debug_times)
    }
  end

  defp average(list), do: Enum.sum(list) / length(list)

  defp std_dev(list) do
    mean = average(list)
    variance = Enum.map(list, &:math.pow(&1 - mean, 2)) |> average()
    :math.sqrt(variance)
  end

  defp median(list) do
    sorted = Enum.sort(list)
    mid = div(length(sorted), 2)

    if rem(length(sorted), 2) == 0 do
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
    else
      Enum.at(sorted, mid)
    end
  end

  defp paired_t_test(mean1, mean2, std1, std2, n) do
    # Simplified t-test
    pooled_std = :math.sqrt((std1 * std1 + std2 * std2) / 2)
    t_stat = (mean1 - mean2) / (pooled_std * :math.sqrt(2 / n))

    # Approximate p-value (would use proper t-distribution in real study)
    df = 2 * n - 2
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    {t_stat, p_value}
  end

  defp cohens_d(stats1, stats2) do
    pooled_std = :math.sqrt((stats1.std_debug_time ** 2 + stats2.std_debug_time ** 2) / 2)
    (stats1.avg_debug_time - stats2.avg_debug_time) / pooled_std
  end

  defp normal_cdf(x) do
    # Approximation of standard normal CDF
    0.5 * (1 + :math.erf(x / :math.sqrt(2)))
  end
end
```

### H6: Transparency Improves Trust

**Hypothesis**: Developers trust AI-generated code 25% more when causal traces are provided.

**Study Design**:

```elixir
defmodule TransparencyStudy.H6 do
  @doc """
  Protocol for H6 user study on trust in AI-generated code.
  """

  @num_participants 40
  @num_scenarios 8  # 4 with traces, 4 without

  defstruct [
    :participant_id,
    :scenarios,
    :trust_ratings,
    :adoption_decisions,
    :qualitative_feedback
  ]

  @doc """
  Run trust study for a participant.
  """
  def run_trust_study(participant_id) do
    participant = %__MODULE__{
      participant_id: participant_id,
      scenarios: generate_scenarios(participant_id)
    }

    IO.puts("\n=== AI Code Trust Study ===")
    IO.puts("You will review #{length(participant.scenarios)} code snippets")
    IO.puts("generated by AI. For each, you will:")
    IO.puts("1. Rate your trust in the code")
    IO.puts("2. Decide if you would use it in production")
    IO.puts("3. Explain your reasoning")
    IO.puts("\nPress Enter to begin...")
    IO.gets("")

    results =
      Enum.map(participant.scenarios, fn scenario ->
        present_scenario_and_collect_responses(scenario)
      end)

    participant = %{
      participant
      | trust_ratings: Enum.map(results, & &1.trust_rating),
        adoption_decisions: Enum.map(results, & &1.would_adopt),
        qualitative_feedback: Enum.map(results, & &1.feedback)
    }

    # Save data
    save_trust_study_data(participant)

    participant
  end

  defp generate_scenarios(participant_id) do
    seed = participant_id
    :rand.seed(:exsss, {seed, seed, seed})

    [
      # Scenario 1: High quality code, with trace
      %{
        id: 1,
        code: generate_high_quality_code(:authentication),
        has_trace: true,
        trace: generate_confident_trace(:authentication),
        quality: :high
      },

      # Scenario 2: High quality code, without trace
      %{
        id: 2,
        code: generate_high_quality_code(:caching),
        has_trace: false,
        trace: nil,
        quality: :high
      },

      # Scenario 3: Medium quality (with subtle bug), with trace showing low confidence
      %{
        id: 3,
        code: generate_medium_quality_code(:api),
        has_trace: true,
        trace: generate_uncertain_trace(:api),
        quality: :medium
      },

      # Scenario 4: Medium quality, without trace
      %{
        id: 4,
        code: generate_medium_quality_code(:data_processing),
        has_trace: false,
        trace: nil,
        quality: :medium
      },

      # Scenario 5: Low quality (multiple bugs), with trace showing contradictions
      %{
        id: 5,
        code: generate_low_quality_code(:database),
        has_trace: true,
        trace: generate_contradictory_trace(:database),
        quality: :low
      },

      # Scenario 6: Low quality, without trace
      %{
        id: 6,
        code: generate_low_quality_code(:validation),
        has_trace: false,
        trace: nil,
        quality: :low
      },

      # Additional scenarios...
      # (Total 8, balanced between with/without traces and quality levels)
    ]
    |> Enum.shuffle()
  end

  defp present_scenario_and_collect_responses(scenario) do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Scenario #{scenario.id}")
    IO.puts(String.duplicate("=", 60))

    # Show code
    IO.puts("\nGenerated Code:")
    IO.puts(scenario.code)

    # Show trace if available
    if scenario.has_trace do
      IO.puts("\n--- Causal Trace ---")
      display_trace_summary(scenario.trace)
    else
      IO.puts("\n(No causal trace available)")
    end

    # Collect trust rating
    IO.puts("\n")
    trust_rating = ask_scale(
      "Trust rating (1=no trust, 7=complete trust): ",
      1..7
    )

    # Collect adoption decision
    would_adopt = ask_yes_no("Would you use this code in production?")

    # Collect reasoning
    IO.puts("\nPlease explain your trust rating and adoption decision:")
    feedback = ask_multiline_text()

    # Additional questions
    if scenario.has_trace do
      trace_influence = ask_scale(
        "How much did the trace influence your decision? (1=not at all, 5=very much): ",
        1..5
      )

      trace_helpful = ask_yes_no("Was the trace helpful?")
    else
      trace_influence = nil
      trace_helpful = nil
    end

    %{
      scenario_id: scenario.id,
      had_trace: scenario.has_trace,
      quality: scenario.quality,
      trust_rating: trust_rating,
      would_adopt: would_adopt,
      feedback: feedback,
      trace_influence: trace_influence,
      trace_helpful: trace_helpful
    }
  end

  defp generate_confident_trace(category) do
    chain = CausalTrace.Chain.new("#{category} implementation")

    chain
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Use standard security pattern",
        "Well-established pattern with proven security",
        confidence: 0.95
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :pattern_applied,
        "Apply bcrypt for password hashing",
        "Industry standard, OWASP recommended",
        confidence: 0.98
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :constraint_evaluated,
        "Cost factor set to 12",
        "Balances security and performance per OWASP guidelines",
        confidence: 0.93
      )
    )
  end

  defp generate_uncertain_trace(category) do
    chain = CausalTrace.Chain.new("#{category} implementation")

    chain
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Implement rate limiting",
        "Requirements mention 'prevent abuse' but specifics unclear",
        confidence: 0.58,
        alternatives: ["Token bucket", "Fixed window", "Sliding window"]
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :ambiguity_flagged,
        "Rate limit threshold unclear",
        "No specification of requests per time period",
        confidence: 0.45
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Assume 100 requests/minute threshold",
        "Based on common API practices, but NOT confirmed",
        confidence: 0.52
      )
    )
  end

  defp generate_contradictory_trace(category) do
    chain = CausalTrace.Chain.new("#{category} implementation")

    chain
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Use connection pooling for efficiency",
        "Reuse database connections",
        confidence: 0.82
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :alternative_rejected,
        "Don't use connection pooling",
        "Actually, requirements say 'simple implementation'",
        confidence: 0.65
      )
    )
    |> CausalTrace.Chain.add_event(
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Use direct connections without pooling",
        "Contradicts earlier decision, may cause performance issues",
        confidence: 0.48
      )
    )
  end

  defp display_trace_summary(chain) do
    Enum.each(chain.events, fn event ->
      confidence_marker =
        cond do
          event.confidence >= 0.8 -> "✓"
          event.confidence >= 0.6 -> "~"
          true -> "⚠"
        end

      IO.puts("#{confidence_marker} #{event.type}: #{event.decision}")
      IO.puts("   Confidence: #{Float.round(event.confidence, 2)}")
      IO.puts("   #{event.reasoning}")

      if length(event.alternatives) > 0 do
        IO.puts("   Alternatives: #{Enum.join(event.alternatives, ", ")}")
      end

      IO.puts("")
    end)
  end

  defp ask_multiline_text do
    IO.puts("(Enter your response. Type 'DONE' on a new line when finished.)")

    Stream.repeatedly(fn -> IO.gets("") end)
    |> Enum.take_while(&(&1 |> String.trim() != "DONE"))
    |> Enum.join("")
    |> String.trim()
  end

  defp save_trust_study_data(participant) do
    filename = "study_data/h6/participant_#{participant.participant_id}.json"
    json = Jason.encode!(participant, pretty: true)
    File.write!(filename, json)
  end

  @doc """
  Analyze trust study results.
  """
  def analyze_trust_results(participant_files) do
    participants =
      Enum.map(participant_files, fn file ->
        file |> File.read!() |> Jason.decode!(keys: :atoms)
      end)

    # Extract all scenario results
    all_results =
      Enum.flat_map(participants, fn p ->
        Enum.zip([
          Enum.map(p.scenarios, & &1.has_trace),
          p.trust_ratings,
          p.adoption_decisions
        ])
        |> Enum.map(fn {had_trace, trust, adopt} ->
          %{had_trace: had_trace, trust: trust, would_adopt: adopt}
        end)
      end)

    # Split by trace availability
    with_trace = Enum.filter(all_results, & &1.had_trace)
    without_trace = Enum.reject(all_results, & &1.had_trace)

    # Calculate statistics
    with_trace_stats = %{
      avg_trust: average(Enum.map(with_trace, & &1.trust)),
      std_trust: std_dev(Enum.map(with_trace, & &1.trust)),
      adoption_rate: Enum.count(with_trace, & &1.would_adopt) / length(with_trace)
    }

    without_trace_stats = %{
      avg_trust: average(Enum.map(without_trace, & &1.trust)),
      std_trust: std_dev(Enum.map(without_trace, & &1.trust)),
      adoption_rate: Enum.count(without_trace, & &1.would_adopt) / length(without_trace)
    }

    # Statistical test
    {t_stat, p_value} = independent_t_test(
      with_trace_stats.avg_trust,
      without_trace_stats.avg_trust,
      with_trace_stats.std_trust,
      without_trace_stats.std_trust,
      length(with_trace),
      length(without_trace)
    )

    trust_increase_percentage =
      (with_trace_stats.avg_trust - without_trace_stats.avg_trust) /
      without_trace_stats.avg_trust * 100

    %{
      with_trace: with_trace_stats,
      without_trace: without_trace_stats,
      trust_increase_percentage: trust_increase_percentage,
      t_statistic: t_stat,
      p_value: p_value,
      significant: p_value < 0.05,
      hypothesis_supported: trust_increase_percentage >= 25 and p_value < 0.05
    }
  end

  defp average(list), do: Enum.sum(list) / length(list)

  defp std_dev(list) do
    mean = average(list)
    variance = Enum.map(list, &:math.pow(&1 - mean, 2)) |> average()
    :math.sqrt(variance)
  end

  defp independent_t_test(mean1, mean2, std1, std2, n1, n2) do
    pooled_std = :math.sqrt(
      ((n1 - 1) * std1 * std1 + (n2 - 1) * std2 * std2) / (n1 + n2 - 2)
    )

    se = pooled_std * :math.sqrt(1 / n1 + 1 / n2)
    t_stat = (mean1 - mean2) / se

    df = n1 + n2 - 2
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    {t_stat, p_value}
  end

  defp normal_cdf(x) do
    0.5 * (1 + :math.erf(x / :math.sqrt(2)))
  end

  # Helper functions (same as H5)
  defp ask_scale(prompt, range) do
    # ... (implementation from H5)
  end

  defp ask_yes_no(prompt) do
    # ... (implementation from H5)
  end

  # Code generation helpers
  defp generate_high_quality_code(_category) do
    # ... generate bug-free, well-structured code
  end

  defp generate_medium_quality_code(_category) do
    # ... generate code with subtle issues
  end

  defp generate_low_quality_code(_category) do
    # ... generate code with obvious problems
  end
end
```

### Running the Studies

```elixir
# H5: Debugging speed study
participants_h5 = 1..30

Enum.each(participants_h5, fn participant_id ->
  TransparencyStudy.H5.run_participant_session(participant_id)
end)

# Analyze results
h5_files = Path.wildcard("study_data/h5/*.json")
h5_results = TransparencyStudy.H5.analyze_results(h5_files)

IO.puts("H5 Results: Debugging Speed")
IO.puts("Treatment group avg time: #{h5_results.treatment.avg_debug_time}ms")
IO.puts("Control group avg time: #{h5_results.control.avg_debug_time}ms")
IO.puts("Speed improvement: #{((1 - h5_results.treatment.avg_debug_time / h5_results.control.avg_debug_time) * 100)}%")
IO.puts("p-value: #{h5_results.p_value}")
IO.puts("Significant: #{h5_results.significant}")

# H6: Trust study
participants_h6 = 1..40

Enum.each(participants_h6, fn participant_id ->
  TransparencyStudy.H6.run_trust_study(participant_id)
end)

# Analyze results
h6_files = Path.wildcard("study_data/h6/*.json")
h6_results = TransparencyStudy.H6.analyze_trust_results(h6_files)

IO.puts("\nH6 Results: Trust")
IO.puts("With trace avg trust: #{h6_results.with_trace.avg_trust}")
IO.puts("Without trace avg trust: #{h6_results.without_trace.avg_trust}")
IO.puts("Trust increase: #{Float.round(h6_results.trust_increase_percentage, 1)}%")
IO.puts("p-value: #{h6_results.p_value}")
IO.puts("Hypothesis supported: #{h6_results.hypothesis_supported}")
```

---

## Integration with Code Generation

### Instrumenting LLM Calls

```elixir
defmodule InstrumentedCodeGenerator do
  @doc """
  Generate code with causal tracing.
  """
  def generate_with_tracing(specification) do
    chain = CausalTrace.Chain.new(
      "Code Generation: #{specification.task_name}",
      metadata: %{
        spec_id: specification.id,
        start_time: DateTime.utc_now()
      }
    )

    # Parse specification
    {chain, requirements} = parse_specification(chain, specification)

    # Generate architecture
    {chain, architecture} = design_architecture(chain, requirements)

    # Implement modules
    {chain, code} = implement_modules(chain, architecture)

    # Final review
    chain = review_and_finalize(chain, code)

    %{
      code: code,
      chain: chain,
      metadata: %{
        end_time: DateTime.utc_now(),
        total_events: length(chain.events)
      }
    }
  end

  defp parse_specification(chain, spec) do
    # Extract requirements from specification
    requirements = extract_requirements(spec.text)

    # Log as events
    chain =
      Enum.reduce(requirements, chain, fn req, acc_chain ->
        CausalTrace.Chain.add_event(acc_chain,
          CausalTrace.Event.new(
            :constraint_evaluated,
            "Requirement: #{req.description}",
            "Parsed from specification section #{req.section}",
            confidence: req.clarity_score,
            spec_reference: req.section
          )
        )
      end)

    # Flag ambiguities
    ambiguous_reqs = Enum.filter(requirements, &(&1.clarity_score < 0.7))

    chain =
      Enum.reduce(ambiguous_reqs, chain, fn req, acc_chain ->
        CausalTrace.Chain.add_event(acc_chain,
          CausalTrace.Event.new(
            :ambiguity_flagged,
            "Unclear requirement: #{req.description}",
            "Low clarity score (#{req.clarity_score}). May need clarification.",
            confidence: req.clarity_score,
            alternatives: suggest_interpretations(req)
          )
        )
      end)

    {chain, requirements}
  end

  defp design_architecture(chain, requirements) do
    # Propose architecture
    architecture_options = generate_architecture_options(requirements)

    # Evaluate each option
    {chain, evaluations} =
      Enum.reduce(architecture_options, {chain, []}, fn option, {acc_chain, acc_evals} ->
        evaluation = evaluate_architecture(option, requirements)

        event = CausalTrace.Event.new(
          if evaluation.score == Enum.max_by(architecture_options, &evaluate_architecture(&1, requirements).score).score do
            :hypothesis_formed
          else
            :alternative_rejected
          end,
          "Architecture option: #{option.name}",
          evaluation.reasoning,
          confidence: evaluation.score,
          alternatives: Enum.map(architecture_options, & &1.name) -- [option.name]
        )

        {CausalTrace.Chain.add_event(acc_chain, event), [evaluation | acc_evals]}
      end)

    # Select best architecture
    best = Enum.max_by(evaluations, & &1.score)

    {chain, best.architecture}
  end

  defp implement_modules(chain, architecture) do
    {chain, modules} =
      Enum.reduce(architecture.modules, {chain, []}, fn module_spec, {acc_chain, acc_modules} ->
        {updated_chain, module_code} = implement_module(acc_chain, module_spec)
        {updated_chain, [module_code | acc_modules]}
      end)

    code = Enum.reverse(modules) |> Enum.join("\n\n")
    {chain, code}
  end

  defp implement_module(chain, module_spec) do
    # Log hypothesis
    chain = CausalTrace.Chain.add_event(chain,
      CausalTrace.Event.new(
        :hypothesis_formed,
        "Implement #{module_spec.name}",
        "Using pattern: #{module_spec.pattern}",
        confidence: module_spec.confidence,
        code_section: "defmodule #{module_spec.name}"
      )
    )

    # Generate code (call to LLM)
    code = call_llm_for_module(module_spec)

    # Validate against requirements
    validation = validate_module(code, module_spec)

    chain = CausalTrace.Chain.add_event(chain,
      CausalTrace.Event.new(
        :constraint_evaluated,
        "Validate #{module_spec.name}",
        "Validation result: #{validation.summary}",
        confidence: validation.confidence
      )
    )

    # If validation issues, log them
    chain =
      if validation.confidence < 0.7 do
        CausalTrace.Chain.add_event(chain,
          CausalTrace.Event.new(
            :confidence_updated,
            "Low confidence in #{module_spec.name}",
            "Issues: #{Enum.join(validation.issues, ", ")}",
            confidence: validation.confidence,
            metadata: %{
              issues: validation.issues,
              suggestions: validation.suggestions
            }
          )
        )
      else
        chain
      end

    {chain, code}
  end

  defp review_and_finalize(chain, code) do
    # Final confidence assessment
    stats = CausalTrace.Chain.statistics(chain)

    chain = CausalTrace.Chain.add_event(chain,
      CausalTrace.Event.new(
        :confidence_updated,
        "Final review",
        """
        Generated code with #{stats.total_events} decisions.
        Average confidence: #{Float.round(stats.avg_confidence, 2)}
        #{if stats.avg_confidence < 0.7 do
          "⚠ REVIEW RECOMMENDED: Below confidence threshold"
        else
          "✓ Code meets confidence standards"
        end}
        """,
        confidence: stats.avg_confidence
      )
    )

    chain
  end

  # Helper functions
  defp extract_requirements(text) do
    # Parse specification text into structured requirements
    # Returns list of %{description, section, clarity_score}
    []
  end

  defp suggest_interpretations(requirement) do
    # Suggest possible interpretations for ambiguous requirement
    []
  end

  defp generate_architecture_options(requirements) do
    # Generate multiple architecture proposals
    []
  end

  defp evaluate_architecture(option, requirements) do
    # Evaluate architecture against requirements
    # Returns %{score, reasoning, architecture}
    %{score: 0.5, reasoning: "", architecture: option}
  end

  defp call_llm_for_module(module_spec) do
    # Call LLM to generate module code
    ""
  end

  defp validate_module(code, module_spec) do
    # Validate generated code
    # Returns %{confidence, summary, issues, suggestions}
    %{confidence: 0.8, summary: "", issues: [], suggestions: []}
  end
end

# Usage
spec = %{
  id: "TASK-123",
  task_name: "User Authentication API",
  text: """
  Implement user authentication with:
  - JWT tokens
  - Password hashing
  - Rate limiting (threshold TBD)
  - Session management
  """
}

result = InstrumentedCodeGenerator.generate_with_tracing(spec)

# Save code and chain
File.write!("generated_code.ex", result.code)
ChainStorage.save(result.chain, "traces/task_123_chain.json")

# Generate report
CausalTrace.Viewer.save_html(result.chain, "reports/task_123.html")

# Analyze chain
analysis = ChainAnalyzer.analyze(result.chain)

if length(analysis.issues) > 0 do
  IO.puts("⚠ Generated code has issues. Review recommended:")
  Enum.each(analysis.issues, fn issue ->
    IO.puts("  - #{issue.message}")
  end)
end
```

---

## Advanced Debugging

### Debugging with Causal Traces

```elixir
defmodule CausalDebugger do
  @doc """
  Debug generated code using its causal trace.
  """
  def debug(code, chain, error) do
    IO.puts("=== Causal Debugging Session ===")
    IO.puts("Error: #{inspect(error)}")
    IO.puts("\nAnalyzing causal trace...")

    # Find relevant events
    relevant_events = find_events_related_to_error(chain, error)

    if Enum.empty?(relevant_events) do
      IO.puts("No directly relevant events found in trace.")
      IO.puts("This may indicate an issue not anticipated during generation.")
    else
      IO.puts("\nRelevant decisions:")

      Enum.each(relevant_events, fn event ->
        display_event_debug_info(event)
      end)
    end

    # Check for low confidence decisions
    low_conf = CausalTrace.Chain.find_low_confidence(chain, 0.7)

    if length(low_conf) > 0 do
      IO.puts("\n⚠ Found #{length(low_conf)} low-confidence decisions:")

      Enum.each(low_conf, fn event ->
        IO.puts("\n  Confidence: #{event.confidence}")
        IO.puts("  Decision: #{event.decision}")
        IO.puts("  Reasoning: #{event.reasoning}")
        IO.puts("  Alternatives: #{inspect(event.alternatives)}")
      end)

      IO.puts("\nSuggestion: Review these decisions and consider alternatives.")
    end

    # Check for ambiguities
    ambiguities = CausalTrace.Chain.get_events_by_type(chain, :ambiguity_flagged)

    if length(ambiguities) > 0 do
      IO.puts("\n⚠ Found #{length(ambiguities)} unresolved ambiguities:")

      Enum.each(ambiguities, fn event ->
        IO.puts("\n  Ambiguity: #{event.decision}")
        IO.puts("  Reasoning: #{event.reasoning}")
        IO.puts("  Possible interpretations: #{inspect(event.alternatives)}")
      end)

      IO.puts("\nSuggestion: Clarify requirements and regenerate.")
    end

    # Suggest fixes
    suggestions = generate_fix_suggestions(chain, error, relevant_events)

    if length(suggestions) > 0 do
      IO.puts("\n=== Suggested Fixes ===")

      Enum.with_index(suggestions, 1)
      |> Enum.each(fn {suggestion, idx} ->
        IO.puts("\n#{idx}. #{suggestion.description}")
        IO.puts("   Rationale: #{suggestion.rationale}")
        IO.puts("   Confidence: #{suggestion.confidence}")

        if suggestion.code_change do
          IO.puts("   Code change:")
          IO.puts("   ```")
          IO.puts("   #{suggestion.code_change}")
          IO.puts("   ```")
        end
      end)
    end
  end

  defp find_events_related_to_error(chain, error) do
    # Extract keywords from error message
    error_keywords = extract_keywords(inspect(error))

    # Find events that mention these keywords
    chain.events
    |> Enum.filter(fn event ->
      text = "#{event.decision} #{event.reasoning}"
      Enum.any?(error_keywords, &String.contains?(String.downcase(text), &1))
    end)
  end

  defp extract_keywords(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.filter(&(String.length(&1) > 3))
    |> Enum.take(10)
  end

  defp display_event_debug_info(event) do
    IO.puts("\n--- Event: #{event.type} ---")
    IO.puts("Decision: #{event.decision}")
    IO.puts("Reasoning: #{event.reasoning}")
    IO.puts("Confidence: #{event.confidence}")

    if event.code_section do
      IO.puts("Code section: #{event.code_section}")
    end

    if length(event.alternatives) > 0 do
      IO.puts("Alternatives considered:")
      Enum.each(event.alternatives, fn alt ->
        IO.puts("  - #{alt}")
      end)
    end

    if event.confidence < 0.7 do
      IO.puts("\n⚠ LOW CONFIDENCE - This decision was uncertain!")
    end
  end

  defp generate_fix_suggestions(chain, error, relevant_events) do
    suggestions = []

    # Suggestion 1: If low confidence events, try alternatives
    suggestions =
      if Enum.any?(relevant_events, &(&1.confidence < 0.7)) do
        low_conf_event = Enum.find(relevant_events, &(&1.confidence < 0.7))

        [
          %{
            description: "Try alternative: #{List.first(low_conf_event.alternatives)}",
            rationale: "Original decision had low confidence (#{low_conf_event.confidence})",
            confidence: 0.7,
            code_change: nil
          }
          | suggestions
        ]
      else
        suggestions
      end

    # Suggestion 2: If ambiguity, suggest clarification
    ambiguous = Enum.filter(relevant_events, &(&1.type == :ambiguity_flagged))

    suggestions =
      if length(ambiguous) > 0 do
        amb_event = List.first(ambiguous)

        [
          %{
            description: "Clarify ambiguous requirement",
            rationale: "Ambiguity flagged: #{amb_event.decision}",
            confidence: 0.8,
            code_change: nil
          }
          | suggestions
        ]
      else
        suggestions
      end

    # Additional domain-specific suggestions based on error type
    suggestions ++ domain_specific_suggestions(error)
  end

  defp domain_specific_suggestions(error) do
    # Placeholder for domain-specific suggestion logic
    []
  end
end

# Example usage
code = """
defmodule Auth do
  def verify_token(token) do
    JWT.verify(token)
  end
end
"""

chain = ChainStorage.load("traces/auth_chain.json")

error = %ArgumentError{message: "Token expired"}

CausalDebugger.debug(code, chain, error)

# Output:
# === Causal Debugging Session ===
# Error: %ArgumentError{message: "Token expired"}
#
# Analyzing causal trace...
#
# Relevant decisions:
#
# --- Event: ambiguity_flagged ---
# Decision: Unclear: Should check token expiration?
# Reasoning: Requirements don't specify expiration handling
# Confidence: 0.55
# Alternatives considered:
#   - Check expiration
#   - Skip expiration check
#
# ⚠ LOW CONFIDENCE - This decision was uncertain!
#
# ⚠ Found 1 unresolved ambiguities:
#
#   Ambiguity: Unclear: Should check token expiration?
#   Reasoning: Requirements don't specify expiration handling
#   Possible interpretations: ["Check expiration", "Skip expiration check"]
#
# Suggestion: Clarify requirements and regenerate.
#
# === Suggested Fixes ===
#
# 1. Clarify ambiguous requirement
#    Rationale: Ambiguity flagged: Unclear: Should check token expiration?
#    Confidence: 0.8
```

---

## Research Applications

### Measuring Transparency Impact

```elixir
defmodule TransparencyMetrics do
  @doc """
  Measure the impact of transparency on code quality and developer experience.
  """
  def measure_transparency_impact(with_traces, without_traces) do
    %{
      debugging_speed: measure_debugging_speed(with_traces, without_traces),
      trust_levels: measure_trust(with_traces, without_traces),
      code_quality: measure_quality(with_traces, without_traces),
      developer_satisfaction: measure_satisfaction(with_traces, without_traces)
    }
  end

  defp measure_debugging_speed(with_traces, without_traces) do
    with_avg = average_debug_time(with_traces)
    without_avg = average_debug_time(without_traces)

    improvement = (without_avg - with_avg) / without_avg * 100

    %{
      with_traces_ms: with_avg,
      without_traces_ms: without_avg,
      improvement_percentage: improvement
    }
  end

  defp measure_trust(with_traces, without_traces) do
    with_trust = average_trust_score(with_traces)
    without_trust = average_trust_score(without_traces)

    increase = (with_trust - without_trust) / without_trust * 100

    %{
      with_traces_score: with_trust,
      without_traces_score: without_trust,
      increase_percentage: increase
    }
  end

  defp measure_quality(with_traces, without_traces) do
    with_quality = assess_code_quality(with_traces)
    without_quality = assess_code_quality(without_traces)

    %{
      with_traces: with_quality,
      without_traces: without_quality,
      difference: with_quality - without_quality
    }
  end

  defp measure_satisfaction(with_traces, without_traces) do
    with_sat = average_satisfaction(with_traces)
    without_sat = average_satisfaction(without_traces)

    %{
      with_traces_score: with_sat,
      without_traces_score: without_sat,
      difference: with_sat - without_sat
    }
  end

  # Helper functions
  defp average_debug_time(data) do
    data
    |> Enum.map(& &1.debug_time_ms)
    |> average()
  end

  defp average_trust_score(data) do
    data
    |> Enum.map(& &1.trust_score)
    |> average()
  end

  defp assess_code_quality(data) do
    # Composite quality score
    data
    |> Enum.map(fn item ->
      (item.correctness + item.maintainability + item.readability) / 3
    end)
    |> average()
  end

  defp average_satisfaction(data) do
    data
    |> Enum.map(& &1.satisfaction_score)
    |> average()
  end

  defp average(list) do
    if length(list) > 0 do
      Enum.sum(list) / length(list)
    else
      0.0
    end
  end
end
```

---

## API Reference

### CausalTrace.Chain

```elixir
# Create new chain
@spec new(String.t(), keyword()) :: t()

# Add event
@spec add_event(t(), Event.t()) :: t()

# Query events
@spec get_events_by_type(t(), atom()) :: [Event.t()]
@spec find_low_confidence(t(), float()) :: [Event.t()]
@spec find_decision_points(t()) :: [map()]

# Statistics
@spec statistics(t()) :: map()

# Serialization
@spec to_map(t()) :: map()
@spec from_map(map()) :: t()
```

### CausalTrace.Event

```elixir
# Create new event
@spec new(event_type(), String.t(), String.t(), keyword()) :: t()

# Validate event
@spec validate(t()) :: {:ok, t()} | {:error, term()}

# Serialization
@spec to_map(t()) :: map()
@spec from_map(map()) :: t()
```

### CausalTrace.Viewer

```elixir
# Generate HTML
@spec generate_html(Chain.t(), keyword()) :: String.t()

# Save to file
@spec save_html(Chain.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}

# Open in browser
@spec open_in_browser(Chain.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
```

---

## Conclusion

Causal transparency transforms opaque AI reasoning into interpretable decision chains. Key benefits:

1. **Debugging**: 30% faster with traces (H5 hypothesis)
2. **Trust**: 25% increased trust with transparency (H6 hypothesis)
3. **Learning**: Developers learn patterns from explicit reasoning
4. **Quality**: Early detection of low-confidence and ambiguous decisions
5. **Research**: Enables systematic study of AI reasoning

For questions or contributions, contact the research team.

---

**Key References**

1. Lipton, Z. C. (2016). "The Mythos of Model Interpretability." arXiv:1606.03490
2. Doshi-Velez, F., & Kim, B. (2017). "Towards A Rigorous Science of Interpretable Machine Learning." arXiv:1702.08608
3. Miller, T. (2019). "Explanation in Artificial Intelligence: Insights from the Social Sciences." Artificial Intelligence, 267:1-38
4. Chen, M., et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374
5. Ribeiro, M. T., et al. (2016). "Why Should I Trust You?" KDD 2016
6. Peng, S., et al. (2023). "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot." arXiv:2302.06590
