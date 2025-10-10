# Instrumentation & Telemetry - Complete Guide

**TelemetryResearch: Research-Grade Observability for AI/ML Experiments**

Version: 0.1.0
Last Updated: 2025-10-08

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Event Collection](#event-collection)
4. [Metrics & Analysis](#metrics--analysis)
5. [Export Formats](#export-formats)
6. [Real-Time Monitoring](#real-time-monitoring)
7. [Experiment Lifecycle](#experiment-lifecycle)
8. [Integration Patterns](#integration-patterns)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)
11. [References](#references)

---

## Overview

TelemetryResearch provides specialized observability infrastructure for rigorous AI/ML experimentation. Unlike production telemetry (focusing on uptime/errors), research telemetry focuses on **complete data capture**, **reproducibility**, and **statistical analysis**.

### Key Features

- **Experiment Isolation**: Run multiple experiments concurrently without data contamination
- **Complete Capture**: No sampling—every event recorded for reproducibility
- **Rich Metadata**: Automatic enrichment with experiment context, timestamps, tags
- **Multiple Exports**: CSV, JSON Lines, Parquet for analysis in Python, R, Julia
- **Statistical Integration**: Built-in descriptive statistics and metrics
- **Zero Overhead**: ETS-based storage for minimal performance impact

### Quick Start

```elixir
# Start an experiment
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "ensemble_vs_single",
  hypothesis: "5-model ensemble achieves >99% reliability",
  condition: "treatment",
  tags: ["accuracy", "reliability"],
  sample_size: 1000
)

# Events are automatically collected
# Run your AI workload...
result = Ensemble.predict("What is 2+2?", models: [:gpt4, :claude, :gemini])

# Stop and export
{:ok, experiment} = TelemetryResearch.stop_experiment(experiment.id)
{:ok, path} = TelemetryResearch.export(experiment.id, :csv)

# Analyze
metrics = TelemetryResearch.calculate_metrics(experiment.id)
IO.inspect metrics.latency.p99
IO.inspect metrics.reliability.success_rate
```

### Design Philosophy

1. **Complete Data Collection**: No sampling by default—full reproducibility
2. **Experiment-Centric**: Organize around scientific experiments, not services
3. **Analysis-Ready**: Export formats optimized for statistical software
4. **Minimal Interference**: Zero impact on system performance

---

## Core Concepts

### Experiment Lifecycle

An experiment represents an isolated scientific study:

**States:**
- `running`: Actively collecting data
- `stopped`: Collection complete, data frozen
- `archived`: Long-term storage

**Lifecycle:**

```
Create → Attach Handlers → Collect Data → Stop → Analyze → Export → Archive
```

### Event Model

Every event contains:

**Core Fields:**
- `event_name`: Telemetry event identifier
- `measurements`: Numeric measurements (duration, count, etc.)
- `metadata`: Contextual information
- `timestamp`: Precise event time

**Enrichment:**
- `experiment_id`: Which experiment
- `condition`: Treatment/control/baseline
- `tags`: Categorization
- `session_id`: Grouping related events

**Example Event:**

```elixir
%{
  event_name: [:ensemble, :predict, :stop],
  measurements: %{
    duration: 450_000  # microseconds
  },
  metadata: %{
    models: [:gpt4, :claude, :gemini],
    strategy: :majority,
    consensus: 1.0,
    cost_usd: 0.00015
  },
  timestamp: ~U[2025-10-08 10:30:45.123456Z],
  experiment_id: "exp_abc123",
  condition: "treatment",
  tags: ["ensemble", "majority_voting"]
}
```

### Storage Backends

**ETS (Default):**
- In-memory
- Extremely fast (millions of writes/sec)
- Lost on restart (use export)
- Perfect for short experiments

**PostgreSQL (Optional):**
- Persistent
- Query with SQL
- Multi-node support
- Long-running studies

---

## Event Collection

### Automatic Collection

TelemetryResearch automatically captures events from integrated libraries:

**Supported Events:**

```elixir
# Ensemble library
[:ensemble, :predict, :start]
[:ensemble, :predict, :stop]
[:ensemble, :predict, :exception]
[:ensemble, :vote, :complete]

# Hedging library
[:hedging, :request, :start]
[:hedging, :request, :stop]
[:hedging, :hedge, :fired]
[:hedging, :hedge, :won]

# LLM requests (via req_llm)
[:req_llm, :request, :start]
[:req_llm, :request, :stop]
[:req_llm, :request, :exception]

# Custom events
[:my_app, :custom, :event]
```

**Configuration:**

```elixir
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "my_experiment",
  metrics_config: %{
    latency: true,       # Track latency metrics
    cost: true,          # Track cost
    tokens: true,        # Track token usage
    success_rate: true,  # Track failures
    custom: [:accuracy, :f1_score]  # Custom metrics
  }
)
```

### Manual Event Emission

```elixir
# Emit custom events
:telemetry.execute(
  [:my_app, :accuracy, :measured],
  %{accuracy: 0.94, f1: 0.91},
  %{model: :gpt4, dataset: :mmlu}
)

# Will be automatically captured if experiment is running
```

### Event Filtering

```elixir
# Filter which events to collect
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "filtered_experiment",
  event_filter: fn event_name, _measurements, metadata ->
    # Only collect GPT-4 events
    metadata[:model] == :gpt4
  end
)
```

### Sampling (When Needed)

```elixir
# Sometimes you need sampling for high-volume events
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "sampled_experiment",
  sampling_rate: 0.1  # Collect 10% of events
)

# Or conditional sampling
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "smart_sample",
  sampling_strategy: fn event_name, measurements, metadata ->
    cond do
      # Always sample errors
      event_name == [:ensemble, :predict, :exception] -> true

      # Always sample slow requests
      measurements[:duration] > 1_000_000 -> true

      # 10% of normal requests
      true -> :rand.uniform() < 0.1
    end
  end
)
```

---

## Metrics & Analysis

### Built-in Metrics

TelemetryResearch calculates comprehensive metrics automatically:

**Latency Metrics:**

```elixir
metrics = TelemetryResearch.calculate_metrics(experiment_id)

metrics.latency
# => %{
#   count: 1000,
#   mean: 450.2,
#   median: 420.0,
#   min: 180.0,
#   max: 2300.0,
#   p50: 420.0,
#   p90: 680.0,
#   p95: 890.0,
#   p99: 1450.0,
#   p99_9: 2100.0,
#   stdev: 234.5,
#   variance: 54990.25
# }
```

**Cost Metrics:**

```elixir
metrics.cost
# => %{
#   total_usd: 1.234,
#   per_request_mean: 0.001234,
#   per_request_median: 0.001100,
#   by_model: %{
#     gpt4: 0.678,
#     claude: 0.456,
#     gemini: 0.100
#   },
#   cost_per_success: 0.00125
# }
```

**Reliability Metrics:**

```elixir
metrics.reliability
# => %{
#   total_requests: 1000,
#   successes: 987,
#   failures: 13,
#   success_rate: 0.987,
#   failure_rate: 0.013,
#   error_types: %{
#     timeout: 8,
#     api_error: 5
#   },
#   mtbf: 76.9  # Mean time between failures
# }
```

**Token Usage:**

```elixir
metrics.tokens
# => %{
#   total_input: 45_678,
#   total_output: 12_345,
#   total: 58_023,
#   avg_input: 45.7,
#   avg_output: 12.3,
#   by_model: %{
#     gpt4: %{input: 20_000, output: 5_000},
#     claude: %{input: 25_678, output: 7_345}
#   }
# }
```

### Custom Metrics

Define your own metric calculators:

```elixir
defmodule MyMetrics do
  @behaviour TelemetryResearch.MetricCalculator

  @impl true
  def calculate(events) do
    # Calculate custom metric
    accuracy_events = Enum.filter(events, fn e ->
      e.event_name == [:my_app, :accuracy, :measured]
    end)

    accuracies = Enum.map(accuracy_events, & &1.measurements.accuracy)

    %{
      accuracy: %{
        mean: Enum.sum(accuracies) / length(accuracies),
        min: Enum.min(accuracies),
        max: Enum.max(accuracies),
        count: length(accuracies)
      }
    }
  end
end

# Register custom calculator
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "with_custom_metrics",
  custom_metrics: [MyMetrics]
)
```

### Time-Series Analysis

```elixir
defmodule TimeSeriesAnalysis do
  def analyze_over_time(experiment_id, metric, window_seconds \\ 60) do
    events = TelemetryResearch.Store.get_all(experiment_id)

    # Group by time windows
    events
    |> Enum.group_by(fn event ->
      DateTime.truncate(event.timestamp, :second)
      |> DateTime.to_unix()
      |> div(window_seconds)
    end)
    |> Enum.map(fn {window, events} ->
      values = Enum.map(events, &extract_metric(&1, metric))

      %{
        window: window * window_seconds,
        mean: Enum.sum(values) / length(values),
        min: Enum.min(values),
        max: Enum.max(values),
        count: length(values)
      }
    end)
    |> Enum.sort_by(& &1.window)
  end

  defp extract_metric(event, :latency) do
    event.measurements[:duration] / 1000
  end

  defp extract_metric(event, :cost) do
    event.metadata[:cost_usd] || 0.0
  end
end

# Analyze latency over time
time_series = TimeSeriesAnalysis.analyze_over_time(
  experiment_id,
  :latency,
  window_seconds: 300  # 5-minute windows
)

# Detect anomalies
anomalies = Enum.filter(time_series, fn window ->
  window.mean > 1000  # Latency spike
end)
```

### Comparative Analysis

```elixir
defmodule ComparativeAnalysis do
  def compare_experiments(exp1_id, exp2_id) do
    metrics1 = TelemetryResearch.calculate_metrics(exp1_id)
    metrics2 = TelemetryResearch.calculate_metrics(exp2_id)

    %{
      latency_improvement: calculate_improvement(
        metrics1.latency.p99,
        metrics2.latency.p99
      ),
      cost_change: calculate_change(
        metrics1.cost.total_usd,
        metrics2.cost.total_usd
      ),
      reliability_delta: metrics2.reliability.success_rate -
                         metrics1.reliability.success_rate,
      statistical_test: Bench.compare(
        extract_latencies(exp1_id),
        extract_latencies(exp2_id)
      )
    }
  end

  defp calculate_improvement(baseline, treatment) do
    (baseline - treatment) / baseline * 100
  end

  defp calculate_change(baseline, treatment) do
    (treatment - baseline) / baseline * 100
  end
end

# Compare control vs treatment
comparison = ComparativeAnalysis.compare_experiments(
  control_exp_id,
  treatment_exp_id
)

# => %{
#   latency_improvement: 35.2,  # 35.2% faster
#   cost_change: -12.5,         # 12.5% cheaper
#   reliability_delta: 0.05,    # 5% more reliable
#   statistical_test: %{p_value: 0.001, significant: true}
# }
```

---

## Export Formats

### CSV Export

**Best for: Excel, R, simple analysis**

```elixir
{:ok, path} = TelemetryResearch.export(
  experiment_id,
  :csv,
  path: "results/experiment.csv"
)

# Output format:
# timestamp,event_name,duration_us,cost_usd,model,strategy,consensus
# 2025-10-08T10:30:45.123Z,ensemble.predict.stop,450000,0.00015,gpt4,majority,1.0
```

**Custom columns:**

```elixir
{:ok, path} = TelemetryResearch.export(
  experiment_id,
  :csv,
  path: "results/custom.csv",
  columns: [:timestamp, :duration_ms, :model, :accuracy],
  transformers: %{
    duration_ms: fn event -> event.measurements.duration / 1000 end,
    accuracy: fn event -> event.metadata[:accuracy] end
  }
)
```

### JSON Lines Export

**Best for: Python, streaming processing, JSON analysis**

```elixir
{:ok, path} = TelemetryResearch.export(
  experiment_id,
  :jsonl,
  path: "results/experiment.jsonl"
)

# Output format (one JSON object per line):
# {"timestamp":"2025-10-08T10:30:45.123Z","event_name":"ensemble.predict.stop",...}
# {"timestamp":"2025-10-08T10:30:46.456Z","event_name":"ensemble.predict.stop",...}
```

**Streaming large datasets:**

```elixir
# For very large datasets, stream to avoid memory issues
TelemetryResearch.Export.stream(experiment_id, :jsonl)
|> Stream.chunk_every(1000)
|> Stream.each(fn chunk ->
  # Process chunk
  analyze_chunk(chunk)
end)
|> Stream.run()
```

### Parquet Export (Planned)

**Best for: Big data, Spark, Pandas, Arrow**

```elixir
# Columnar format for efficient analytics
{:ok, path} = TelemetryResearch.export(
  experiment_id,
  :parquet,
  path: "results/experiment.parquet",
  compression: :snappy
)
```

### Python Integration

**Load in Pandas:**

```python
import pandas as pd

# From CSV
df = pd.read_csv('results/experiment.csv')

# From JSON Lines
df = pd.read_json('results/experiment.jsonl', lines=True)

# Analyze
print(df.describe())
print(df.groupby('model')['duration_us'].mean())

# Visualize
import matplotlib.pyplot as plt
df.boxplot(column='duration_us', by='model')
plt.show()
```

**R Integration:**

```r
library(tidyverse)

# Load data
df <- read_csv("results/experiment.csv")

# Analyze
df %>%
  group_by(model) %>%
  summarize(
    mean_latency = mean(duration_us),
    p95 = quantile(duration_us, 0.95)
  )

# Plot
ggplot(df, aes(x=model, y=duration_us)) +
  geom_boxplot()
```

---

## Real-Time Monitoring

### Live Dashboard

```elixir
defmodule LiveDashboard do
  use GenServer

  def start_link(experiment_id) do
    GenServer.start_link(__MODULE__, experiment_id, name: __MODULE__)
  end

  def init(experiment_id) do
    # Subscribe to experiment events
    :telemetry.attach_many(
      "live-dashboard",
      [
        [:ensemble, :predict, :stop],
        [:hedging, :request, :stop]
      ],
      &handle_event/4,
      experiment_id
    )

    # Schedule periodic refresh
    schedule_refresh()

    {:ok, %{
      experiment_id: experiment_id,
      recent_events: :queue.new(),
      window_size: 100
    }}
  end

  def handle_event(_event_name, measurements, metadata, experiment_id) do
    GenServer.cast(__MODULE__, {:event, measurements, metadata})
  end

  def handle_cast({:event, measurements, metadata}, state) do
    # Add to recent events
    recent = :queue.in({measurements, metadata}, state.recent_events)

    # Trim to window
    recent = if :queue.len(recent) > state.window_size do
      {_, trimmed} = :queue.out(recent)
      trimmed
    else
      recent
    end

    # Calculate rolling metrics
    metrics = calculate_rolling_metrics(recent)

    # Broadcast to connected clients
    Phoenix.PubSub.broadcast(
      MyApp.PubSub,
      "experiment:#{state.experiment_id}",
      {:metrics_update, metrics}
    )

    {:noreply, %{state | recent_events: recent}}
  end

  def handle_info(:refresh, state) do
    # Calculate and broadcast full metrics
    metrics = TelemetryResearch.calculate_metrics(state.experiment_id)

    Phoenix.PubSub.broadcast(
      MyApp.PubSub,
      "experiment:#{state.experiment_id}",
      {:full_metrics, metrics}
    )

    schedule_refresh()
    {:noreply, state}
  end

  defp schedule_refresh do
    Process.send_after(self(), :refresh, 5_000)  # Every 5s
  end

  defp calculate_rolling_metrics(events) do
    list = :queue.to_list(events)

    latencies = Enum.map(list, fn {m, _} -> m[:duration] / 1000 end)
    costs = Enum.map(list, fn {_, meta} -> meta[:cost_usd] || 0.0 end)

    %{
      recent_count: length(list),
      avg_latency: Enum.sum(latencies) / max(length(latencies), 1),
      total_cost: Enum.sum(costs)
    }
  end
end

# Start dashboard
{:ok, _} = LiveDashboard.start_link(experiment_id)

# In Phoenix LiveView
def mount(_params, %{"experiment_id" => exp_id}, socket) do
  Phoenix.PubSub.subscribe(MyApp.PubSub, "experiment:#{exp_id}")

  {:ok, assign(socket, metrics: %{})}
end

def handle_info({:metrics_update, metrics}, socket) do
  {:noreply, assign(socket, :metrics, metrics)}
end
```

### Alerting

```elixir
defmodule ExperimentAlerting do
  use GenServer

  def start_link(experiment_id, rules) do
    GenServer.start_link(__MODULE__, {experiment_id, rules})
  end

  def init({experiment_id, rules}) do
    # Attach to events
    :telemetry.attach_many(
      "experiment-alerts-#{experiment_id}",
      event_names(rules),
      &handle_event/4,
      %{experiment_id: experiment_id, rules: rules}
    )

    {:ok, %{experiment_id: experiment_id, rules: rules}}
  end

  def handle_event(event_name, measurements, metadata, config) do
    Enum.each(config.rules, fn rule ->
      if rule_triggered?(rule, event_name, measurements, metadata) do
        send_alert(rule, event_name, measurements, metadata)
      end
    end)
  end

  defp rule_triggered?(rule, event_name, measurements, metadata) do
    case rule do
      {:latency_threshold, threshold} ->
        measurements[:duration] > threshold * 1000

      {:error_rate, max_rate} ->
        event_name == [:ensemble, :predict, :exception]

      {:cost_threshold, threshold} ->
        (metadata[:cost_usd] || 0.0) > threshold

      {:custom, checker} ->
        checker.(event_name, measurements, metadata)
    end
  end

  defp send_alert(rule, event_name, measurements, metadata) do
    # Send to alerting service
    Logger.warning("ALERT: #{inspect(rule)} triggered",
      event: event_name,
      measurements: measurements,
      metadata: metadata
    )

    # Could also send to PagerDuty, Slack, etc.
  end
end

# Configure alerting
{:ok, _} = ExperimentAlerting.start_link(experiment_id, [
  {:latency_threshold, 1000},  # Alert if >1s
  {:cost_threshold, 0.01},     # Alert if cost >$0.01
  {:custom, fn _, m, _ -> m[:consensus] < 0.7 end}  # Low consensus
])
```

### Real-Time Visualization

```elixir
# Phoenix LiveView component
defmodule MyAppWeb.ExperimentLive do
  use Phoenix.LiveView

  def mount(%{"id" => exp_id}, _session, socket) do
    # Subscribe to updates
    Phoenix.PubSub.subscribe(MyApp.PubSub, "experiment:#{exp_id}")

    # Load initial data
    metrics = TelemetryResearch.calculate_metrics(exp_id)

    {:ok, assign(socket,
      experiment_id: exp_id,
      metrics: metrics,
      time_series: []
    )}
  end

  def render(assigns) do
    ~H"""
    <div class="dashboard">
      <h1>Experiment: <%= @experiment_id %></h1>

      <div class="metrics-grid">
        <div class="metric">
          <h3>P99 Latency</h3>
          <p class="value"><%= Float.round(@metrics.latency.p99, 1) %>ms</p>
        </div>

        <div class="metric">
          <h3>Success Rate</h3>
          <p class="value"><%= Float.round(@metrics.reliability.success_rate * 100, 2) %>%</p>
        </div>

        <div class="metric">
          <h3>Total Cost</h3>
          <p class="value">$<%= Float.round(@metrics.cost.total_usd, 4) %></p>
        </div>
      </div>

      <div class="chart">
        <%= live_component(TimeSeriesChart, data: @time_series) %>
      </div>
    </div>
    """
  end

  def handle_info({:metrics_update, metrics}, socket) do
    # Update time series
    time_series = socket.assigns.time_series ++ [
      %{time: System.monotonic_time(:millisecond), value: metrics.avg_latency}
    ]

    # Keep last 100 points
    time_series = Enum.take(time_series, -100)

    {:noreply, assign(socket, time_series: time_series)}
  end

  def handle_info({:full_metrics, metrics}, socket) do
    {:noreply, assign(socket, :metrics, metrics)}
  end
end
```

---

## Experiment Lifecycle

### Creating Experiments

```elixir
# Basic experiment
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "baseline_gpt4",
  hypothesis: "GPT-4 achieves >90% accuracy on MMLU",
  condition: "baseline",
  tags: ["accuracy", "mmlu"]
)

# Detailed configuration
{:ok, experiment} = TelemetryResearch.start_experiment(
  name: "ensemble_study",
  hypothesis: "5-model ensemble improves reliability by 10x",
  condition: "treatment",
  tags: ["ensemble", "reliability", "h1"],
  sample_size: 1000,
  metadata: %{
    researcher: "John Doe",
    grant_id: "NSF-12345",
    protocol_version: "v2.1"
  },
  metrics_config: %{
    latency: true,
    cost: true,
    tokens: true,
    success_rate: true,
    custom: [:consensus, :model_agreement]
  },
  storage_backend: :ets
)
```

### Running Experiments

```elixir
defmodule ExperimentRunner do
  def run_ab_test do
    # Control group
    {:ok, control} = TelemetryResearch.start_experiment(
      name: "control_single_model",
      condition: "control",
      tags: ["ab_test", "control"]
    )

    # Run control trials
    Enum.each(1..500, fn _ ->
      single_model_prediction()
    end)

    {:ok, _} = TelemetryResearch.stop_experiment(control.id)

    # Treatment group
    {:ok, treatment} = TelemetryResearch.start_experiment(
      name: "treatment_ensemble",
      condition: "treatment",
      tags: ["ab_test", "treatment"]
    )

    # Run treatment trials
    Enum.each(1..500, fn _ ->
      ensemble_prediction()
    end)

    {:ok, _} = TelemetryResearch.stop_experiment(treatment.id)

    # Compare
    ComparativeAnalysis.compare_experiments(control.id, treatment.id)
  end
end
```

### Archiving & Cleanup

```elixir
# Archive experiment for long-term storage
{:ok, archive_path} = TelemetryResearch.Experiment.archive(
  experiment_id,
  destination: :local,
  path: "archives/exp_#{experiment_id}.jsonl"
)

# Archive to S3 (when implemented)
{:ok, s3_url} = TelemetryResearch.Experiment.archive(
  experiment_id,
  destination: :s3,
  bucket: "research-archives",
  prefix: "experiments/2025/"
)

# Cleanup experiment data
TelemetryResearch.Experiment.cleanup(
  experiment_id,
  keep_data: false  # Delete all data
)

# Or keep data but remove from active list
TelemetryResearch.Experiment.cleanup(
  experiment_id,
  keep_data: true
)
```

---

## Integration Patterns

### With Ensemble Library

```elixir
defmodule EnsembleExperiment do
  def run_experiment(queries) do
    {:ok, experiment} = TelemetryResearch.start_experiment(
      name: "ensemble_evaluation",
      hypothesis: "Majority voting achieves >95% accuracy"
    )

    results = Enum.map(queries, fn query ->
      # Ensemble automatically emits telemetry
      {:ok, result} = Ensemble.predict(query,
        models: [:gpt4, :claude, :gemini],
        strategy: :majority
      )

      result
    end)

    {:ok, _} = TelemetryResearch.stop_experiment(experiment.id)

    # Analyze
    metrics = TelemetryResearch.calculate_metrics(experiment.id)
    {:ok, path} = TelemetryResearch.export(experiment.id, :csv)

    %{
      experiment_id: experiment.id,
      metrics: metrics,
      export_path: path
    }
  end
end
```

### With Hedging Library

```elixir
defmodule HedgingExperiment do
  def compare_strategies do
    strategies = [:fixed, :percentile, :adaptive]

    results = Enum.map(strategies, fn strategy ->
      {:ok, exp} = TelemetryResearch.start_experiment(
        name: "hedging_#{strategy}",
        condition: to_string(strategy),
        tags: ["hedging", "latency"]
      )

      # Run trials
      Enum.each(1..1000, fn _ ->
        Hedging.request(
          fn -> slow_api_call() end,
          strategy: strategy
        )
      end)

      {:ok, _} = TelemetryResearch.stop_experiment(exp.id)

      {strategy, TelemetryResearch.calculate_metrics(exp.id)}
    end)

    # Find best strategy
    best = Enum.min_by(results, fn {_, metrics} ->
      metrics.latency.p99
    end)

    best
  end
end
```

### With Statistical Testing

```elixir
defmodule StatisticalExperiment do
  def run_with_stats(n_trials \\ 100) do
    # Control
    {:ok, control_exp} = TelemetryResearch.start_experiment(
      name: "control",
      condition: "control"
    )

    control_accuracies = measure_accuracy(:baseline, n_trials)
    {:ok, _} = TelemetryResearch.stop_experiment(control_exp.id)

    # Treatment
    {:ok, treatment_exp} = TelemetryResearch.start_experiment(
      name: "treatment",
      condition: "treatment"
    )

    treatment_accuracies = measure_accuracy(:improved, n_trials)
    {:ok, _} = TelemetryResearch.stop_experiment(treatment_exp.id)

    # Statistical test
    test_result = Bench.compare(control_accuracies, treatment_accuracies)
    effect_size = Bench.effect_size(control_accuracies, treatment_accuracies)

    # Export both experiments
    {:ok, control_path} = TelemetryResearch.export(control_exp.id, :csv)
    {:ok, treatment_path} = TelemetryResearch.export(treatment_exp.id, :csv)

    %{
      significant: test_result.p_value < 0.05,
      p_value: test_result.p_value,
      effect_size: effect_size.cohens_d,
      control_export: control_path,
      treatment_export: treatment_path
    }
  end
end
```

---

## Advanced Topics

### Multi-Experiment Studies

```elixir
defmodule MultiExperimentStudy do
  def factorial_design do
    # 2x2 factorial design
    conditions = [
      {strategy: :majority, execution: :parallel},
      {strategy: :majority, execution: :sequential},
      {strategy: :weighted, execution: :parallel},
      {strategy: :weighted, execution: :sequential}
    ]

    results = Enum.map(conditions, fn config ->
      {:ok, exp} = TelemetryResearch.start_experiment(
        name: "factorial_#{config[:strategy]}_#{config[:execution]}",
        condition: "#{config[:strategy]}_#{config[:execution]}",
        tags: ["factorial", to_string(config[:strategy]), to_string(config[:execution])]
      )

      # Run trials
      run_trials(config)

      {:ok, _} = TelemetryResearch.stop_experiment(exp.id)

      {config, exp.id}
    end)

    # Analyze with ANOVA
    analyze_factorial(results)
  end

  defp analyze_factorial(results) do
    # Extract metrics
    data = Enum.map(results, fn {config, exp_id} ->
      metrics = TelemetryResearch.calculate_metrics(exp_id)
      %{
        strategy: config[:strategy],
        execution: config[:execution],
        p99_latency: metrics.latency.p99,
        cost: metrics.cost.total_usd,
        accuracy: metrics.custom[:accuracy]
      }
    end)

    # Group by factors
    by_strategy = Enum.group_by(data, & &1.strategy)
    by_execution = Enum.group_by(data, & &1.execution)

    # ANOVA for each DV
    latency_anova = Bench.Stats.ANOVA.one_way([
      Enum.map(by_strategy[:majority], & &1.p99_latency),
      Enum.map(by_strategy[:weighted], & &1.p99_latency)
    ])

    %{
      strategy_effect: latency_anova,
      raw_data: data
    }
  end
end
```

### Custom Storage Backends

```elixir
defmodule CustomStorageBackend do
  @behaviour TelemetryResearch.StorageBackend

  @impl true
  def init(experiment) do
    # Initialize storage (e.g., open file, connect to DB)
    {:ok, file} = File.open("#{experiment.id}.log", [:write, :append])
    {:ok, %{file: file}}
  end

  @impl true
  def store_event(event, state) do
    # Write event
    IO.puts(state.file, Jason.encode!(event))
    {:ok, state}
  end

  @impl true
  def get_all(state) do
    # Read all events
    File.stream!("#{state.experiment_id}.log")
    |> Stream.map(&Jason.decode!/1)
    |> Enum.to_list()
  end

  @impl true
  def cleanup(state) do
    File.close(state.file)
    :ok
  end
end

# Use custom backend
{:ok, exp} = TelemetryResearch.start_experiment(
  name: "custom_storage",
  storage_backend: CustomStorageBackend
)
```

### Distributed Experiments

```elixir
defmodule DistributedExperiment do
  def run_across_nodes(nodes) do
    # Start coordinating experiment
    {:ok, coordinator} = TelemetryResearch.start_experiment(
      name: "distributed_study",
      distributed: true
    )

    # Start on each node
    Enum.each(nodes, fn node ->
      :rpc.call(node, TelemetryResearch, :join_experiment, [coordinator.id])
    end)

    # Run workload on all nodes
    tasks = Enum.map(nodes, fn node ->
      Task.async(fn ->
        :rpc.call(node, __MODULE__, :run_local_workload, [])
      end)
    end)

    # Wait for completion
    Task.await_many(tasks)

    # Aggregate from all nodes
    {:ok, _} = TelemetryResearch.stop_experiment(coordinator.id)

    # Metrics now include data from all nodes
    TelemetryResearch.calculate_metrics(coordinator.id)
  end
end
```

---

## Best Practices

### 1. Always Name Experiments Clearly

```elixir
# ❌ Bad: Vague name
{:ok, exp} = TelemetryResearch.start_experiment(name: "test1")

# ✅ Good: Descriptive name
{:ok, exp} = TelemetryResearch.start_experiment(
  name: "h1_ensemble_vs_single_mmlu_accuracy",
  hypothesis: "5-model ensemble achieves >95% accuracy on MMLU",
  tags: ["hypothesis_1", "mmlu", "accuracy"]
)
```

### 2. Tag Appropriately

```elixir
# Use hierarchical tags
tags: ["study_id:abc123", "hypothesis:h1", "dataset:mmlu", "condition:treatment"]
```

### 3. Include Sufficient Metadata

```elixir
{:ok, exp} = TelemetryResearch.start_experiment(
  name: "experiment",
  metadata: %{
    researcher: "Jane Smith",
    protocol_version: "v1.2",
    random_seed: 42,
    dataset_version: "mmlu_v1.0",
    git_commit: "abc123",
    hardware: "AWS m5.2xlarge"
  }
)
```

### 4. Export Regularly

```elixir
# Export checkpoints during long experiments
defmodule CheckpointExport do
  def run_with_checkpoints(n_trials) do
    {:ok, exp} = TelemetryResearch.start_experiment(name: "long_study")

    Enum.each(1..n_trials, fn i ->
      run_trial()

      # Export every 100 trials
      if rem(i, 100) == 0 do
        TelemetryResearch.export(exp.id, :jsonl,
          path: "checkpoints/exp_#{exp.id}_#{i}.jsonl"
        )
      end
    end)

    {:ok, _} = TelemetryResearch.stop_experiment(exp.id)
  end
end
```

### 5. Validate Data Quality

```elixir
defmodule DataQuality do
  def validate_experiment(experiment_id) do
    events = TelemetryResearch.Store.get_all(experiment_id)

    %{
      total_events: length(events),
      missing_timestamps: count_missing(events, :timestamp),
      missing_measurements: count_missing(events, :measurements),
      outliers: detect_outliers(events),
      data_quality_score: calculate_quality_score(events)
    }
  end
end
```

---

## References

### Research Papers

1. **Telemetry in Distributed Systems**
   - Sambasivan, R., et al. (2016). "Principled Workflow-Centric Tracing of Distributed Systems"

2. **Experiment Management**
   - Bakshy, E., et al. (2014). "Designing and Deploying Online Field Experiments"

3. **Observability**
   - Sigelman, B., et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure"

### Tools & Standards

1. **OpenTelemetry**: https://opentelemetry.io/
2. **Elixir Telemetry**: https://hexdocs.pm/telemetry/
3. **Arrow/Parquet**: https://arrow.apache.org/

---

## Appendix: Full API Reference

### Experiment Management

```elixir
@spec start_experiment(opts :: keyword()) :: {:ok, Experiment.t()} | {:error, term()}
@spec stop_experiment(id :: String.t()) :: {:ok, Experiment.t()} | {:error, term()}
@spec get_experiment(id :: String.t()) :: {:ok, Experiment.t()} | {:error, term()}
@spec list_experiments() :: [Experiment.t()]
```

### Export Functions

```elixir
@spec export(id :: String.t(), format :: atom(), opts :: keyword()) ::
  {:ok, path :: String.t()} | {:error, term()}

# Formats: :csv, :jsonl, :parquet
# Options: :path, :columns, :transformers
```

### Metrics

```elixir
@spec calculate_metrics(id :: String.t()) :: map()

# Returns:
# %{
#   latency: %{count, mean, median, p50, p90, p95, p99, p99_9, ...},
#   cost: %{total_usd, per_request_mean, by_model, ...},
#   reliability: %{success_rate, failure_rate, mtbf, ...},
#   tokens: %{total, avg_input, avg_output, by_model, ...},
#   custom: %{...}
# }
```

---

**End of Guide**

For updates and contributions: https://github.com/your-org/elixir_ai_research

Last Updated: 2025-10-08
Version: 0.1.0
