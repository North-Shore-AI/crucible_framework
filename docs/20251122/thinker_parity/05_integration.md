# Integration with Crucible Ecosystem

## Overview

How thinker parity modules integrate with existing crucible libraries.

## crucible_harness Integration

### Experiment DSL

```elixir
defmodule Crucible.Thinker.Experiments.ClaimExtraction do
  use Crucible.Harness.Experiment

  experiment "claim-extraction-scifact" do
    description "Train claim extraction model on SciFact dataset"

    # Dataset configuration
    dataset do
      source Crucible.Thinker.Datasets.SciFact
      split :train
      limit 15
      validation_expectations [
        {:expect_column_to_exist, :claim},
        {:expect_column_values_to_not_be_null, :claim}
      ]
    end

    # Training configuration
    training do
      model "meta-llama/Llama-3.1-8B-Instruct"
      adapter :lora, rank: 16, alpha: 32

      hyperparameters do
        learning_rate 2.0e-4
        epochs 3
        batch_size 8
        warmup_steps 100
      end

      loss_config do
        citation_validity_weight 5.0
      end
    end

    # Evaluation configuration
    evaluation do
      metrics [:schema_compliance, :citation_accuracy, :mean_entailment, :mean_similarity]

      thresholds do
        schema_compliance 0.95
        citation_accuracy 0.95
        mean_entailment 0.50
      end

      antagonist_analysis true
    end

    # Reporting
    output do
      format [:markdown, :json]
      include_telemetry true
      include_statistics true
    end
  end
end
```

### Running Experiments

```elixir
alias Crucible.Harness.Runner

# Run single experiment
{:ok, result} = Runner.run(Crucible.Thinker.Experiments.ClaimExtraction)

# Run with variations
{:ok, results} = Runner.run_grid(
  Crucible.Thinker.Experiments.ClaimExtraction,
  variations: [
    lora_rank: [8, 16, 32],
    learning_rate: [1.0e-4, 2.0e-4, 5.0e-4]
  ]
)
```

## crucible_telemetry Integration

### Event Schema

```elixir
defmodule Crucible.Thinker.Telemetry.Events do
  @events [
    # Dataset events
    [:crucible, :thinker, :dataset, :load],
    [:crucible, :thinker, :dataset, :validate],

    # Training events
    [:crucible, :thinker, :training, :start],
    [:crucible, :thinker, :training, :progress],
    [:crucible, :thinker, :training, :complete],

    # Validation events
    [:crucible, :thinker, :validation, :schema],
    [:crucible, :thinker, :validation, :citation],
    [:crucible, :thinker, :validation, :entailment],
    [:crucible, :thinker, :validation, :similarity],
    [:crucible, :thinker, :validation, :complete],

    # Antagonist events
    [:crucible, :thinker, :antagonist, :complete]
  ]

  def events, do: @events
end
```

### Telemetry Handler

```elixir
defmodule Crucible.Thinker.Telemetry.Handler do
  alias Crucible.Telemetry.Research

  def attach do
    :telemetry.attach_many(
      "crucible-thinker-handler",
      Crucible.Thinker.Telemetry.Events.events(),
      &handle_event/4,
      %{experiment_id: nil}
    )
  end

  def handle_event(event, measurements, metadata, config) do
    # Store in crucible_telemetry research store
    Research.capture(%{
      event: event,
      measurements: measurements,
      metadata: metadata,
      timestamp: DateTime.utc_now(),
      experiment_id: config.experiment_id
    })
  end
end
```

### Querying Telemetry Data

```elixir
alias Crucible.Telemetry.Research

# Get all training progress for experiment
{:ok, events} = Research.query(
  event: [:crucible, :thinker, :training, :progress],
  experiment_id: "exp-123"
)

# Aggregate validation scores
{:ok, scores} = Research.aggregate(
  event: [:crucible, :thinker, :validation, :complete],
  experiment_id: "exp-123",
  aggregations: [:mean, :std, :min, :max]
)
```

## crucible_bench Integration

### Statistical Analysis

```elixir
defmodule Crucible.Thinker.Analysis do
  alias Crucible.Bench

  def analyze_experiment(experiment_id) do
    # Fetch validation scores from telemetry
    {:ok, events} = Crucible.Telemetry.Research.query(
      event: [:crucible, :thinker, :validation, :complete],
      experiment_id: experiment_id
    )

    entailment_scores = Enum.map(events, & &1.measurements.mean_entailment)
    similarity_scores = Enum.map(events, & &1.measurements.mean_similarity)

    %{
      entailment: %{
        descriptive: Bench.describe(entailment_scores),
        normality: Bench.test_normality(entailment_scores),
        confidence_interval: Bench.confidence_interval(entailment_scores, 0.95)
      },
      similarity: %{
        descriptive: Bench.describe(similarity_scores),
        normality: Bench.test_normality(similarity_scores),
        confidence_interval: Bench.confidence_interval(similarity_scores, 0.95)
      },
      correlation: Bench.correlation(entailment_scores, similarity_scores)
    }
  end

  def compare_experiments(exp_a, exp_b) do
    {:ok, scores_a} = get_entailment_scores(exp_a)
    {:ok, scores_b} = get_entailment_scores(exp_b)

    %{
      comparison: Bench.compare(scores_a, scores_b, test: :auto),
      effect_size: Bench.effect_size(scores_a, scores_b, type: :cohens_d)
    }
  end

  defp get_entailment_scores(experiment_id) do
    {:ok, events} = Crucible.Telemetry.Research.query(
      event: [:crucible, :thinker, :validation, :complete],
      experiment_id: experiment_id
    )
    {:ok, Enum.map(events, & &1.measurements.mean_entailment)}
  end
end
```

## ExDataCheck Integration

### Dataset Validation

```elixir
defmodule Crucible.Thinker.Datasets.Validation do
  alias ExDataCheck.Validator
  alias ExDataCheck.Profiler

  @expectations [
    # Schema expectations
    {:expect_column_to_exist, :id},
    {:expect_column_to_exist, :claim},
    {:expect_column_to_exist, :evidence},
    {:expect_column_to_exist, :cited_doc_ids},

    # Value expectations
    {:expect_column_values_to_not_be_null, :claim},
    {:expect_column_values_to_be_of_type, :id, :integer},
    {:expect_column_values_to_be_unique, :id},

    # Statistical expectations
    {:expect_column_value_lengths_to_be_between, :claim, 10, 1000}
  ]

  def validate(dataset) do
    result = Validator.validate(dataset, @expectations)

    if result.success do
      {:ok, dataset}
    else
      {:error, {:validation_failed, result.results}}
    end
  end

  def profile(dataset) do
    Profiler.profile(dataset, columns: [:claim, :evidence])
  end
end
```

### Training Data Quality Gates

```elixir
defmodule Crucible.Thinker.QualityGate do
  alias ExDataCheck.Validator

  def check_before_training(dataset) do
    expectations = [
      # No empty claims
      {:expect_column_values_to_not_be_null, :claim},

      # Claims have reasonable length
      {:expect_column_value_lengths_to_be_between, :claim, 10, 500},

      # Each claim has at least one citation
      {:expect_column_values_to_match_regex, :cited_doc_ids, ~r/\d+/}
    ]

    case Validator.validate(dataset, expectations) do
      %{success: true} -> :ok
      %{success: false, results: results} ->
        failed = Enum.filter(results, &(!&1.success))
        {:error, {:quality_gate_failed, failed}}
    end
  end
end
```

## ExFairness Integration

### Output Fairness Analysis

```elixir
defmodule Crucible.Thinker.Fairness do
  alias ExFairness.Metrics

  @doc """
  Analyze fairness of model outputs across document categories.
  """
  def analyze_by_category(validation_results, metadata) do
    # Group by document category (e.g., medical domain)
    grouped = Enum.group_by(validation_results, fn result ->
      get_category(result.claim.doc_id, metadata)
    end)

    # Calculate success rates per group
    rates = Enum.map(grouped, fn {category, results} ->
      successes = Enum.count(results, &(&1.entailment_score > 0.5))
      {category, successes / length(results)}
    end)

    # Check demographic parity
    %{
      rates_by_category: Map.new(rates),
      demographic_parity: Metrics.demographic_parity_difference(rates),
      disparate_impact: Metrics.disparate_impact_ratio(rates)
    }
  end

  defp get_category(doc_id, metadata) do
    Map.get(metadata.categories, doc_id, :unknown)
  end
end
```

## Full Workflow Example

```elixir
alias Crucible.Thinker.{
  Datasets.SciFact,
  Lora.Config,
  Lora.TrainingLoop,
  Validation.Pipeline,
  CNS.Antagonist,
  Analysis
}

# 1. Load and validate dataset
{:ok, dataset} = SciFact.load(split: :train, limit: 15)

# 2. Configure training
config = Config.new(
  name: "claim-extractor-v1",
  base_model: "meta-llama/Llama-3.1-8B-Instruct",
  lora_rank: 16,
  epochs: 3
)

# 3. Train model
{:ok, training_result} = TrainingLoop.run(config, dataset)

# 4. Run evaluation
{:ok, eval_dataset} = SciFact.load(split: :validation, limit: 10)
outputs = generate_outputs(training_result.model_id, eval_dataset)

# 5. Validate outputs
validation_results = Enum.map(Enum.zip(outputs, eval_dataset), fn {output, sample} ->
  Pipeline.validate(output, %{
    corpus: load_corpus(),
    evidence: sample.evidence,
    expected: sample.expected_output
  })
end)

# 6. Antagonist analysis
antagonist_reports = Enum.map(validation_results, &Antagonist.analyze/1)

# 7. Statistical analysis
stats = Analysis.analyze_experiment("claim-extractor-v1")

# 8. Generate report
report = Crucible.Reporter.generate(
  experiment: "claim-extractor-v1",
  training: training_result,
  validation: validation_results,
  antagonist: antagonist_reports,
  statistics: stats,
  format: :markdown
)

File.write!("experiment_report.md", report)
```

## Configuration

```elixir
# config/config.exs

config :crucible,
  # Tinkex API
  tinkex_url: System.get_env("TINKEX_URL", "http://localhost:8080"),

  # Telemetry storage
  telemetry_backend: :ets,  # or :postgres

  # Validation implementations
  entailment_impl: Crucible.Thinker.Validation.Entailment.Tinkex,
  similarity_impl: Crucible.Thinker.Validation.Similarity.Tinkex

# config/prod.exs - Switch to Bumblebee when ready
config :crucible,
  entailment_impl: Crucible.Thinker.Validation.Entailment.Bumblebee,
  similarity_impl: Crucible.Thinker.Validation.Similarity.Bumblebee
```
