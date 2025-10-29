#!/usr/bin/env elixir

# Experiment Definition Example
# Shows how to define and configure experiments

alias CrucibleFramework.Experiment

IO.puts("\n=== Crucible Framework: Experiment Definition Example ===\n")

# Example 1: Create experiment with new/1
IO.puts("1. Creating Experiment with new/1")
IO.puts("=" |> String.duplicate(50))

{:ok, config} =
  Experiment.new(
    name: "Ensemble Reliability Study",
    description: "Testing if 5-model ensemble achieves >99% accuracy",
    conditions: ["baseline_gpt4", "ensemble_5_models"],
    metrics: [:accuracy, :latency_p99, :cost_per_query, :consensus],
    repeat: 3,
    seed: 42
  )

IO.puts("Experiment Configuration:")
IO.puts("  Name:         #{config.name}")
IO.puts("  Description:  #{config.description}")
IO.puts("  Conditions:   #{inspect(config.conditions)}")
IO.puts("  Metrics:      #{inspect(config.metrics)}")
IO.puts("  Repetitions:  #{config.repeat}")
IO.puts("  Seed:         #{config.seed}")

# Example 2: Validate experiment configuration
IO.puts("\n2. Validating Experiment Configuration")
IO.puts("=" |> String.duplicate(50))

case Experiment.validate(config) do
  {:ok, _} ->
    IO.puts("✓ Configuration is valid")

  {:error, reason} ->
    IO.puts("✗ Configuration error: #{reason}")
end

# Example 3: Invalid experiment
IO.puts("\n3. Testing Invalid Configuration")
IO.puts("=" |> String.duplicate(50))

invalid_config = %{
  name: "Incomplete Experiment"
  # Missing conditions and metrics
}

case Experiment.validate(invalid_config) do
  {:ok, _} ->
    IO.puts("✓ Configuration is valid")

  {:error, reason} ->
    IO.puts("✗ Expected error: #{reason}")
end

# Example 4: Using the Experiment behaviour
IO.puts("\n4. Defining Experiment Module")
IO.puts("=" |> String.duplicate(50))

defmodule HedgingLatencyExperiment do
  use CrucibleFramework.Experiment

  def name, do: "Request Hedging P99 Latency Reduction"

  def description do
    "Hypothesis 2: Request hedging reduces P99 latency by ≥50% with <15% cost increase"
  end

  def conditions, do: ["baseline_no_hedging", "hedging_p95_strategy"]
  def metrics, do: [:latency_p50, :latency_p95, :latency_p99, :cost_per_query, :hedge_fire_rate]

  def run do
    {:ok,
     %{
       experiment: config(),
       status: "Configured and ready to run",
       note: "Use ResearchHarness library for actual execution"
     }}
  end
end

hedging_config = HedgingLatencyExperiment.config()
IO.puts("Experiment: #{hedging_config.name}")
IO.puts("Description: #{hedging_config.description}")
IO.puts("Conditions: #{inspect(hedging_config.conditions)}")
IO.puts("Metrics: #{inspect(hedging_config.metrics)}")
IO.puts("Status: #{hedging_config.status}")

# Run the experiment (just shows config in this example)
{:ok, result} = HedgingLatencyExperiment.run()
IO.puts("\nRun Result:")
IO.puts("  Status: #{result.status}")
IO.puts("  Note: #{result.note}")

# Example 5: Multiple experiment types
IO.puts("\n5. Different Experiment Types")
IO.puts("=" |> String.duplicate(50))

experiments = [
  Experiment.new(
    name: "Cost-Accuracy Trade-off",
    conditions: ["cheap_models", "expensive_models"],
    metrics: [:accuracy, :cost_usd],
    repeat: 5
  ),
  Experiment.new(
    name: "Ensemble Size Study",
    conditions: ["ensemble_3", "ensemble_5", "ensemble_7"],
    metrics: [:accuracy, :consensus, :latency],
    repeat: 3
  ),
  Experiment.new(
    name: "Prompt Strategy Comparison",
    conditions: ["zero_shot", "few_shot", "chain_of_thought"],
    metrics: [:accuracy, :reasoning_quality],
    repeat: 3
  )
]

IO.puts("Created #{length(experiments)} experiment configurations:")

Enum.each(experiments, fn {:ok, exp} ->
  IO.puts("  - #{exp.name}")
  IO.puts("    Conditions: #{length(exp.conditions)}, Metrics: #{length(exp.metrics)}")
end)

IO.puts("\n=== Example Complete ===\n")
