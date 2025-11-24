#!/usr/bin/env elixir

# CNS SciFact Claim Extraction Example
#
# This example demonstrates the full CNS + Crucible + Tinkex integration
# using the new Crucible IR framework.
#
# Run with: mix run examples/cns_scifact.exs
#
# Options:
#   --limit N        Process only N examples (default: 100)
#   --batch-size N   Training batch size (default: 4)
#   --model MODEL    Base model (default: meta-llama/Llama-3.2-1B)
#   --rank N         LoRA rank (default: 8)
#   --no-train       Skip training, only validate
#   --verbose        Enable verbose logging

# Parse command-line arguments
args = System.argv()

opts = [
  limit:
    args
    |> Enum.find_value(100, fn
      "--limit" -> true
      _ -> false
    end)
    |> then(fn
      true ->
        idx = Enum.find_index(args, &(&1 == "--limit"))

        if idx && idx + 1 < length(args) do
          args |> Enum.at(idx + 1) |> String.to_integer()
        else
          100
        end

      _ ->
        100
    end),
  batch_size:
    args
    |> Enum.find_value(4, fn
      "--batch-size" -> true
      _ -> false
    end)
    |> then(fn
      true ->
        idx = Enum.find_index(args, &(&1 == "--batch-size"))

        if idx && idx + 1 < length(args) do
          args |> Enum.at(idx + 1) |> String.to_integer()
        else
          4
        end

      _ ->
        4
    end),
  base_model:
    args
    |> Enum.find_value("meta-llama/Llama-3.2-1B", fn
      "--model" -> true
      _ -> false
    end)
    |> then(fn
      true ->
        idx = Enum.find_index(args, &(&1 == "--model"))

        if idx && idx + 1 < length(args) do
          Enum.at(args, idx + 1)
        else
          "meta-llama/Llama-3.2-1B"
        end

      _ ->
        "meta-llama/Llama-3.2-1B"
    end),
  lora_rank:
    args
    |> Enum.find_value(8, fn
      "--rank" -> true
      _ -> false
    end)
    |> then(fn
      true ->
        idx = Enum.find_index(args, &(&1 == "--rank"))

        if idx && idx + 1 < length(args) do
          args |> Enum.at(idx + 1) |> String.to_integer()
        else
          8
        end

      _ ->
        8
    end),
  skip_training: "--no-train" in args,
  verbose: "--verbose" in args
]

IO.puts("""
================================================================================
CNS SciFact Claim Extraction Experiment
================================================================================

This example demonstrates:
1. Loading SciFact claim extraction dataset
2. Building Crucible Experiment IR
3. Training via Tinkex LoRA backend
4. Evaluating with CNS metrics
5. Generating comprehensive reports

Configuration:
- Limit: #{opts[:limit]} examples
- Batch size: #{opts[:batch_size]}
- Base model: #{opts[:base_model]}
- LoRA rank: #{opts[:lora_rank]}
- Training: #{if opts[:skip_training], do: "DISABLED", else: "ENABLED"}
- Verbose: #{if opts[:verbose], do: "ON", else: "OFF"}

================================================================================
""")

# Configure logging
if opts[:verbose] do
  Logger.configure(level: :debug)
else
  Logger.configure(level: :info)
end

# Check if we're in cns_experiments or crucible_framework
cns_exp_available = Code.ensure_loaded?(CnsExperiments.Experiments.ScifactClaimExtraction)

if cns_exp_available do
  IO.puts("Using CnsExperiments.Experiments.ScifactClaimExtraction\n")

  # Run the experiment
  case CnsExperiments.Experiments.ScifactClaimExtraction.run(opts) do
    {:ok, context} ->
      IO.puts("\n✅ Experiment completed successfully!\n")

      # Print key metrics
      if context[:metrics] do
        IO.puts("Key Metrics:")
        IO.puts("------------")

        if context.metrics[:cns] do
          cns = context.metrics.cns
          IO.puts("CNS Metrics:")
          IO.puts("  Schema compliance: #{Float.round(cns[:schema_compliance] || 0.0, 3)}")
          IO.puts("  Citation accuracy: #{Float.round(cns[:citation_accuracy] || 0.0, 3)}")
          IO.puts("  Topology score: #{Float.round(cns[:topology_score] || 0.0, 3)}")
          IO.puts("  Chirality score: #{Float.round(cns[:chirality_score] || 0.0, 3)}")
        end

        if context.metrics[:training] do
          training = context.metrics.training
          IO.puts("\nTraining Metrics:")
          IO.puts("  Final loss: #{Float.round(training[:final_loss] || 0.0, 4)}")
          IO.puts("  Total steps: #{training[:total_steps] || 0}")
        end

        if context.metrics[:bench] do
          bench = context.metrics.bench
          IO.puts("\nStatistical Analysis:")
          IO.puts("  Bootstrap CI: #{inspect(bench[:bootstrap_ci])}")
          IO.puts("  Effect size: #{Float.round(bench[:effect_size] || 0.0, 3)}")
        end
      end

      # Report output locations
      if context[:outputs] do
        IO.puts("\nOutputs Generated:")
        IO.puts("-----------------")

        Enum.each(context.outputs, fn {name, info} ->
          IO.puts("  #{name}: #{info[:path] || info[:location] || "in memory"}")
        end)
      end

    {:error, reason} ->
      IO.puts("\n❌ Experiment failed: #{inspect(reason)}\n")
      System.halt(1)
  end
else
  # Fallback: demonstrate the experiment structure without running
  IO.puts("CnsExperiments module not available. Showing experiment structure:\n")

  # Build and display the experiment
  alias Crucible.IR.{
    Experiment,
    DatasetRef,
    BackendRef,
    ReliabilityConfig,
    EnsembleConfig,
    HedgingConfig,
    GuardrailConfig,
    StatsConfig,
    FairnessConfig,
    OutputSpec,
    StageDef
  }

  experiment = %Experiment{
    id: "cns_scifact_demo_#{System.unique_integer([:positive]) |> rem(10000)}",
    description: "CNS claim extraction on SciFact via Tinkex LoRA backend",
    owner: "demo",
    tags: ["cns", "scifact", "tinkex", "lora"],
    dataset: %DatasetRef{
      provider: nil,
      name: "scifact_claim_extractor",
      split: :train,
      options: %{
        path: "priv/data/scifact_claim_extractor_clean.jsonl",
        batch_size: opts[:batch_size],
        limit: opts[:limit],
        input_key: :prompt,
        output_key: :completion
      }
    },
    pipeline: [
      %StageDef{name: :data_load},
      %StageDef{name: :data_checks},
      %StageDef{name: :guardrails},
      %StageDef{name: :backend_call},
      %StageDef{name: :cns_metrics},
      %StageDef{name: :bench},
      %StageDef{name: :report}
    ],
    backend: %BackendRef{
      id: :tinkex,
      profile: :lora_finetune,
      options: %{
        base_model: opts[:base_model],
        lora_rank: opts[:lora_rank],
        lora_alpha: opts[:lora_rank] * 2,
        learning_rate: 1.0e-4,
        warmup_steps: 100
      }
    },
    reliability: %ReliabilityConfig{
      ensemble: %EnsembleConfig{strategy: :none},
      hedging: %HedgingConfig{strategy: :off},
      guardrails: %GuardrailConfig{profiles: [:default]},
      stats: %StatsConfig{tests: [:bootstrap]},
      fairness: %FairnessConfig{enabled: false}
    },
    outputs: [
      %OutputSpec{
        name: :metrics_report,
        formats: [:markdown, :json],
        sink: :file
      },
      %OutputSpec{
        name: :checkpoint,
        formats: [],
        sink: :file
      }
    ]
  }

  IO.puts("Experiment Structure:")
  IO.puts("--------------------")
  IO.puts("ID: #{experiment.id}")
  IO.puts("Description: #{experiment.description}")
  IO.puts("\nDataset:")
  IO.puts("  Name: #{experiment.dataset.name}")
  IO.puts("  Limit: #{experiment.dataset.options.limit}")
  IO.puts("  Batch size: #{experiment.dataset.options.batch_size}")
  IO.puts("\nPipeline Stages:")

  Enum.each(experiment.pipeline, fn stage ->
    IO.puts("  - #{stage.name}")
  end)

  IO.puts("\nBackend:")
  IO.puts("  Type: #{experiment.backend.id}")
  IO.puts("  Profile: #{experiment.backend.profile}")
  IO.puts("  Model: #{experiment.backend.options.base_model}")
  IO.puts("  LoRA rank: #{experiment.backend.options.lora_rank}")
  IO.puts("\nOutputs:")

  Enum.each(experiment.outputs, fn output ->
    IO.puts("  - #{output.name} (#{output.sink})")
  end)

  IO.puts("\nTo run this experiment:")
  IO.puts("1. Ensure cns_experiments is compiled: cd cns_experiments && mix compile")
  IO.puts("2. Set TINKER_API_KEY environment variable")
  IO.puts("3. Run: mix run examples/cns_scifact.exs")
end

IO.puts("\n================================================================================")
IO.puts("Example Complete")
IO.puts("================================================================================\n")
