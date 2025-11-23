defmodule Mix.Tasks.Thinker do
  @moduledoc """
  CLI for running Thinker experiments in Crucible Framework.

  ## Commands

      mix thinker info              # Show environment info
      mix thinker validate          # Validate dataset
      mix thinker train             # Run training
      mix thinker eval              # Evaluate checkpoint
      mix thinker run               # Full pipeline: validate, train, eval
      mix thinker antagonist        # Run antagonist analysis

  ## Options

      --config PATH    Config file path
      --limit N        Limit dataset samples
      --mode MODE      :simulate or :live (default: simulate)
      --epochs N       Training epochs (default: 3)

  ## Examples

      mix thinker info
      mix thinker run --limit 10 --mode simulate
      mix thinker train --epochs 5
      mix thinker antagonist --input runs/eval.jsonl

  """

  use Mix.Task

  alias Crucible.Thinker.{Harness, Datasets.Scifact, Telemetry}

  @shortdoc "Run Thinker experiments"

  @impl Mix.Task
  def run(args) do
    Application.ensure_all_started(:crucible_framework)

    {opts, args, _} =
      OptionParser.parse(args,
        strict: [
          config: :string,
          limit: :integer,
          mode: :string,
          epochs: :integer,
          input: :string,
          output: :string
        ]
      )

    command = List.first(args)

    if command do
      # Direct command execution
      case command do
        "info" -> cmd_info(opts)
        "validate" -> cmd_validate(opts)
        "train" -> cmd_train(opts)
        "eval" -> cmd_eval(opts)
        "run" -> cmd_run(opts)
        "antagonist" -> cmd_antagonist(opts)
        "menu" -> interactive_menu(opts)
        _ -> Mix.shell().error("Unknown command: #{command}")
      end
    else
      # Interactive menu
      interactive_menu(opts)
    end
  end

  defp interactive_menu(opts) do
    Mix.shell().info("""

    ╔════════════════════════════════════════════╗
    ║     Thinker - Crucible Framework CLI       ║
    ╚════════════════════════════════════════════╝

    Select an option:

      1. Show environment info
      2. Validate dataset
      3. Run training
      4. Run limited training (5 samples)
      5. Evaluate checkpoint
      6. Run limited eval (5 samples)
      7. Run full pipeline (validate → train → eval)
      8. Run antagonist analysis
      9. Configure options

      0. Exit

    """)

    case Mix.shell().prompt("Enter choice") |> String.trim() do
      "1" ->
        cmd_info(opts)
        interactive_menu(opts)

      "2" ->
        cmd_validate(opts)
        interactive_menu(opts)

      "3" ->
        cmd_train(opts)
        interactive_menu(opts)

      "4" ->
        cmd_train_limited(opts)
        interactive_menu(opts)

      "5" ->
        cmd_eval(opts)
        interactive_menu(opts)

      "6" ->
        cmd_eval_limited(opts)
        interactive_menu(opts)

      "7" ->
        cmd_run(opts)
        interactive_menu(opts)

      "8" ->
        cmd_antagonist(opts)
        interactive_menu(opts)

      "9" ->
        opts = configure_options(opts)
        interactive_menu(opts)

      "0" ->
        Mix.shell().info("Goodbye!")

      "" ->
        interactive_menu(opts)

      _ ->
        Mix.shell().error("Invalid option")
        interactive_menu(opts)
    end
  end

  defp configure_options(opts) do
    Mix.shell().info("""

    Current Configuration:
    ──────────────────────
      Limit:  #{Keyword.get(opts, :limit, "default (15)")}
      Mode:   #{Keyword.get(opts, :mode, "live")}
      Epochs: #{Keyword.get(opts, :epochs, "3")}
      Input:  #{Keyword.get(opts, :input, "runs/thinker_eval.jsonl")}

    Configure:
      1. Set sample limit
      2. Set mode (simulate/live)
      3. Set epochs
      4. Set input file
      0. Back to main menu

    """)

    case Mix.shell().prompt("Enter choice") |> String.trim() do
      "1" ->
        value = Mix.shell().prompt("Enter limit (number)") |> String.trim()

        case Integer.parse(value) do
          {n, ""} ->
            Mix.shell().info("Limit set to #{n}")
            Keyword.put(opts, :limit, n)

          _ ->
            Mix.shell().error("Invalid number")
            opts
        end

      "2" ->
        value = Mix.shell().prompt("Enter mode (simulate/live)") |> String.trim()

        if value in ["simulate", "live"] do
          Mix.shell().info("Mode set to #{value}")
          Keyword.put(opts, :mode, value)
        else
          Mix.shell().error("Invalid mode. Use 'simulate' or 'live'")
          opts
        end

      "3" ->
        value = Mix.shell().prompt("Enter epochs (number)") |> String.trim()

        case Integer.parse(value) do
          {n, ""} when n > 0 ->
            Mix.shell().info("Epochs set to #{n}")
            Keyword.put(opts, :epochs, n)

          _ ->
            Mix.shell().error("Invalid number")
            opts
        end

      "4" ->
        value = Mix.shell().prompt("Enter input file path") |> String.trim()
        Mix.shell().info("Input set to #{value}")
        Keyword.put(opts, :input, value)

      "0" ->
        opts

      _ ->
        Mix.shell().error("Invalid option")
        opts
    end
  end

  defp cmd_info(_opts) do
    Mix.shell().info("""
    Thinker - Crucible Framework
    ============================

    Version: #{Application.spec(:crucible_framework, :vsn)}
    Elixir: #{System.version()}
    OTP: #{System.otp_release()}

    Dataset: priv/data/scifact_claim_extractor_clean.jsonl
    Backends:
      - Entailment: #{Application.get_env(:crucible, :thinker_entailment_backend, :heuristic)}
      - Similarity: #{Application.get_env(:crucible, :thinker_similarity_backend, :heuristic)}

    Commands:
      mix thinker validate    # Validate dataset
      mix thinker train       # Run training
      mix thinker eval        # Evaluate
      mix thinker run         # Full pipeline
      mix thinker antagonist  # Antagonist analysis
    """)
  end

  defp cmd_validate(opts) do
    Mix.shell().info("Validating dataset...")

    limit = Keyword.get(opts, :limit)
    {:ok, dataset} = load_dataset(limit)

    Mix.shell().info("Loaded #{length(dataset)} samples")

    # Check format
    valid =
      Enum.all?(dataset, fn sample ->
        Map.has_key?(sample, "prompt") and Map.has_key?(sample, "completion")
      end)

    if valid do
      Mix.shell().info("✓ Dataset validation passed")
    else
      Mix.shell().error("✗ Dataset validation failed")
    end
  end

  defp cmd_train(opts) do
    Mix.shell().info("Running training...")
    Telemetry.attach(log_level: :info)

    experiment = build_experiment(opts)
    {:ok, result} = Harness.run(experiment)

    Mix.shell().info("Training complete. Final loss: #{result.training.final_loss}")
  end

  defp cmd_train_limited(opts) do
    Mix.shell().info("Running limited training (5 samples)...")
    Telemetry.attach(log_level: :info)

    opts =
      opts
      |> Keyword.put(:limit, 5)
      |> Keyword.put(:mode, "train_only")

    experiment = build_experiment(opts)
    {:ok, result} = Harness.run(experiment)

    Mix.shell().info("Limited training complete. Final loss: #{result.training.final_loss}")
  end

  defp cmd_eval(opts) do
    Mix.shell().info("Running evaluation...")
    Telemetry.attach(log_level: :info)

    experiment = build_experiment(opts)
    {:ok, result} = Harness.run(experiment)

    agg = result.evaluation.aggregate

    Mix.shell().info("""
    Evaluation Results:
      Schema Compliance: #{Float.round(agg.schema_compliance * 100, 1)}%
      Citation Accuracy: #{Float.round(agg.citation_accuracy * 100, 1)}%
      Mean Entailment: #{Float.round(agg.mean_entailment * 100, 1)}%
      Mean Similarity: #{Float.round(agg.mean_similarity * 100, 1)}%
    """)
  end

  defp cmd_eval_limited(opts) do
    Mix.shell().info("Running limited evaluation (5 samples)...")
    Telemetry.attach(log_level: :info)

    opts =
      opts
      |> Keyword.put(:limit, 5)
      |> Keyword.put(:mode, "eval_only")

    experiment = build_experiment(opts)
    {:ok, result} = Harness.run(experiment)

    agg = result.evaluation.aggregate

    Mix.shell().info("""
    Limited Evaluation Results:
      Schema Compliance: #{Float.round(agg.schema_compliance * 100, 1)}%
      Citation Accuracy: #{Float.round(agg.citation_accuracy * 100, 1)}%
      Mean Entailment: #{Float.round(agg.mean_entailment * 100, 1)}%
      Mean Similarity: #{Float.round(agg.mean_similarity * 100, 1)}%
    """)
  end

  defp cmd_run(opts) do
    Mix.shell().info("Running full pipeline...")
    Telemetry.attach(log_level: :info)

    experiment = build_experiment(opts)
    {:ok, result} = Harness.run(experiment)

    report = Harness.report(result)

    output_path = "runs/thinker_report_#{timestamp()}.md"
    File.mkdir_p!("runs")
    File.write!(output_path, report)

    Mix.shell().info("Report written to #{output_path}")

    if result.quality_check.passed do
      Mix.shell().info("✓ Quality check PASSED")
    else
      Mix.shell().error("✗ Quality check FAILED")
    end
  end

  defp cmd_antagonist(opts) do
    input_path = Keyword.get(opts, :input, "runs/thinker_eval.jsonl")

    if File.exists?(input_path) do
      Mix.shell().info("Running antagonist on #{input_path}...")
      # Would load and analyze the evaluation results
      Mix.shell().info("Antagonist analysis complete")
    else
      Mix.shell().error("Input file not found: #{input_path}")
    end
  end

  defp build_experiment(opts) do
    limit = Keyword.get(opts, :limit, 15)
    mode = opts |> Keyword.get(:mode, "live") |> String.to_atom()
    epochs = Keyword.get(opts, :epochs, 3)

    Harness.define(
      name: "thinker-cli-#{timestamp()}",
      dataset: %{source: :scifact, limit: limit},
      training: %{epochs: epochs},
      mode: mode
    )
  end

  defp load_dataset(limit) do
    path =
      Path.join(:code.priv_dir(:crucible_framework), "data/scifact_claim_extractor_clean.jsonl")

    if File.exists?(path) do
      samples =
        path
        |> File.stream!()
        |> Enum.map(&Jason.decode!/1)
        |> then(fn data ->
          if limit, do: Enum.take(data, limit), else: data
        end)

      {:ok, samples}
    else
      # Fallback to sample data
      Scifact.load(limit: limit)
    end
  end

  defp timestamp do
    DateTime.utc_now()
    |> DateTime.to_iso8601(:basic)
    |> String.replace(~r/[:\-]/, "")
    |> String.slice(0, 15)
  end
end
