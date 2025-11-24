alias Crucible.IR.{
  BackendRef,
  DatasetRef,
  Experiment,
  OutputSpec,
  ReliabilityConfig,
  StageDef
}

alias Crucible.IR.{GuardrailConfig, StatsConfig, FairnessConfig, EnsembleConfig, HedgingConfig}

IO.puts("""
Running live Tinkex demo.
Ensure TINKER_API_KEY is exported and PostgreSQL is reachable (see config/dev.exs).
This demo uses a tiny slice of the SciFact claim extractor dataset.
""")

experiment =
  %Experiment{
    id: "tinkex_scifact_demo",
    description: "Minimal end-to-end training loop against Tinkex using Crucible pipeline.",
    dataset: %DatasetRef{
      name: "scifact_claims",
      options: %{
        path: "priv/data/scifact_claim_extractor_clean.jsonl",
        limit: 4,
        batch_size: 2
      }
    },
    pipeline: [
      %StageDef{name: :data_load, options: %{input_key: :prompt, output_key: :completion}},
      %StageDef{name: :data_checks, options: %{required_fields: [:input, :output]}},
      %StageDef{name: :guardrails, options: %{fail_on_violation: false}},
      %StageDef{
        name: :backend_call,
        options: %{
          mode: :train,
          sample_prompts: [
            "CLAIM: AI systems are reliable with sufficient evaluation. Write a counterclaim."
          ],
          create_sampler?: true
        }
      },
      %StageDef{name: :cns_metrics},
      %StageDef{name: :bench},
      %StageDef{name: :report}
    ],
    backend: %BackendRef{
      id: :tinkex,
      profile: :lora_finetune,
      options: %{
        base_model: System.get_env("TINKER_BASE_MODEL", "meta-llama/Llama-3.2-1B"),
        train_timeout: 120_000
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
        name: :report,
        description: "Pipeline summary",
        formats: [:markdown, :json],
        sink: :stdout,
        options: %{path: "reports/tinkex_scifact_demo.md"}
      }
    ]
  }

case CrucibleFramework.run(experiment, persist: true) do
  {:ok, ctx} ->
    IO.puts("Run complete.")
    IO.inspect(ctx.metrics, label: "metrics")

  {:error, {stage, reason, _ctx}} ->
    IO.puts("Run failed at #{inspect(stage)}: #{inspect(reason)}")
end
