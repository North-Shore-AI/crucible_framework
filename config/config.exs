import Config

config :crucible_framework,
  ecto_repos: [CrucibleFramework.Repo],
  enable_repo: true,
  backends: %{
    tinkex: Crucible.Backend.Tinkex
  },
  stage_registry: %{
    data_load: Crucible.Stage.DataLoad,
    data_checks: Crucible.Stage.DataChecks,
    guardrails: Crucible.Stage.Guardrails,
    backend_call: Crucible.Stage.BackendCall,
    analysis_surrogate_validation: Crucible.Stage.Analysis.SurrogateValidation,
    analysis_tda_validation: Crucible.Stage.Analysis.TDAValidation,
    analysis_metrics: Crucible.Stage.Analysis.Metrics,
    analysis_filter: Crucible.Stage.Analysis.Filter,
    fairness: Crucible.Stage.Fairness,
    bench: Crucible.Stage.Bench,
    report: Crucible.Stage.Report
  },
  # Analysis adapters: default to no-ops; override in integration apps (e.g., cns_crucible)
  analysis_adapter: Crucible.Analysis.Noop,
  analysis_surrogate_adapter: Crucible.Analysis.SurrogateNoop,
  analysis_tda_adapter: Crucible.Analysis.TDANoop,
  # Fairness Adapter: default to noop; override with ExFairnessAdapter when ExFairness is available
  fairness_adapter: Crucible.Fairness.Noop,
  guardrail_adapter: Crucible.Stage.Guardrails.Noop,
  tinkex_client: Crucible.Backend.Tinkex.LiveClient

config :logger, :console,
  format: "[$level] $message\n",
  metadata: [:module]

import_config "#{config_env()}.exs"
