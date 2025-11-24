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
    cns_surrogate_validation: Crucible.Stage.CNSSurrogateValidation,
    cns_tda_validation: Crucible.Stage.CNSTDAValidation,
    cns_metrics: Crucible.Stage.CNSMetrics,
    bench: Crucible.Stage.Bench,
    report: Crucible.Stage.Report
  },
  # CNS Adapters: default to no-ops; override in integration apps (e.g., cns_experiments)
  cns_adapter: Crucible.CNS.Noop,
  cns_surrogate_adapter: Crucible.CNS.SurrogateNoop,
  cns_tda_adapter: Crucible.CNS.TDANoop,
  guardrail_adapter: Crucible.Stage.Guardrails.Noop,
  tinkex_client: Crucible.Backend.Tinkex.LiveClient

config :logger, :console,
  format: "[$level] $message\n",
  metadata: [:module]

import_config "#{config_env()}.exs"
