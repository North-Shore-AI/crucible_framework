import Config

config :crucible_framework,
  ecto_repos: [CrucibleFramework.Repo],
  enable_repo: true,
  stage_registry: %{
    validate: Crucible.Stage.Validate,
    data_checks: Crucible.Stage.DataChecks,
    guardrails: Crucible.Stage.Guardrails,
    bench: Crucible.Stage.Bench,
    report: Crucible.Stage.Report
  },
  guardrail_adapter: Crucible.Stage.Guardrails.Noop

config :logger, :console,
  format: "[$level] $message\n",
  metadata: [:module]

import_config "#{config_env()}.exs"
