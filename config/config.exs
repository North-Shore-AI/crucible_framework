import Config

# Note: Repo is NOT started by default. Host applications should:
# 1. Configure: config :crucible_framework, repo: MyApp.Repo
# 2. Start their own Repo in their supervision tree
#
# For standalone use, set start_repo: true and configure the Repo below.
config :crucible_framework,
  ecto_repos: [CrucibleFramework.Repo],
  start_repo: false,
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
