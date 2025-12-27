# Configuration Guide

Configure Crucible Framework through application environment.

## Stage Registry

Map stage names to modules:

```elixir
# config/config.exs
config :crucible_framework,
  stage_registry: %{
    validate: Crucible.Stage.Validate,
    data_checks: Crucible.Stage.DataChecks,
    guardrails: Crucible.Stage.Guardrails,
    bench: Crucible.Stage.Bench,
    report: Crucible.Stage.Report,
    # Add your custom stages
    my_stage: MyApp.Stages.MyStage
  }
```

## Persistence

Configure database persistence (optional):

```elixir
config :crucible_framework,
  ecto_repos: [CrucibleFramework.Repo],
  enable_repo: true  # Set to false to disable persistence
```

Database URL in runtime config:

```elixir
# config/runtime.exs
config :crucible_framework, CrucibleFramework.Repo,
  url: System.get_env("DATABASE_URL")
```

Disable persistence per-run:

```elixir
CrucibleFramework.run(experiment, persist: false)
```

## Guardrails Adapter

Configure the guardrails adapter:

```elixir
config :crucible_framework,
  guardrail_adapter: Crucible.Stage.Guardrails.Noop  # Default: no-op
```

Implement custom adapter:

```elixir
defmodule MyApp.GuardrailAdapter do
  @behaviour Crucible.Stage.Guardrails.Adapter

  @impl true
  def check(examples, opts) do
    # Return {:ok, []} for no violations
    # Return {:ok, violations} with list of violation maps
    {:ok, []}
  end
end

# config/config.exs
config :crucible_framework,
  guardrail_adapter: MyApp.GuardrailAdapter
```

## Optional Dependencies

### crucible_bench (Statistical Analysis)

The `:bench` stage requires `crucible_bench`:

```elixir
# mix.exs
{:crucible_bench, "~> 0.1.0"}
```

Without it, the stage returns:
```elixir
{:error, {:missing_dependency, :crucible_bench}}
```

### crucible_trace (Causal Tracing)

Enable tracing with `crucible_trace`:

```elixir
# mix.exs
{:crucible_trace, "~> 0.1.0"}
```

Enable per-run:

```elixir
CrucibleFramework.run(experiment, enable_trace: true)
```

Without the dependency, tracing is disabled with a warning.

## Environment-Specific Config

```elixir
# config/dev.exs
config :crucible_framework,
  enable_repo: true

# config/test.exs
config :crucible_framework,
  enable_repo: false  # Use in-memory for tests

# config/prod.exs
config :crucible_framework,
  enable_repo: true
```

## Full Configuration Example

```elixir
# config/config.exs
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

import_config "#{config_env()}.exs"
```

## Querying the Registry

Programmatically access registered stages:

```elixir
# Get module for a stage name
{:ok, module} = Crucible.Registry.stage_module(:validate)

# List all registered stage names
Crucible.Registry.list_stages()
#=> [:validate, :data_checks, :guardrails, :bench, :report]

# Get all stages with their schemas
Crucible.Registry.list_stages_with_schemas()

# Get schema for specific stage
{:ok, schema} = Crucible.Registry.stage_schema(:validate)
```

## CLI Stage Discovery

```bash
# List all registered stages
mix crucible.stages

# Show detailed schema for a stage
mix crucible.stages --name validate
mix crucible.stages -n data_checks
```
