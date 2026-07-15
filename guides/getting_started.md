# Getting Started with Crucible Framework

A thin orchestration layer for ML experimentation pipelines.

## Installation

Add `crucible_framework` to your `mix.exs`:

```elixir
def deps do
  [
    {:crucible_framework, "~> 0.5.3"},
    {:crucible_ir, "~> 0.2.1"}  # Required for experiment/stage definitions
  ]
end
```

Then fetch dependencies:

```bash
mix deps.get
```

## Quick Start

### 1. Define an Experiment

Experiments are defined using `CrucibleIR.Experiment` structs:

```elixir
alias CrucibleIR.{Experiment, StageDef}

experiment = %Experiment{
  id: "my-first-experiment",
  name: "My First Experiment",
  pipeline: [
    %StageDef{name: :validate},
    %StageDef{name: :data_checks, options: %{required_fields: [:input, :expected]}},
    %StageDef{name: :report}
  ]
}
```

### 2. Run the Pipeline

```elixir
{:ok, context} = CrucibleFramework.run(experiment,
  assigns: %{examples: my_data}
)

# Access results
IO.inspect(context.outputs)
IO.inspect(context.metrics)
```

### 3. Options

```elixir
CrucibleFramework.run(experiment,
  run_id: "custom-run-id",           # Custom run identifier (default: UUID)
  persist: false,                     # Disable database persistence
  enable_trace: true,                 # Enable causal tracing (requires crucible_trace)
  enable_lineage: true,               # Emit lineage spans/artifacts (default: true)
  trace_id: "trace-uuid",             # Optional lineage trace id
  assigns: %{examples: data},         # Initial context data
  validate_options: :warn             # :off, :warn, or :error
)
```

## Built-in Stages

Crucible Framework includes the core stages below, plus `Crucible.Stage.PlanStep`
for plan-driven pipelines:

| Stage | Purpose |
|-------|---------|
| `:validate` | Pre-flight validation of pipeline stages |
| `:data_checks` | Validate examples in `context.assigns[:examples]` |
| `:guardrails` | Apply safety guardrail checks via adapter |
| `:bench` | Statistical analysis (requires `crucible_bench`) |
| `:report` | Generate and output reports |

## Plan-Driven Pipelines (Jido.Plan)

Use `Crucible.PlanAdapter` to compile a `Jido.Plan` into pipeline stages:

```elixir
alias Jido.Plan
alias Crucible.PlanAdapter
alias CrucibleIR.{BackendRef, Experiment}

plan =
  Plan.new()
  |> Plan.add(:fetch, MyApp.Actions.Fetch)
  |> Plan.add(:summarize, MyApp.Actions.Summarize, depends_on: :fetch)

{:ok, stage_defs} = PlanAdapter.to_stage_defs(plan)

experiment = %Experiment{
  id: "plan-demo",
  backend: %BackendRef{id: :noop},
  pipeline: stage_defs
}

{:ok, ctx} = CrucibleFramework.run(experiment, persist: false)

IO.inspect(ctx.assigns.plan_results)
```

`Crucible.Stage.PlanStep` executes each action using `Jido.Exec` when available,
falling back to `action.run/2` if `jido_action` is not installed.

### Example: Validation Pipeline

```elixir
experiment = %Experiment{
  id: "validation-check",
  name: "Validate Pipeline Configuration",
  pipeline: [
    %StageDef{name: :validate, options: %{strict: true}}
  ]
}

{:ok, ctx} = CrucibleFramework.run(experiment)
```

### Example: Data Processing Pipeline

```elixir
experiment = %Experiment{
  id: "data-processing",
  name: "Process and Report",
  pipeline: [
    %StageDef{name: :data_checks, options: %{
      required_fields: [:id, :input, :expected],
      fail_fast: false
    }},
    %StageDef{name: :report}
  ],
  outputs: [
    %CrucibleIR.OutputSpec{
      name: :summary,
      formats: [:json],
      sink: :file,
      options: %{path: "output/results.json"}
    }
  ]
}

examples = [
  %{id: 1, input: "test", expected: "result"},
  %{id: 2, input: "test2", expected: "result2"}
]

{:ok, ctx} = CrucibleFramework.run(experiment,
  assigns: %{examples: examples}
)
```

## Working with Context

The `Crucible.Context` struct flows through all stages:

```elixir
# Access after pipeline completes
context.outputs        # List of stage outputs
context.metrics        # Map of collected metrics
context.artifacts      # Map of generated artifacts
context.assigns        # Domain-specific data

# Check stage completion
Crucible.Context.stage_completed?(context, :validate)
Crucible.Context.completed_stages(context)
```

## List Available Stages

Use the Mix task to see registered stages:

```bash
# List all stages
mix crucible.stages

# Show schema for specific stage
mix crucible.stages --name validate
```

## Next Steps

- [Creating Custom Stages](stages.md) - Build your own pipeline stages
- [Configuration Guide](configuration.md) - Registry, adapters, and optional dependencies
