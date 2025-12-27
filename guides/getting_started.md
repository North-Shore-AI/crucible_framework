# Getting Started with Crucible Framework

A thin orchestration layer for ML experimentation pipelines.

## Installation

Add `crucible_framework` to your `mix.exs`:

```elixir
def deps do
  [
    {:crucible_framework, "~> 0.5.0"},
    {:crucible_ir, "~> 0.1.0"}  # Required for experiment/stage definitions
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
  stages: [
    %StageDef{name: :validate},
    %StageDef{name: :data_checks, options: %{required_fields: [:input, :expected]}},
    %StageDef{name: :report, options: %{format: :markdown, sink: :stdout}}
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
  assigns: %{examples: data},         # Initial context data
  validate_options: :warn             # :off, :warn, or :error
)
```

## Built-in Stages

Crucible Framework includes five built-in stages:

| Stage | Purpose |
|-------|---------|
| `:validate` | Pre-flight validation of pipeline stages |
| `:data_checks` | Validate examples in `context.assigns[:examples]` |
| `:guardrails` | Apply safety guardrail checks via adapter |
| `:bench` | Statistical analysis (requires `crucible_bench`) |
| `:report` | Generate and output reports |

### Example: Validation Pipeline

```elixir
experiment = %Experiment{
  id: "validation-check",
  name: "Validate Pipeline Configuration",
  stages: [
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
  stages: [
    %StageDef{name: :data_checks, options: %{
      required_fields: [:id, :input, :expected],
      fail_fast: false
    }},
    %StageDef{name: :report, options: %{
      format: :json,
      sink: {:file, "output/results.json"}
    }}
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
