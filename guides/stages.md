# Creating Custom Stages

Stages are composable pipeline steps that implement the `Crucible.Stage` behaviour.

## Stage Behaviour

Every stage must implement two callbacks:

```elixir
@callback describe(keyword()) :: map()
@callback run(Crucible.Context.t(), map()) :: {:ok, Crucible.Context.t()} | {:error, term()}
```

## Minimal Stage Example

```elixir
defmodule MyApp.Stages.Transform do
  @behaviour Crucible.Stage

  @impl true
  def describe(_opts) do
    %{
      name: :transform,
      description: "Transforms input data",
      required: [],
      optional: [:multiplier],
      types: %{
        multiplier: :integer
      },
      defaults: %{
        multiplier: 1
      }
    }
  end

  @impl true
  def run(context, opts) do
    multiplier = Map.get(opts, :multiplier, 1)

    examples = context.assigns[:examples] || []
    transformed = Enum.map(examples, fn ex ->
      Map.update(ex, :value, 0, &(&1 * multiplier))
    end)

    {:ok, Crucible.Context.assign(context, :examples, transformed)}
  end
end
```

## Schema Specification

The `describe/1` callback returns a canonical schema:

```elixir
%{
  # Required fields
  name: :my_stage,                    # Stage identifier (atom)
  description: "What this stage does", # Human-readable description
  required: [:input_field],           # Required option keys
  optional: [:timeout, :format],      # Optional option keys
  types: %{                           # Type specifications
    input_field: :atom,
    timeout: :integer,
    format: {:enum, [:json, :csv]}
  },

  # Optional fields
  defaults: %{                        # Default values
    timeout: 5000,
    format: :json
  },
  version: "1.0.0"                    # Schema version
}
```

### Supported Types

| Type | Example |
|------|---------|
| `:string` | `"hello"` |
| `:integer` | `42` |
| `:float` | `3.14` |
| `:boolean` | `true` |
| `:atom` | `:example` |
| `:map` | `%{key: "value"}` |
| `:list` | `[1, 2, 3]` |
| `:module` | `MyModule` |
| `:any` | Any value |
| `{:struct, Module}` | `%MyStruct{}` |
| `{:enum, [values]}` | `{:enum, [:a, :b, :c]}` |
| `{:list, inner_type}` | `{:list, :string}` |
| `{:map, key_type, val_type}` | `{:map, :atom, :string}` |
| `{:tuple, [types]}` | `{:tuple, [:atom, :integer]}` |
| `{:function, arity}` | `{:function, 2}` |
| `{:union, [types]}` | `{:union, [:string, :integer]}` |

## Using Context Helpers

The `Crucible.Context` module provides helpers for stage implementations:

```elixir
def run(context, opts) do
  # Store metrics
  context = Crucible.Context.put_metric(context, :processed_count, 100)
  context = Crucible.Context.update_metric(context, :total, &(&1 + 1))

  # Add outputs
  context = Crucible.Context.add_output(context, %{result: "data"})

  # Store artifacts
  context = Crucible.Context.put_artifact(context, :report, %{
    path: "/tmp/report.md",
    format: :markdown
  })

  # Access/modify assigns
  data = context.assigns[:examples]
  context = Crucible.Context.assign(context, :processed, true)

  {:ok, context}
end
```

## Registering Custom Stages

Add stages to the registry in your config:

```elixir
# config/config.exs
config :crucible_framework,
  stage_registry: %{
    validate: Crucible.Stage.Validate,
    data_checks: Crucible.Stage.DataChecks,
    transform: MyApp.Stages.Transform,      # Your custom stage
    analyze: MyApp.Stages.Analyze           # Another custom stage
  }
```

Then use by name in experiments:

```elixir
%Experiment{
  stages: [
    %StageDef{name: :transform, options: %{multiplier: 2}},
    %StageDef{name: :analyze}
  ]
}
```

## Using Explicit Modules

Alternatively, specify the module directly:

```elixir
%Experiment{
  stages: [
    %StageDef{module: MyApp.Stages.Transform, options: %{multiplier: 2}}
  ]
}
```

## Error Handling

Return `{:error, reason}` to halt the pipeline:

```elixir
def run(context, opts) do
  case validate_input(context.assigns[:data]) do
    :ok ->
      {:ok, process(context)}
    {:error, reason} ->
      {:error, {:validation_failed, reason}}
  end
end
```

The pipeline runner wraps errors with stage context:
```elixir
{:error, {:stage_name, original_error, context_at_failure}}
```

## Options Validation

Enable validation to check options against your schema:

```elixir
CrucibleFramework.run(experiment,
  validate_options: :error  # :off, :warn, or :error
)
```

- `:off` - No validation (default, fastest)
- `:warn` - Log warnings but continue
- `:error` - Fail immediately on validation errors

## Testing Stages

```elixir
defmodule MyApp.Stages.TransformTest do
  use ExUnit.Case

  alias Crucible.Context
  alias MyApp.Stages.Transform

  test "describe/1 returns valid schema" do
    schema = Transform.describe([])
    assert schema.name == :transform
    assert :multiplier in schema.optional
  end

  test "run/2 transforms examples" do
    context = %Context{
      assigns: %{examples: [%{value: 10}, %{value: 20}]}
    }

    {:ok, result} = Transform.run(context, %{multiplier: 2})

    assert [%{value: 20}, %{value: 40}] = result.assigns[:examples]
  end
end
```
