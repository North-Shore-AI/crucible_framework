# CrucibleFramework Examples

These examples are runnable scripts aligned with the current Crucible IR and pipeline runner.

## Quick Start

Run a single example from the repo root:

```bash
mix run examples/01_core_pipeline.exs
```

Run all examples:

```bash
./examples/run_all.sh
```

## Examples

### 01_core_pipeline.exs

**Purpose:** Minimal pipeline using only built-in stages.

**Highlights:**
- Seeds demo examples into `context.assigns`
- Runs `validate`, `data_checks`, `guardrails`, and `report`
- Requires no optional dependencies

---

### 02_bench_optional.exs

**Purpose:** Bench stage demo with optional dependency.

**Highlights:**
- Seeds baseline vs treatment metrics
- Runs `Crucible.Stage.Bench` when `crucible_bench` is installed
- Skips the bench stage with a clear message when missing

---

### 03_trace_optional.exs

**Purpose:** Trace integration demo with optional dependency.

**Highlights:**
- Emits a custom decision event via `Crucible.TraceIntegration`
- Enables `enable_trace: true` only when `crucible_trace` is installed
- Shows trace event counts and export status

## Optional Dependencies

Some examples enable extra functionality when these deps are available:

```elixir
def deps do
  [
    {:crucible_framework, "~> 0.5.1"},
    {:crucible_bench, "~> 0.4.0"},
    {:crucible_trace, "~> 0.3.0"}
  ]
end
```

## Notes

- Examples run with `persist: false` to avoid database setup.
- If you see compile errors, run `mix deps.get` and `mix compile`.
