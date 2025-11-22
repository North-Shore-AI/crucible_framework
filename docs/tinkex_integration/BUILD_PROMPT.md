# Crucible Framework Tinkex Integration - Build Prompt

**Date:** 2025-11-21
**Target:** Complete Tinkex integration into Crucible Framework
**Duration:** Multi-week intensive development
**Strategy:** Parallel multi-agent development with iterative review cycles

---

## Executive Summary

Build the complete Tinkex adapter layer for Crucible Framework, enabling LoRA fine-tuning, ML inference, and experiment orchestration through a unified Elixir API. This integrates the `tinkex` Elixir SDK into the crucible ecosystem.

---

## Required Reading

### Architecture Documents (READ ALL BEFORE STARTING)

```
S:\crucible_framework\docs\20251121\tinkex_integration\
├── 00_architecture_overview.md      # 6-layer architecture, API surface
├── 01_tinkex_adapter.md             # Session GenServer, checkpoints
├── 02_lora_training_interface.md    # LoRA config, loss registry
└── 03_ensemble_ml_integration.md    # AdapterPool, voting strategies
```

### Tinkex Source Code (STUDY THOROUGHLY)

```
S:\tinkex\
├── lib\tinkex\
│   ├── training_client.ex           # LoRA training operations
│   ├── sampling_client.ex           # Text generation
│   ├── service_client.ex            # Model/session management
│   ├── rest_client.ex               # Checkpoint management
│   ├── checkpoint_download.ex       # Artifact retrieval
│   ├── types\                       # All type definitions
│   │   ├── datum.ex
│   │   ├── model_input.ex
│   │   ├── sampling_params.ex
│   │   └── adam_params.ex
│   ├── config.ex                    # Configuration
│   └── telemetry.ex                 # Telemetry events
├── mix.exs                          # Dependencies
└── README.md                        # Usage examples
```

### Existing Crucible Framework Code

```
S:\crucible_framework\
├── lib\crucible_framework\
│   ├── ensemble.ex                  # Multi-model voting
│   ├── hedging.ex                   # Request hedging
│   ├── bench.ex                     # Statistical testing
│   ├── trace.ex                     # Causal tracing
│   ├── telemetry.ex                 # Metrics
│   └── harness.ex                   # Experiment orchestration
├── mix.exs
└── test\
```

### Related Crucible Libraries

```
S:\crucible_ensemble\lib\            # Voting strategies
S:\crucible_hedging\lib\             # Latency optimization
S:\crucible_bench\lib\               # Statistical tests
S:\crucible_telemetry\lib\           # Instrumentation
S:\crucible_harness\lib\             # Experiment DSL
S:\crucible_datasets\lib\            # Dataset loading
```

---

## Development Strategy

### Phase 1: Foundation (Agents 1-3 in Parallel)

**Agent 1: Core Tinkex Adapter**
```elixir
# Build these modules:
lib/crucible_framework/tinkex/
├── adapter.ex              # Main adapter behaviour
├── session.ex              # GenServer for session lifecycle
├── config.ex               # Tinkex-specific configuration
└── telemetry_bridge.ex     # Event translation
```

**Agent 2: LoRA Training Interface**
```elixir
# Build these modules:
lib/crucible_framework/lora/
├── config.ex               # LoRA hyperparameters struct
├── trainer.ex              # High-level training API
├── loss_registry.ex        # Custom loss function registration
└── gradient_hooks.ex       # Research gradient interception
```

**Agent 3: Checkpoint & Model Management**
```elixir
# Build these modules:
lib/crucible_framework/tinkex/
├── checkpoint_manager.ex   # Save/load/version checkpoints
├── model_registry.ex       # Track trained models
└── artifact_store.ex       # Local artifact caching
```

### Phase 2: Integration (Agents 4-6 in Parallel)

**Agent 4: Ensemble ML Integration**
```elixir
# Build these modules:
lib/crucible_framework/ensemble/
├── adapter_pool.ex         # Multiple sampling clients
├── ml_voting.ex            # ML-aware voting strategies
└── model_ensemble.ex       # Multi-model orchestration
```

**Agent 5: Harness Integration**
```elixir
# Extend existing harness:
lib/crucible_framework/harness/
├── tinkex_runner.ex        # Run experiments with Tinkex
├── ml_experiment.ex        # ML experiment configuration
└── training_reporter.ex    # Training metrics reporting
```

**Agent 6: Dataset Integration**
```elixir
# Bridge to crucible_datasets:
lib/crucible_framework/datasets/
├── ml_loader.ex            # Load datasets for training
├── batch_iterator.ex       # Efficient batching
└── tokenization.ex         # Tokenizer integration
```

### Phase 3: Advanced Features (Agents 7-8 in Parallel)

**Agent 7: Hedging for ML Inference**
```elixir
# Extend hedging for ML:
lib/crucible_framework/hedging/
├── inference_hedger.ex     # Hedged sampling requests
└── adaptive_routing.ex     # Route based on model performance
```

**Agent 8: Telemetry & Observability**
```elixir
# Complete telemetry:
lib/crucible_framework/telemetry/
├── ml_metrics.ex           # Training/inference metrics
├── experiment_tracker.ex   # Track experiment runs
└── dashboard_data.ex       # Data for crucible_ui
```

---

## Test-Driven Development Instructions

### TDD Workflow for Each Agent

1. **Write tests FIRST** before any implementation
2. Run tests to see them fail (Red)
3. Implement minimum code to pass (Green)
4. Refactor while keeping tests green (Refactor)
5. Repeat for each function/module

### Test Structure

```elixir
# test/crucible_framework/tinkex/adapter_test.exs
defmodule CrucibleFramework.Tinkex.AdapterTest do
  use ExUnit.Case, async: true

  # Use Mox for mocking Tinkex clients
  import Mox

  setup :verify_on_exit!

  describe "start_session/1" do
    test "creates training client with valid config" do
      # Arrange
      config = %CrucibleFramework.Lora.Config{
        base_model: "meta-llama/Llama-3.1-8B",
        rank: 32
      }

      # Mock Tinkex calls
      expect(Tinkex.ServiceClientMock, :new, fn -> {:ok, %{}} end)
      expect(Tinkex.ServiceClientMock, :create_lora_training_client, fn _, _ ->
        {:ok, %{model_id: "test-123"}}
      end)

      # Act
      result = CrucibleFramework.Tinkex.Adapter.start_session(config)

      # Assert
      assert {:ok, session} = result
      assert session.model_id == "test-123"
    end

    test "returns error with invalid config" do
      config = %CrucibleFramework.Lora.Config{base_model: nil}
      assert {:error, :invalid_config} =
        CrucibleFramework.Tinkex.Adapter.start_session(config)
    end
  end
end
```

### Required Test Coverage

Each module must have:
- Unit tests for all public functions
- Property-based tests for data transformations
- Integration tests for Tinkex communication
- Error case coverage (network failures, invalid inputs)
- Concurrent access tests where applicable

### Quality Gates (ALL MUST PASS)

```bash
# Run from S:\crucible_framework
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && mix test"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && mix dialyzer"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && mix compile --warnings-as-errors"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && mix format --check-formatted"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && mix credo --strict"
```

---

## Agent Coordination Protocol

### Spawn Pattern

```
┌─────────────────────────────────────────────────────┐
│  ORCHESTRATOR AGENT                                 │
│  - Spawns parallel work agents                      │
│  - Monitors progress                                │
│  - Triggers review cycles                           │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Agent 1 │   │ Agent 2 │   │ Agent 3 │
   │ Adapter │   │  LoRA   │   │Checkpts │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              ┌───────────────┐
              │ REVIEW AGENT  │
              │ - Run tests   │
              │ - Fix errors  │
              │ - Dialyzer    │
              └───────┬───────┘
                      │
                      ▼
              [Spawn more agents if needed]
              [Repeat until all green]
```

### Review Agent Instructions

After each parallel phase completes:

1. **Collect all work** from parallel agents
2. **Run full test suite**: `mix test --trace`
3. **Run dialyzer**: `mix dialyzer`
4. **Check warnings**: `mix compile --warnings-as-errors`
5. **Analyze failures** and categorize:
   - Type errors → specific fixes
   - Logic errors → may need agent re-work
   - Integration errors → cross-module issues
6. **Spawn fix agents** for each category
7. **Repeat** until all green

### Communication Between Agents

Each agent must:
- Document public API in `@moduledoc` and `@doc`
- Define clear `@type` and `@spec` for all functions
- Use consistent naming conventions
- Emit telemetry events for observability
- Handle errors with `{:ok, result} | {:error, reason}`

---

## Implementation Details

### Key Types to Define

```elixir
# lib/crucible_framework/lora/config.ex
defmodule CrucibleFramework.Lora.Config do
  @type t :: %__MODULE__{
    base_model: String.t(),
    rank: pos_integer(),
    alpha: float(),
    dropout: float(),
    target_modules: [String.t()],
    learning_rate: float(),
    batch_size: pos_integer(),
    max_steps: pos_integer(),
    checkpoint_interval: pos_integer()
  }

  defstruct [
    :base_model,
    rank: 16,
    alpha: 32.0,
    dropout: 0.05,
    target_modules: ["q_proj", "v_proj"],
    learning_rate: 1.0e-4,
    batch_size: 4,
    max_steps: 1000,
    checkpoint_interval: 100
  ]
end
```

### Adapter Behaviour

```elixir
# lib/crucible_framework/tinkex/adapter.ex
defmodule CrucibleFramework.Tinkex.Adapter do
  @callback start_session(CrucibleFramework.Lora.Config.t()) ::
    {:ok, session_id :: String.t()} | {:error, term()}

  @callback forward_backward(session_id :: String.t(), batch :: [map()]) ::
    {:ok, %{loss: float(), gradients: map()}} | {:error, term()}

  @callback optim_step(session_id :: String.t(), params :: map()) ::
    {:ok, :updated} | {:error, term()}

  @callback save_checkpoint(session_id :: String.t(), name :: String.t()) ::
    {:ok, checkpoint_path :: String.t()} | {:error, term()}

  @callback create_sampler(session_id :: String.t()) ::
    {:ok, sampler_id :: String.t()} | {:error, term()}

  @callback sample(sampler_id :: String.t(), prompt :: String.t(), params :: map()) ::
    {:ok, [String.t()]} | {:error, term()}
end
```

### Loss Function Registry

```elixir
# lib/crucible_framework/lora/loss_registry.ex
defmodule CrucibleFramework.Lora.LossRegistry do
  use GenServer

  # Built-in losses
  @builtin_losses %{
    cross_entropy: &__MODULE__.cross_entropy_loss/2,
    mse: &__MODULE__.mse_loss/2
  }

  def register(name, loss_fn) when is_function(loss_fn, 2) do
    GenServer.call(__MODULE__, {:register, name, loss_fn})
  end

  def get(name) do
    GenServer.call(__MODULE__, {:get, name})
  end

  # Example custom loss for CNS
  def topological_loss(outputs, targets) do
    # Compute beta_1 penalty for circular reasoning
    # This would integrate with CNS topology analysis
  end
end
```

---

## Dependencies to Add

Update `mix.exs`:

```elixir
defp deps do
  [
    # Existing deps...

    # Add Tinkex
    {:tinkex, path: "../tinkex"},

    # For property-based testing
    {:stream_data, "~> 0.6", only: [:test]},

    # For mocking
    {:mox, "~> 1.0", only: :test}
  ]
end
```

---

## Success Criteria

### Minimum Viable Integration

- [ ] Can start a training session via `CrucibleFramework.Lora.Trainer.start/1`
- [ ] Can run forward/backward pass with custom loss
- [ ] Can save and load checkpoints
- [ ] Can create sampling client from trained model
- [ ] Can generate text with sampling parameters
- [ ] All operations emit telemetry events
- [ ] Integration with crucible_harness for experiments

### Quality Metrics

- [ ] 100% of public functions have typespecs
- [ ] 100% of modules have documentation
- [ ] >90% test coverage
- [ ] Zero dialyzer warnings
- [ ] Zero compilation warnings
- [ ] All tests pass

### Integration Verification

```elixir
# This end-to-end test must pass:
test "full training loop with Tinkex" do
  config = %CrucibleFramework.Lora.Config{
    base_model: "meta-llama/Llama-3.1-8B",
    rank: 16,
    max_steps: 10
  }

  {:ok, trainer} = CrucibleFramework.Lora.Trainer.start(config)

  # Load dataset
  {:ok, dataset} = CrucibleFramework.Datasets.load(:scifact, split: :train)

  # Train
  {:ok, result} = CrucibleFramework.Lora.Trainer.train(trainer, dataset)
  assert result.final_loss < result.initial_loss

  # Save checkpoint
  {:ok, checkpoint} = CrucibleFramework.Lora.Trainer.save(trainer, "test-run")
  assert File.exists?(checkpoint.local_path)

  # Create sampler and generate
  {:ok, sampler} = CrucibleFramework.Lora.Trainer.create_sampler(trainer)
  {:ok, outputs} = CrucibleFramework.Sampling.generate(sampler, "Test prompt")
  assert length(outputs) > 0
end
```

---

## Iteration Protocol

### Week 1: Foundation
- Phase 1 agents build core adapters
- Review agent ensures tests pass
- Fix any integration issues

### Week 2: Integration
- Phase 2 agents build ensemble/harness/dataset integration
- Review agent runs full test suite
- Address cross-module issues

### Week 3: Advanced & Polish
- Phase 3 agents build hedging/telemetry
- Review agent does final quality check
- Documentation review and examples

### Week 4+: Hardening
- Performance optimization
- Edge case handling
- Real-world testing with actual Tinkex API
- Integration testing with CNS

---

## Path Formats Reminder

- **File tools** (Read/Write/Edit/Grep/Glob): `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\crucible_framework\...`
- **Bash commands**: `wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/crucible_framework && <command>"`

---

## Start Command

To begin implementation, an orchestrator agent should:

1. Read all required documents listed above
2. Spawn Phase 1 agents (1-3) in parallel
3. Wait for completion
4. Run review cycle
5. Spawn Phase 2 agents (4-6) in parallel
6. Repeat review cycle
7. Continue until all success criteria met

**The goal is maximum parallelization with quality gates at each phase.**
