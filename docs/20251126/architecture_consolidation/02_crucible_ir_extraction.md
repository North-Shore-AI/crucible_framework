# Crucible IR Extraction Design

**Date:** 2025-11-26
**Status:** Design Proposal

---

## Objective

Extract the Intermediate Representation (IR) from `crucible_framework` into a new standalone library `crucible_ir` that serves as the shared vocabulary for all Crucible ecosystem components.

## Current IR Structure

Located in `crucible_framework/lib/crucible/ir/`:

```elixir
# lib/crucible/ir/experiment.ex
defmodule Crucible.IR.Experiment do
  @enforce_keys [:id, :backend, :pipeline]
  defstruct [
    :id, :description, :owner, :created_at, :updated_at,
    tags: [], metadata: %{}, dataset: nil, pipeline: [],
    backend: nil, reliability: %ReliabilityConfig{}, outputs: []
  ]
end

# lib/crucible/ir/reliability_config.ex
defmodule Crucible.IR.ReliabilityConfig do
  defstruct ensemble: %EnsembleConfig{},
            hedging: %HedgingConfig{},
            guardrails: %GuardrailConfig{},
            stats: %StatsConfig{},
            fairness: %FairnessConfig{}
end
```

## New Library Structure

### Package: `crucible_ir`

```
crucible_ir/
├── mix.exs
├── README.md
├── CHANGELOG.md
├── lib/
│   ├── crucible_ir.ex              # Main module with convenience functions
│   └── crucible_ir/
│       ├── experiment.ex           # Experiment definition
│       ├── dataset_ref.ex          # Dataset reference
│       ├── backend_ref.ex          # Backend reference
│       ├── stage_def.ex            # Stage definition
│       ├── output_spec.ex          # Output specification
│       ├── reliability/            # Reliability configs subdirectory
│       │   ├── config.ex           # ReliabilityConfig container
│       │   ├── ensemble.ex         # EnsembleConfig
│       │   ├── hedging.ex          # HedgingConfig
│       │   ├── stats.ex            # StatsConfig
│       │   ├── fairness.ex         # FairnessConfig
│       │   └── guardrail.ex        # GuardrailConfig
│       ├── validation.ex           # Struct validation helpers
│       └── serialization.ex        # JSON encoding/decoding
└── test/
    ├── crucible_ir_test.exs
    └── crucible_ir/
        ├── experiment_test.exs
        ├── validation_test.exs
        └── serialization_test.exs
```

### mix.exs

```elixir
defmodule CrucibleIR.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_ir"

  def project do
    [
      app: :crucible_ir,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Intermediate Representation for the Crucible ML reliability ecosystem",
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      maintainers: ["North-Shore-AI"]
    ]
  end

  defp docs do
    [
      main: "CrucibleIR",
      extras: ["README.md", "CHANGELOG.md"]
    ]
  end
end
```

## Module Definitions

### Main Module

```elixir
# lib/crucible_ir.ex
defmodule CrucibleIR do
  @moduledoc """
  Intermediate Representation for the Crucible ML reliability ecosystem.

  This library provides the canonical data structures for defining experiments,
  configurations, and pipeline stages. It serves as the shared vocabulary
  between all Crucible components.

  ## Core Structures

  - `CrucibleIR.Experiment` - Top-level experiment definition
  - `CrucibleIR.DatasetRef` - Dataset reference
  - `CrucibleIR.BackendRef` - Backend reference
  - `CrucibleIR.StageDef` - Pipeline stage definition
  - `CrucibleIR.OutputSpec` - Output artifact specification

  ## Reliability Configurations

  - `CrucibleIR.Reliability.Config` - Container for all reliability settings
  - `CrucibleIR.Reliability.Ensemble` - Multi-model voting configuration
  - `CrucibleIR.Reliability.Hedging` - Request hedging configuration
  - `CrucibleIR.Reliability.Stats` - Statistical testing configuration
  - `CrucibleIR.Reliability.Fairness` - Fairness evaluation configuration
  - `CrucibleIR.Reliability.Guardrail` - Safety guardrail configuration

  ## Usage

      alias CrucibleIR.{Experiment, DatasetRef, BackendRef, StageDef}
      alias CrucibleIR.Reliability

      experiment = %Experiment{
        id: "my_experiment",
        pipeline: [
          %StageDef{name: :data_load},
          %StageDef{name: :backend_call},
          %StageDef{name: :bench}
        ],
        backend: %BackendRef{id: :tinkex, profile: :lora_finetune},
        reliability: %Reliability.Config{
          stats: %Reliability.Stats{tests: [:ttest, :bootstrap], alpha: 0.05}
        }
      }
  """

  # Convenience aliases for common usage
  defdelegate validate(struct), to: CrucibleIR.Validation
  defdelegate to_json(struct), to: CrucibleIR.Serialization
  defdelegate from_json(json, type), to: CrucibleIR.Serialization
end
```

### Experiment

```elixir
# lib/crucible_ir/experiment.ex
defmodule CrucibleIR.Experiment do
  @moduledoc """
  Top-level experiment definition.

  An Experiment is the primary unit of work in the Crucible ecosystem.
  It describes what data to use, what pipeline stages to execute,
  which backend to target, and what reliability features to enable.
  """

  @derive {Jason.Encoder, only: [
    :id, :description, :owner, :tags, :metadata,
    :dataset, :pipeline, :backend, :reliability, :outputs,
    :created_at, :updated_at
  ]}

  alias CrucibleIR.{BackendRef, DatasetRef, OutputSpec, StageDef}
  alias CrucibleIR.Reliability

  @type t :: %__MODULE__{
    id: String.t(),
    description: String.t() | nil,
    owner: String.t() | nil,
    tags: [String.t()],
    metadata: map(),
    dataset: DatasetRef.t() | [DatasetRef.t()] | nil,
    pipeline: [StageDef.t()],
    backend: BackendRef.t(),
    reliability: Reliability.Config.t(),
    outputs: [OutputSpec.t()],
    created_at: DateTime.t() | nil,
    updated_at: DateTime.t() | nil
  }

  @enforce_keys [:id, :backend, :pipeline]
  defstruct [
    :id,
    :description,
    :owner,
    tags: [],
    metadata: %{},
    dataset: nil,
    pipeline: [],
    backend: nil,
    reliability: %Reliability.Config{},
    outputs: [],
    created_at: nil,
    updated_at: nil
  ]

  @doc """
  Creates a new experiment with the given attributes.

  ## Examples

      iex> CrucibleIR.Experiment.new(
      ...>   id: "exp_001",
      ...>   backend: %BackendRef{id: :tinkex},
      ...>   pipeline: [%StageDef{name: :data_load}]
      ...> )
      %CrucibleIR.Experiment{id: "exp_001", ...}
  """
  def new(attrs) when is_list(attrs) or is_map(attrs) do
    struct!(__MODULE__, attrs)
  end

  @doc """
  Adds a stage to the experiment pipeline.
  """
  def add_stage(%__MODULE__{} = exp, %StageDef{} = stage) do
    %{exp | pipeline: exp.pipeline ++ [stage]}
  end

  @doc """
  Updates the reliability configuration.
  """
  def with_reliability(%__MODULE__{} = exp, %Reliability.Config{} = config) do
    %{exp | reliability: config}
  end
end
```

### Reliability Configs

```elixir
# lib/crucible_ir/reliability/config.ex
defmodule CrucibleIR.Reliability.Config do
  @moduledoc """
  Container for all reliability feature configurations.
  """

  @derive {Jason.Encoder, only: [:ensemble, :hedging, :guardrails, :stats, :fairness]}

  alias CrucibleIR.Reliability.{Ensemble, Hedging, Guardrail, Stats, Fairness}

  @type t :: %__MODULE__{
    ensemble: Ensemble.t(),
    hedging: Hedging.t(),
    guardrails: Guardrail.t(),
    stats: Stats.t(),
    fairness: Fairness.t()
  }

  defstruct ensemble: %Ensemble{},
            hedging: %Hedging{},
            guardrails: %Guardrail{},
            stats: %Stats{},
            fairness: %Fairness{}
end

# lib/crucible_ir/reliability/ensemble.ex
defmodule CrucibleIR.Reliability.Ensemble do
  @moduledoc """
  Configuration for multi-model ensemble voting.

  ## Strategies

  - `:none` - No ensemble, single model
  - `:majority` - Simple majority voting
  - `:weighted` - Weighted by model confidence/quality
  - `:best_confidence` - Select highest confidence response
  - `:unanimous` - Require all models to agree

  ## Execution Modes

  - `:parallel` - Run all models concurrently
  - `:sequential` - Run models one at a time
  - `:hedged` - Start with one, add others if slow
  - `:cascade` - Try models in order until success
  """

  @derive Jason.Encoder

  @type strategy :: :none | :majority | :weighted | :best_confidence | :unanimous
  @type execution_mode :: :parallel | :sequential | :hedged | :cascade

  @type t :: %__MODULE__{
    strategy: strategy(),
    execution_mode: execution_mode(),
    models: [atom() | String.t()],
    weights: %{(atom() | String.t()) => float()},
    min_agreement: float(),
    timeout_ms: pos_integer(),
    options: map()
  }

  defstruct strategy: :none,
            execution_mode: :parallel,
            models: [],
            weights: %{},
            min_agreement: 0.5,
            timeout_ms: 30_000,
            options: %{}
end

# lib/crucible_ir/reliability/hedging.ex
defmodule CrucibleIR.Reliability.Hedging do
  @moduledoc """
  Configuration for request hedging to reduce tail latency.

  ## Strategies

  - `:off` - No hedging
  - `:fixed` - Hedge after fixed delay
  - `:percentile` - Hedge based on latency percentile
  - `:adaptive` - Dynamically adjust based on recent latency
  - `:workload_aware` - Consider queue depth and load
  """

  @derive Jason.Encoder

  @type strategy :: :off | :fixed | :percentile | :adaptive | :workload_aware

  @type t :: %__MODULE__{
    strategy: strategy(),
    delay_ms: pos_integer(),
    percentile: float(),
    max_hedges: pos_integer(),
    budget_percent: float(),
    options: map()
  }

  defstruct strategy: :off,
            delay_ms: 100,
            percentile: 0.95,
            max_hedges: 2,
            budget_percent: 0.1,
            options: %{}
end

# lib/crucible_ir/reliability/stats.ex
defmodule CrucibleIR.Reliability.Stats do
  @moduledoc """
  Configuration for statistical testing and analysis.

  ## Available Tests

  - `:ttest` - Two-sample t-test
  - `:welch_ttest` - Welch's t-test (unequal variances)
  - `:paired_ttest` - Paired t-test
  - `:mann_whitney` - Mann-Whitney U test (non-parametric)
  - `:wilcoxon` - Wilcoxon signed-rank test
  - `:bootstrap` - Bootstrap confidence intervals
  - `:anova` - One-way ANOVA
  - `:kruskal_wallis` - Kruskal-Wallis test
  """

  @derive Jason.Encoder

  @type test_type :: :ttest | :welch_ttest | :paired_ttest | :mann_whitney |
                     :wilcoxon | :bootstrap | :anova | :kruskal_wallis

  @type t :: %__MODULE__{
    tests: [test_type()],
    alpha: float(),
    confidence_level: float(),
    effect_size_type: atom(),
    multiple_testing_correction: atom() | nil,
    bootstrap_iterations: pos_integer(),
    options: map()
  }

  defstruct tests: [:ttest, :bootstrap],
            alpha: 0.05,
            confidence_level: 0.95,
            effect_size_type: :cohens_d,
            multiple_testing_correction: nil,
            bootstrap_iterations: 10_000,
            options: %{}
end

# lib/crucible_ir/reliability/fairness.ex
defmodule CrucibleIR.Reliability.Fairness do
  @moduledoc """
  Configuration for fairness evaluation.

  ## Metrics

  - `:demographic_parity` - Equal positive rates across groups
  - `:equalized_odds` - Equal TPR and FPR across groups
  - `:equal_opportunity` - Equal TPR across groups
  - `:predictive_parity` - Equal precision across groups
  """

  @derive Jason.Encoder

  @type metric :: :demographic_parity | :equalized_odds | :equal_opportunity | :predictive_parity

  @type t :: %__MODULE__{
    enabled: boolean(),
    metrics: [metric()],
    group_by: atom() | String.t() | nil,
    threshold: float(),
    fail_on_violation: boolean(),
    options: map()
  }

  defstruct enabled: false,
            metrics: [:demographic_parity, :equalized_odds],
            group_by: nil,
            threshold: 0.1,
            fail_on_violation: false,
            options: %{}
end

# lib/crucible_ir/reliability/guardrail.ex
defmodule CrucibleIR.Reliability.Guardrail do
  @moduledoc """
  Configuration for safety guardrails and content moderation.

  ## Profiles

  - `:default` - Standard safety checks
  - `:strict` - Enhanced safety, block more content
  - `:permissive` - Minimal blocking, log only
  - `:custom` - User-defined rules
  """

  @derive Jason.Encoder

  @type profile :: :default | :strict | :permissive | :custom

  @type t :: %__MODULE__{
    profiles: [profile()],
    prompt_injection_detection: boolean(),
    jailbreak_detection: boolean(),
    pii_detection: boolean(),
    pii_redaction: boolean(),
    content_moderation: boolean(),
    fail_on_detection: boolean(),
    options: map()
  }

  defstruct profiles: [:default],
            prompt_injection_detection: true,
            jailbreak_detection: true,
            pii_detection: true,
            pii_redaction: false,
            content_moderation: false,
            fail_on_detection: false,
            options: %{}
end
```

### Other Core Structs

```elixir
# lib/crucible_ir/dataset_ref.ex
defmodule CrucibleIR.DatasetRef do
  @moduledoc """
  Reference to a dataset for loading.
  """

  @derive Jason.Encoder

  @type t :: %__MODULE__{
    provider: atom(),
    name: String.t(),
    split: atom(),
    options: map()
  }

  defstruct provider: :crucible_datasets,
            name: nil,
            split: :train,
            options: %{}
end

# lib/crucible_ir/backend_ref.ex
defmodule CrucibleIR.BackendRef do
  @moduledoc """
  Reference to a backend for training/inference.
  """

  @derive Jason.Encoder

  @type t :: %__MODULE__{
    id: atom(),
    profile: atom(),
    options: map()
  }

  @enforce_keys [:id]
  defstruct [:id, profile: :default, options: %{}]
end

# lib/crucible_ir/stage_def.ex
defmodule CrucibleIR.StageDef do
  @moduledoc """
  Definition of a pipeline stage.
  """

  @derive Jason.Encoder

  @type t :: %__MODULE__{
    name: atom(),
    module: module() | nil,
    options: map(),
    enabled: boolean()
  }

  @enforce_keys [:name]
  defstruct [:name, module: nil, options: %{}, enabled: true]
end

# lib/crucible_ir/output_spec.ex
defmodule CrucibleIR.OutputSpec do
  @moduledoc """
  Specification for experiment output artifacts.
  """

  @derive Jason.Encoder

  @type format :: :markdown | :json | :html | :latex | :jupyter | :csv
  @type sink :: :file | :database | :s3 | :stdout

  @type t :: %__MODULE__{
    name: atom(),
    formats: [format()],
    sink: sink(),
    options: map()
  }

  @enforce_keys [:name]
  defstruct [:name, formats: [:markdown], sink: :file, options: %{}]
end
```

## Backwards Compatibility

### Alias Module in crucible_framework

```elixir
# lib/crucible/ir.ex (in crucible_framework)
defmodule Crucible.IR do
  @moduledoc """
  Backwards-compatible aliases for IR structs.

  These aliases point to the new `crucible_ir` library.
  Deprecated: use `CrucibleIR.*` directly.
  """

  # Re-export all structs
  defdelegate experiment(), to: CrucibleIR.Experiment
  defdelegate dataset_ref(), to: CrucibleIR.DatasetRef
  defdelegate backend_ref(), to: CrucibleIR.BackendRef
  defdelegate stage_def(), to: CrucibleIR.StageDef
  defdelegate output_spec(), to: CrucibleIR.OutputSpec
end

# Deprecated module aliases
defmodule Crucible.IR.Experiment do
  @moduledoc false
  defdelegate __struct__(), to: CrucibleIR.Experiment
  defdelegate __struct__(kv), to: CrucibleIR.Experiment
end

# ... similar for other modules
```

## Migration Checklist

- [ ] Create new `crucible_ir` repository
- [ ] Copy and adapt structs from `crucible_framework/lib/crucible/ir/`
- [ ] Add Jason encoder derivations
- [ ] Add helper functions (new/1, validation)
- [ ] Write comprehensive tests
- [ ] Publish to Hex.pm
- [ ] Update `crucible_framework` to depend on `crucible_ir`
- [ ] Add backwards-compatible aliases
- [ ] Update documentation
- [ ] Deprecation warnings for old paths
