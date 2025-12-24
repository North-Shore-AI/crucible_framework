defmodule Crucible.IR do
  @moduledoc """
  Backwards-compatible aliases for IR structs from CrucibleIR package.

  This module provides backwards compatibility for code that previously used
  `Crucible.IR.*` module names. All functionality is now provided by the
  `crucible_ir` package.

  ## Migration Guide

  **Old (deprecated):**
  ```elixir
  alias Crucible.IR.Experiment
  alias Crucible.IR.BackendRef
  ```

  **New (recommended):**
  ```elixir
  alias CrucibleIR.Experiment
  alias CrucibleIR.BackendRef
  ```

  ## Deprecation Notice

  These aliases are provided for backwards compatibility and will be removed
  in v1.0.0. Please update your code to use `CrucibleIR` directly.

  ## Available Aliases

  - Crucible.IR.Experiment → CrucibleIR.Experiment
  - Crucible.IR.DatasetRef → CrucibleIR.DatasetRef
  - Crucible.IR.BackendRef → CrucibleIR.BackendRef
  - Crucible.IR.StageDef → CrucibleIR.StageDef
  - Crucible.IR.OutputSpec → CrucibleIR.OutputSpec
  - Crucible.IR.ReliabilityConfig → CrucibleIR.Reliability.Config
  - Crucible.IR.EnsembleConfig → CrucibleIR.Reliability.Ensemble
  - Crucible.IR.HedgingConfig → CrucibleIR.Reliability.Hedging
  - Crucible.IR.StatsConfig → CrucibleIR.Reliability.Stats
  - Crucible.IR.FairnessConfig → CrucibleIR.Reliability.Fairness
  - Crucible.IR.GuardrailConfig → CrucibleIR.Reliability.Guardrail
  """

  # Core IR modules
  defmodule Experiment do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Experiment
    defdelegate __struct__(kv), to: CrucibleIR.Experiment
  end

  defmodule DatasetRef do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.DatasetRef
    defdelegate __struct__(kv), to: CrucibleIR.DatasetRef
  end

  defmodule BackendRef do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.BackendRef
    defdelegate __struct__(kv), to: CrucibleIR.BackendRef
  end

  defmodule StageDef do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.StageDef
    defdelegate __struct__(kv), to: CrucibleIR.StageDef
  end

  defmodule OutputSpec do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.OutputSpec
    defdelegate __struct__(kv), to: CrucibleIR.OutputSpec
  end

  # Reliability config modules
  defmodule ReliabilityConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Config
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Config
  end

  defmodule EnsembleConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Ensemble
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Ensemble
  end

  defmodule HedgingConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Hedging
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Hedging
  end

  defmodule StatsConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Stats
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Stats
  end

  defmodule FairnessConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Fairness
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Fairness
  end

  defmodule GuardrailConfig do
    @moduledoc false
    defdelegate __struct__(), to: CrucibleIR.Reliability.Guardrail
    defdelegate __struct__(kv), to: CrucibleIR.Reliability.Guardrail
  end
end
