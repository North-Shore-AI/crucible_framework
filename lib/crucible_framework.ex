defmodule CrucibleFramework do
  @moduledoc """
  CrucibleFramework: A Scientifically-Rigorous Infrastructure for LLM Reliability and Performance Research

  This is a documentation and coordination module that provides:
  - Version information
  - Component library references
  - Quick access helpers for common operations
  - Integration examples

  ## Overview

  The Crucible Framework is built on a 6-layer architecture with 8 independent OTP applications:

  ### Layer 6: Orchestration
  - ResearchHarness - Experiment DSL and automation

  ### Layer 5: Analysis & Reporting
  - Bench - Statistical testing
  - TelemetryResearch - Metrics collection
  - Reporter - Multi-format output

  ### Layer 4: Reliability Strategies
  - Ensemble - Multi-model voting
  - Hedging - Tail latency reduction

  ### Layer 3: Transparency
  - CausalTrace - Decision provenance

  ### Layer 2: Data Management
  - DatasetManager - Benchmark integration

  ## Quick Start

  ```elixir
  # Get framework version
  CrucibleFramework.version()

  # List all component libraries
  CrucibleFramework.components()

  # Get component status
  CrucibleFramework.component_status(:ensemble)
  ```

  ## Component Libraries

  Each component is maintained as a separate package:

  - `ensemble` - Multi-model ensemble voting
  - `hedging` - Request hedging for latency reduction
  - `bench` - Statistical testing suite
  - `telemetry_research` - Research-grade instrumentation
  - `dataset_manager` - Benchmark dataset management
  - `causal_trace` - LLM decision provenance
  - `research_harness` - Experiment orchestration
  - `reporter` - Multi-format reporting

  See individual component documentation for detailed usage.
  """

  @version Mix.Project.config()[:version]

  @doc """
  Returns the framework version.

  ## Examples

      iex> CrucibleFramework.version()
      "0.1.4"
  """
  @spec version() :: String.t()
  def version, do: @version

  @doc """
  Lists all component libraries in the Crucible Framework.

  ## Examples

      iex> components = CrucibleFramework.components()
      iex> Enum.member?(components, :ensemble)
      true
  """
  @spec components() :: [atom()]
  def components do
    [
      :ensemble,
      :hedging,
      :bench,
      :telemetry_research,
      :dataset_manager,
      :causal_trace,
      :research_harness,
      :reporter
    ]
  end

  @doc """
  Returns the status of a component library.

  Status can be:
  - `:available` - Component is loaded and available
  - `:not_loaded` - Component is not currently loaded
  - `:unknown` - Component is not a valid Crucible component

  ## Examples

      iex> CrucibleFramework.component_status(:ensemble)
      :not_loaded

      iex> CrucibleFramework.component_status(:invalid)
      :unknown
  """
  @spec component_status(atom()) :: :available | :not_loaded | :unknown
  def component_status(component) when is_atom(component) do
    if component in components() do
      case Code.ensure_loaded?(component |> to_module_name()) do
        true -> :available
        false -> :not_loaded
      end
    else
      :unknown
    end
  end

  @doc """
  Returns information about the framework and its current state.

  ## Examples

      iex> info = CrucibleFramework.info()
      iex> Map.has_key?(info, :version)
      true
  """
  @spec info() :: map()
  def info do
    %{
      version: version(),
      components: components(),
      loaded_components: loaded_components(),
      elixir_version: System.version(),
      otp_release: System.otp_release()
    }
  end

  @doc """
  Returns a list of currently loaded components.

  ## Examples

      iex> CrucibleFramework.loaded_components()
      []
  """
  @spec loaded_components() :: [atom()]
  def loaded_components do
    components()
    |> Enum.filter(fn component ->
      component_status(component) == :available
    end)
  end

  # Private helper to convert component atom to module name
  defp to_module_name(component) do
    component
    |> Atom.to_string()
    |> Macro.camelize()
    |> String.to_atom()
  end
end
