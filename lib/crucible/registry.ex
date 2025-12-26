defmodule Crucible.Registry do
  @moduledoc """
  Resolves stage modules from application configuration.

  Stages can be resolved either by:
  1. Explicit module specification in the `StageDef` struct
  2. Name-based lookup from the configured stage registry

  ## Configuration

  Configure the stage registry in your application config:

      config :crucible_framework,
        stage_registry: %{
          validate: Crucible.Stage.Validate,
          bench: Crucible.Stage.Bench,
          report: Crucible.Stage.Report,
          # Add custom stages here
          my_stage: MyApp.Stage.Custom
        }
  """

  @doc """
  Resolves a stage module by name from the configured registry.

  ## Examples

      {:ok, Crucible.Stage.Bench} = Crucible.Registry.stage_module(:bench)
      {:error, {:unknown_stage, :missing}} = Crucible.Registry.stage_module(:missing)
  """
  @spec stage_module(atom()) :: {:ok, module()} | {:error, term()}
  def stage_module(name) when is_atom(name) do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, map} ->
        case Map.fetch(map, name) do
          {:ok, mod} -> {:ok, mod}
          :error -> {:error, {:unknown_stage, name}}
        end

      :error ->
        {:error, :no_stage_registry}
    end
  end
end
