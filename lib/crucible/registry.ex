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

  ## Schema Access

  The registry provides functions to access stage schemas:

      Crucible.Registry.list_stages_with_schemas()
      # => [{:bench, Crucible.Stage.Bench, %{name: :bench, ...}}, ...]

      Crucible.Registry.stage_schema(:bench)
      # => {:ok, %{name: :bench, description: "...", ...}}
  """

  alias Crucible.Stage.Schema.Normalizer

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

  @doc """
  Lists all registered stages with their modules and schemas.

  Returns a list of tuples `{name, module, schema}` for each registered stage.
  Schemas are normalized to canonical format.

  ## Examples

      Crucible.Registry.list_stages_with_schemas()
      # => [
      #      {:bench, Crucible.Stage.Bench, %{name: :bench, ...}},
      #      {:validate, Crucible.Stage.Validate, %{name: :validate, ...}}
      #    ]
  """
  @spec list_stages_with_schemas() :: [{atom(), module(), map()}]
  def list_stages_with_schemas do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, registry} ->
        registry
        |> Enum.map(fn {name, mod} ->
          schema = get_stage_schema(mod)
          {name, mod, schema}
        end)
        |> Enum.sort_by(&elem(&1, 0))

      :error ->
        []
    end
  end

  @doc """
  Gets the schema for a specific registered stage by name.

  Returns `{:ok, schema}` with the normalized schema, or
  `{:error, reason}` if the stage is not found or has no schema.

  ## Examples

      {:ok, schema} = Crucible.Registry.stage_schema(:bench)
      schema.name
      # => :bench

      {:error, {:unknown_stage, :missing}} = Crucible.Registry.stage_schema(:missing)
  """
  @spec stage_schema(atom()) ::
          {:ok, Crucible.Stage.Schema.t()}
          | {:error,
             :no_stage_registry
             | {:no_describe_callback, atom()}
             | {:unknown_stage, atom()}}
  def stage_schema(name) when is_atom(name) do
    case stage_module(name) do
      {:ok, mod} ->
        Code.ensure_loaded!(mod)

        if function_exported?(mod, :describe, 1) do
          schema = mod.describe(%{}) |> Normalizer.normalize()
          {:ok, schema}
        else
          {:error, {:no_describe_callback, mod}}
        end

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Lists all registered stage names.

  ## Examples

      Crucible.Registry.list_stages()
      # => [:bench, :data_checks, :guardrails, :report, :validate]
  """
  @spec list_stages() :: [atom()]
  def list_stages do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, registry} ->
        registry |> Map.keys() |> Enum.sort()

      :error ->
        []
    end
  end

  # Private helpers

  defp get_stage_schema(mod) do
    Code.ensure_loaded!(mod)

    if function_exported?(mod, :describe, 1) do
      mod.describe(%{}) |> Normalizer.normalize()
    else
      %{name: nil, description: "No describe/1 callback", required: [], optional: [], types: %{}}
    end
  end
end
