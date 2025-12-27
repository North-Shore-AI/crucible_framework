defmodule Crucible.Stage.Schema.Normalizer do
  @moduledoc """
  Normalizes legacy `describe/1` schemas to canonical format.

  This module handles conversion of various legacy schema formats used across
  the Crucible ecosystem to the canonical format defined in `Crucible.Stage.Schema`.

  ## Normalization Rules

  1. **Name normalization**: Converts `:stage` key to `:name`, string names to atoms
  2. **Required fields**: Adds missing `required`, `optional`, and `types` fields
  3. **Extensions**: Moves non-core fields to `__extensions__` map

  ## Core Fields

  These fields are preserved in the top-level schema:

  - `:__schema_version__`
  - `:name`
  - `:description`
  - `:required`
  - `:optional`
  - `:types`
  - `:defaults`
  - `:version`
  - `:__extensions__`

  All other fields are moved to `__extensions__`.

  ## Example

      # Input (legacy format)
      %{
        name: "ensemble_voting",
        description: "...",
        strategies: [:majority, :weighted]
      }

      # Output (canonical)
      %{
        name: :ensemble_voting,
        description: "...",
        required: [],
        optional: [],
        types: %{},
        __extensions__: %{
          strategies: [:majority, :weighted]
        }
      }
  """

  @core_keys [
    :__schema_version__,
    :name,
    :description,
    :required,
    :optional,
    :types,
    :defaults,
    :version,
    :__extensions__
  ]

  @doc """
  Normalizes a describe/1 result to canonical schema format.

  Handles:
  - String names → atoms
  - `:stage` key → `:name`
  - Missing required/optional/types fields
  - Non-core fields → `__extensions__`

  ## Examples

      iex> Crucible.Stage.Schema.Normalizer.normalize(%{name: "test", description: "Test"})
      %{name: :test, description: "Test", required: [], optional: [], types: %{}}

      iex> Crucible.Stage.Schema.Normalizer.normalize(%{stage: :my_stage, description: "My stage"})
      %{name: :my_stage, description: "My stage", required: [], optional: [], types: %{}}
  """
  @spec normalize(map()) :: Crucible.Stage.Schema.t()
  def normalize(schema) when is_map(schema) do
    schema
    |> normalize_name()
    |> ensure_required_fields()
    |> extract_extensions()
  end

  # Handle :stage key -> :name (with atom value)
  defp normalize_name(%{stage: stage} = schema) when is_atom(stage) do
    schema
    |> Map.delete(:stage)
    |> Map.put(:name, stage)
  end

  # Handle :stage key -> :name (with string value)
  defp normalize_name(%{stage: stage} = schema) when is_binary(stage) do
    schema
    |> Map.delete(:stage)
    |> Map.put(:name, String.to_atom(stage))
  end

  # Handle string name -> atom
  defp normalize_name(%{name: name} = schema) when is_binary(name) do
    %{schema | name: String.to_atom(name)}
  end

  defp normalize_name(schema), do: schema

  # Ensure required/optional/types exist with defaults
  defp ensure_required_fields(schema) do
    schema
    |> Map.put_new(:required, [])
    |> Map.put_new(:optional, [])
    |> Map.put_new(:types, %{})
  end

  # Move non-core fields to __extensions__
  defp extract_extensions(schema) do
    {core, extras} = Map.split(schema, @core_keys)

    existing_extensions = Map.get(core, :__extensions__, %{})
    all_extensions = Map.merge(existing_extensions, extras)

    if map_size(all_extensions) > 0 do
      Map.put(core, :__extensions__, all_extensions)
    else
      core
    end
  end
end
