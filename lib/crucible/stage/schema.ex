defmodule Crucible.Stage.Schema do
  @moduledoc """
  Canonical schema definition for stage `describe/1` callback.

  This module defines the structure and validation rules for stage schemas.
  All stages implementing the `Crucible.Stage` behaviour must return a schema
  conforming to this specification from their `describe/1` callback.

  ## Schema Structure

  A valid schema must contain:

  - `:name` - Stage identifier (atom, required)
  - `:description` - Human-readable description (non-empty string, required)
  - `:required` - List of required option keys (list of atoms, required)
  - `:optional` - List of optional option keys (list of atoms, required)
  - `:types` - Map of option keys to type specifications (map, required)

  Optional fields:

  - `:__schema_version__` - Schema version (semver string, e.g., "1.0.0")
  - `:defaults` - Default values for optional fields (map)
  - `:version` - Stage version (semver string)
  - `:__extensions__` - Domain-specific metadata (map)

  ## Type Specifications

  Supported type specs:

  - Primitives: `:string`, `:integer`, `:float`, `:boolean`, `:atom`, `:map`, `:list`, `:module`, `:any`
  - Struct: `{:struct, Module}`
  - Enum: `{:enum, [values]}`
  - Typed list: `{:list, inner_type}`
  - Typed map: `{:map, key_type, value_type}`
  - Function: `{:function, arity}`
  - Union: `{:union, [types]}`
  - Tuple: `{:tuple, [types]}`

  ## Example

      %{
        __schema_version__: "1.0.0",
        name: :my_stage,
        description: "Processes data according to configuration",
        required: [:input_path],
        optional: [:format, :batch_size],
        types: %{
          input_path: :string,
          format: {:enum, [:json, :csv]},
          batch_size: :integer
        },
        defaults: %{
          format: :json,
          batch_size: 100
        }
      }
  """

  @primitive_types [:string, :integer, :float, :boolean, :atom, :map, :list, :module, :any]

  @typedoc """
  Type specification for stage options.
  """
  @type type_spec ::
          :string
          | :integer
          | :float
          | :boolean
          | :atom
          | :map
          | :list
          | :module
          | :any
          | {:struct, module()}
          | {:enum, [term()]}
          | {:list, type_spec()}
          | {:map, type_spec(), type_spec()}
          | {:function, non_neg_integer()}
          | {:union, [type_spec()]}
          | {:tuple, [type_spec()]}

  @typedoc """
  Canonical schema format for stage describe/1 callback.
  """
  @type t :: %{
          :name => atom(),
          :description => String.t(),
          :required => [atom()],
          :optional => [atom()],
          :types => %{optional(atom()) => type_spec()},
          optional(:__schema_version__) => String.t(),
          optional(:defaults) => %{optional(atom()) => term()},
          optional(:version) => String.t(),
          optional(:__extensions__) => map()
        }

  @doc """
  Validates a schema map for conformance to the canonical format.

  Returns `:ok` if the schema is valid, or `{:error, [String.t()]}` with
  a list of error messages if validation fails.

  ## Examples

      iex> Crucible.Stage.Schema.validate(%{
      ...>   name: :test,
      ...>   description: "Test stage",
      ...>   required: [],
      ...>   optional: [],
      ...>   types: %{}
      ...> })
      :ok

      iex> Crucible.Stage.Schema.validate(%{description: "Missing name"})
      {:error, ["name is required and must be an atom"]}
  """
  @spec validate(map()) :: :ok | {:error, [String.t()]}
  def validate(schema) when is_map(schema) do
    errors =
      []
      |> validate_name(schema)
      |> validate_description(schema)
      |> validate_required(schema)
      |> validate_optional(schema)
      |> validate_types(schema)
      |> validate_no_overlap(schema)
      |> validate_required_have_types(schema)
      |> validate_defaults(schema)
      |> validate_schema_version(schema)

    case errors do
      [] -> :ok
      errors -> {:error, Enum.reverse(errors)}
    end
  end

  def validate(_), do: {:error, ["schema must be a map"]}

  @doc """
  Checks if a type specification is valid.

  ## Examples

      iex> Crucible.Stage.Schema.valid_type_spec?(:string)
      true

      iex> Crucible.Stage.Schema.valid_type_spec?({:enum, [:a, :b]})
      true

      iex> Crucible.Stage.Schema.valid_type_spec?(:invalid)
      false
  """
  @spec valid_type_spec?(term()) :: boolean()
  def valid_type_spec?(spec) when spec in @primitive_types, do: true

  def valid_type_spec?({:struct, mod}) when is_atom(mod), do: true

  def valid_type_spec?({:enum, values}) when is_list(values), do: true

  def valid_type_spec?({:list, inner}), do: valid_type_spec?(inner)

  def valid_type_spec?({:map, key_type, value_type}) do
    valid_type_spec?(key_type) and valid_type_spec?(value_type)
  end

  def valid_type_spec?({:function, arity}) when is_integer(arity) and arity >= 0, do: true

  def valid_type_spec?({:union, types}) when is_list(types) do
    Enum.all?(types, &valid_type_spec?/1)
  end

  def valid_type_spec?({:tuple, types}) when is_list(types) do
    Enum.all?(types, &valid_type_spec?/1)
  end

  def valid_type_spec?(_), do: false

  # Private validation functions

  defp validate_name(errors, %{name: name}) when is_atom(name), do: errors

  defp validate_name(errors, _schema) do
    ["name is required and must be an atom" | errors]
  end

  defp validate_description(errors, %{description: desc})
       when is_binary(desc) and byte_size(desc) > 0 do
    errors
  end

  defp validate_description(errors, _schema) do
    ["description is required and must be a non-empty string" | errors]
  end

  defp validate_required(errors, %{required: required}) when is_list(required) do
    if Enum.all?(required, &is_atom/1) do
      errors
    else
      ["required must be a list of atoms" | errors]
    end
  end

  defp validate_required(errors, _schema) do
    ["required is required and must be a list" | errors]
  end

  defp validate_optional(errors, %{optional: optional}) when is_list(optional) do
    if Enum.all?(optional, &is_atom/1) do
      errors
    else
      ["optional must be a list of atoms" | errors]
    end
  end

  defp validate_optional(errors, _schema) do
    ["optional is required and must be a list" | errors]
  end

  defp validate_types(errors, %{types: types}) when is_map(types) do
    invalid_specs =
      Enum.filter(types, fn {_key, spec} -> not valid_type_spec?(spec) end)

    case invalid_specs do
      [] ->
        errors

      specs ->
        keys = Enum.map(specs, fn {k, _} -> k end)
        ["invalid type specs for keys: #{inspect(keys)}" | errors]
    end
  end

  defp validate_types(errors, _schema) do
    ["types is required and must be a map" | errors]
  end

  defp validate_no_overlap(errors, %{required: required, optional: optional})
       when is_list(required) and is_list(optional) do
    overlap = MapSet.intersection(MapSet.new(required), MapSet.new(optional))

    if MapSet.size(overlap) == 0 do
      errors
    else
      ["keys overlap between required and optional: #{inspect(MapSet.to_list(overlap))}" | errors]
    end
  end

  defp validate_no_overlap(errors, _schema), do: errors

  defp validate_required_have_types(errors, %{required: required, types: types})
       when is_list(required) and is_map(types) do
    missing = Enum.filter(required, fn key -> not Map.has_key?(types, key) end)

    case missing do
      [] -> errors
      keys -> ["required keys missing from types: #{inspect(keys)}" | errors]
    end
  end

  defp validate_required_have_types(errors, _schema), do: errors

  defp validate_defaults(errors, %{defaults: defaults, optional: optional})
       when is_map(defaults) and is_list(optional) do
    optional_set = MapSet.new(optional)
    invalid_keys = Enum.filter(Map.keys(defaults), fn key -> key not in optional_set end)

    case invalid_keys do
      [] -> errors
      keys -> ["defaults keys must be in optional: #{inspect(keys)}" | errors]
    end
  end

  defp validate_defaults(errors, %{defaults: defaults}) when not is_map(defaults) do
    ["defaults must be a map" | errors]
  end

  defp validate_defaults(errors, _schema), do: errors

  defp validate_schema_version(errors, %{__schema_version__: version}) when is_binary(version) do
    errors
  end

  defp validate_schema_version(errors, %{__schema_version__: _version}) do
    ["__schema_version__ must be a string (semver format)" | errors]
  end

  defp validate_schema_version(errors, _schema), do: errors
end
