defmodule Crucible.Stage.Validator do
  @moduledoc """
  Validates stage options against describe/1 schema.

  This module provides runtime validation of stage options against the schema
  returned by a stage's `describe/1` callback. It checks that:

  1. All required options are present
  2. All provided option values match their type specifications

  ## Usage

      schema = MyStage.describe(%{})
      opts = %{input: "data.csv", batch_size: 100}

      case Crucible.Stage.Validator.validate(opts, schema) do
        :ok -> # proceed with stage execution
        {:error, errors} -> # handle validation errors
      end

  ## Type Validation

  The validator supports all type specifications defined in `Crucible.Stage.Schema`:

  - Primitives: `:string`, `:integer`, `:float`, `:boolean`, `:atom`, `:map`, `:list`, `:module`, `:any`
  - Structs: `{:struct, Module}`
  - Enums: `{:enum, [values]}`
  - Typed lists: `{:list, inner_type}`
  - Typed maps: `{:map, key_type, value_type}`
  - Functions: `{:function, arity}`
  - Unions: `{:union, [types]}`
  - Tuples: `{:tuple, [types]}`
  """

  alias Crucible.Stage.Schema.Normalizer

  @doc """
  Validates options map against a stage's schema.

  The schema is first normalized using `Crucible.Stage.Schema.Normalizer` to handle
  legacy formats. Options can be provided as a map, keyword list, or nil.

  Returns `:ok` if validation passes, or `{:error, [error_messages]}` with a list
  of human-readable error messages if validation fails.

  ## Examples

      iex> schema = %{name: :test, description: "Test", required: [:input], optional: [], types: %{input: :string}}
      iex> Crucible.Stage.Validator.validate(%{input: "hello"}, schema)
      :ok

      iex> schema = %{name: :test, description: "Test", required: [:input], optional: [], types: %{input: :string}}
      iex> Crucible.Stage.Validator.validate(%{}, schema)
      {:error, ["missing required option: input"]}
  """
  @spec validate(map() | keyword() | nil, map()) :: :ok | {:error, [String.t()]}
  def validate(opts, schema) do
    schema = Normalizer.normalize(schema)
    opts = normalize_opts(opts)

    errors =
      []
      |> check_required(opts, schema.required)
      |> check_types(opts, schema.types)

    case errors do
      [] -> :ok
      errors -> {:error, Enum.reverse(errors)}
    end
  end

  defp normalize_opts(nil), do: %{}
  defp normalize_opts(opts) when is_list(opts), do: Map.new(opts)
  defp normalize_opts(opts) when is_map(opts), do: opts

  defp check_required(errors, opts, required) do
    Enum.reduce(required, errors, fn key, acc ->
      if Map.has_key?(opts, key) do
        acc
      else
        ["missing required option: #{key}" | acc]
      end
    end)
  end

  defp check_types(errors, opts, types) do
    Enum.reduce(opts, errors, fn {key, value}, acc ->
      case Map.get(types, key) do
        nil -> acc
        type_spec -> validate_type(key, value, type_spec, acc)
      end
    end)
  end

  # Primitive types
  defp validate_type(_key, value, :string, errors) when is_binary(value), do: errors

  defp validate_type(key, value, :string, errors) do
    ["#{key}: expected string, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :integer, errors) when is_integer(value), do: errors

  defp validate_type(key, value, :integer, errors) do
    ["#{key}: expected integer, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :float, errors) when is_float(value), do: errors

  defp validate_type(key, value, :float, errors) do
    ["#{key}: expected float, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :boolean, errors) when is_boolean(value), do: errors

  defp validate_type(key, value, :boolean, errors) do
    ["#{key}: expected boolean, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :atom, errors) when is_atom(value), do: errors

  defp validate_type(key, value, :atom, errors) do
    ["#{key}: expected atom, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :map, errors) when is_map(value), do: errors

  defp validate_type(key, value, :map, errors) do
    ["#{key}: expected map, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :list, errors) when is_list(value), do: errors

  defp validate_type(key, value, :list, errors) do
    ["#{key}: expected list, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, :module, errors) when is_atom(value), do: errors

  defp validate_type(key, value, :module, errors) do
    ["#{key}: expected module (atom), got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, _value, :any, errors), do: errors

  # Composite types

  defp validate_type(key, value, {:struct, mod}, errors) do
    if is_struct(value, mod) do
      errors
    else
      ["#{key}: expected %#{inspect(mod)}{}, got #{inspect(value)}" | errors]
    end
  end

  defp validate_type(key, value, {:enum, values}, errors) do
    if value in values do
      errors
    else
      ["#{key}: expected one of #{inspect(values)}, got #{inspect(value)}" | errors]
    end
  end

  defp validate_type(key, value, {:list, inner_type}, errors) when is_list(value) do
    Enum.reduce(Enum.with_index(value), errors, fn {item, idx}, acc ->
      case validate_single_value(item, inner_type) do
        :ok -> acc
        {:error, msg} -> ["#{key}[#{idx}]: #{msg}" | acc]
      end
    end)
  end

  defp validate_type(key, value, {:list, _inner_type}, errors) do
    ["#{key}: expected list, got #{inspect(value)}" | errors]
  end

  defp validate_type(_key, value, {:function, arity}, errors)
       when is_function(value, arity) do
    errors
  end

  defp validate_type(key, value, {:function, arity}, errors) when is_function(value) do
    {:arity, actual} = Function.info(value, :arity)
    ["#{key}: expected function with arity #{arity}, got arity #{actual}" | errors]
  end

  defp validate_type(key, value, {:function, arity}, errors) do
    ["#{key}: expected function with arity #{arity}, got #{inspect(value)}" | errors]
  end

  defp validate_type(key, value, {:union, types}, errors) do
    if Enum.any?(types, fn t -> validate_single_value(value, t) == :ok end) do
      errors
    else
      [
        "#{key}: value #{inspect(value)} doesn't match any type in union #{inspect(types)}"
        | errors
      ]
    end
  end

  defp validate_type(key, value, {:map, key_type, value_type}, errors) when is_map(value) do
    Enum.reduce(value, errors, fn {k, v}, acc ->
      acc =
        case validate_single_value(k, key_type) do
          :ok -> acc
          {:error, msg} -> ["#{key} key: #{msg}" | acc]
        end

      case validate_single_value(v, value_type) do
        :ok -> acc
        {:error, msg} -> ["#{key}[#{inspect(k)}]: #{msg}" | acc]
      end
    end)
  end

  defp validate_type(key, value, {:map, _key_type, _value_type}, errors) do
    ["#{key}: expected map, got #{inspect(value)}" | errors]
  end

  defp validate_type(key, value, {:tuple, types}, errors) when is_tuple(value) do
    tuple_list = Tuple.to_list(value)
    validate_tuple_elements(key, tuple_list, types, errors)
  end

  defp validate_type(key, value, {:tuple, types}, errors) do
    ["#{key}: expected tuple of size #{length(types)}, got #{inspect(value)}" | errors]
  end

  # Fallback for unknown type spec
  defp validate_type(key, _value, type_spec, errors) do
    ["#{key}: unknown type spec #{inspect(type_spec)}" | errors]
  end

  # Helper functions for tuple validation
  defp validate_tuple_elements(key, tuple_list, types, errors)
       when length(tuple_list) != length(types) do
    ["#{key}: expected tuple of size #{length(types)}, got size #{length(tuple_list)}" | errors]
  end

  defp validate_tuple_elements(key, tuple_list, types, errors) do
    tuple_list
    |> Enum.zip(types)
    |> Enum.with_index()
    |> Enum.reduce(errors, &validate_tuple_element(key, &1, &2))
  end

  defp validate_tuple_element(key, {{elem, type}, idx}, acc) do
    case validate_single_value(elem, type) do
      :ok -> acc
      {:error, msg} -> ["#{key}[#{idx}]: #{msg}" | acc]
    end
  end

  # Helper to validate a single value and return :ok or {:error, message}
  defp validate_single_value(value, type_spec) do
    case validate_type(:value, value, type_spec, []) do
      [] -> :ok
      [error | _] -> {:error, String.replace(error, "value: ", "")}
    end
  end
end
