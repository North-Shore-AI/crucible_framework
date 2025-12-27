defmodule Crucible.Stage.ConformanceTest do
  @moduledoc """
  Conformance tests for all framework stages.

  These tests verify that all stages implementing the `Crucible.Stage` behaviour
  have proper `describe/1` implementations that conform to the canonical schema.
  """
  use ExUnit.Case, async: true

  alias Crucible.Stage.Schema

  @framework_stages [
    Crucible.Stage.Bench,
    Crucible.Stage.Report,
    Crucible.Stage.Guardrails,
    Crucible.Stage.DataChecks,
    Crucible.Stage.Validate
  ]

  describe "framework stages existence" do
    for stage <- @framework_stages do
      @stage stage

      test "#{inspect(@stage)} exports describe/1" do
        Code.ensure_loaded!(@stage)

        assert function_exported?(@stage, :describe, 1),
               "Stage #{inspect(@stage)} must implement describe/1"
      end

      test "#{inspect(@stage)} exports run/2" do
        Code.ensure_loaded!(@stage)

        assert function_exported?(@stage, :run, 2),
               "Stage #{inspect(@stage)} must implement run/2"
      end
    end
  end

  describe "schema structure" do
    for stage <- @framework_stages do
      @stage stage

      test "#{inspect(@stage)} returns valid schema structure" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        assert is_map(schema), "describe/1 must return a map"
        assert is_atom(schema[:name]), "name must be an atom"
        assert is_binary(schema[:description]), "description must be a string"
        assert byte_size(schema[:description]) > 0, "description must not be empty"
        assert is_list(schema[:required]), "required must be a list"
        assert is_list(schema[:optional]), "optional must be a list"
        assert is_map(schema[:types]), "types must be a map"
      end

      test "#{inspect(@stage)} schema validates with Schema.validate/1" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})
        assert :ok = Schema.validate(schema)
      end
    end
  end

  describe "type coherence" do
    for stage <- @framework_stages do
      @stage stage

      test "#{inspect(@stage)} has no overlap between required and optional" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        required = MapSet.new(schema[:required] || [])
        optional = MapSet.new(schema[:optional] || [])
        overlap = MapSet.intersection(required, optional)

        assert MapSet.size(overlap) == 0,
               "Keys #{inspect(MapSet.to_list(overlap))} appear in both required and optional"
      end

      test "#{inspect(@stage)} has all required fields in types" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})
        types = schema[:types] || %{}

        for key <- schema[:required] || [] do
          assert Map.has_key?(types, key),
                 "Required field :#{key} must have type spec in types"
        end
      end

      test "#{inspect(@stage)} has valid type specs" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        for {key, type_spec} <- schema[:types] || %{} do
          assert Schema.valid_type_spec?(type_spec),
                 "Invalid type spec for :#{key}: #{inspect(type_spec)}"
        end
      end

      test "#{inspect(@stage)} defaults are only for optional fields" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        if defaults = schema[:defaults] do
          optional = MapSet.new(schema[:optional] || [])

          for key <- Map.keys(defaults) do
            assert key in optional,
                   "Default for :#{key} but :#{key} not in optional"
          end
        end
      end
    end
  end

  describe "required/optional field types are atoms" do
    for stage <- @framework_stages do
      @stage stage

      test "#{inspect(@stage)} has atom keys in required list" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        for key <- schema[:required] || [] do
          assert is_atom(key), "Required key #{inspect(key)} must be an atom"
        end
      end

      test "#{inspect(@stage)} has atom keys in optional list" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        for key <- schema[:optional] || [] do
          assert is_atom(key), "Optional key #{inspect(key)} must be an atom"
        end
      end
    end
  end

  describe "stage name consistency" do
    for stage <- @framework_stages do
      @stage stage

      test "#{inspect(@stage)} schema name matches expected pattern" do
        Code.ensure_loaded!(@stage)
        schema = @stage.describe(%{})

        assert is_atom(schema[:name])
        # Name should be a valid atom (lowercase, underscored)
        name_str = Atom.to_string(schema[:name])

        assert name_str =~ ~r/^[a-z][a-z0-9_]*$/,
               "Stage name #{inspect(schema[:name])} should follow snake_case convention"
      end
    end
  end
end
