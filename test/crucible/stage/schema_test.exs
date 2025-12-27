defmodule Crucible.Stage.SchemaTest do
  use ExUnit.Case, async: true

  alias Crucible.Stage.Schema

  describe "validate/1" do
    test "returns :ok for valid minimal schema" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      assert :ok = Schema.validate(schema)
    end

    test "returns :ok for valid schema with all fields" do
      schema = %{
        __schema_version__: "1.0.0",
        name: :test_stage,
        description: "A test stage",
        required: [:input],
        optional: [:verbose, :timeout],
        types: %{
          input: :string,
          verbose: :boolean,
          timeout: :integer
        },
        defaults: %{
          verbose: false,
          timeout: 5000
        },
        version: "0.1.0",
        __extensions__: %{
          custom: %{foo: :bar}
        }
      }

      assert :ok = Schema.validate(schema)
    end

    test "returns error when name is missing" do
      schema = %{
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "name"))
    end

    test "returns error when name is not an atom" do
      schema = %{
        name: "test_stage",
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "name"))
    end

    test "returns error when description is missing" do
      schema = %{
        name: :test_stage,
        required: [],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "description"))
    end

    test "returns error when description is empty" do
      schema = %{
        name: :test_stage,
        description: "",
        required: [],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "description"))
    end

    test "returns error when required is not a list" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: :atom,
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "required"))
    end

    test "returns error when optional is not a list" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: "not a list",
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "optional"))
    end

    test "returns error when types is not a map" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: []
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "types"))
    end

    test "returns error when required field is not an atom" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: ["input"],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "required"))
    end

    test "returns error when optional field is not an atom" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [123],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "optional"))
    end

    test "returns error when required field overlaps with optional" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [:input],
        optional: [:input, :verbose],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "overlap"))
    end

    test "returns error when required field has no type spec" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [:input],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "input"))
    end

    test "returns error when defaults key is not in optional" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [:verbose],
        types: %{verbose: :boolean},
        defaults: %{unknown_key: true}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "unknown_key"))
    end

    test "accepts valid __schema_version__" do
      schema = %{
        __schema_version__: "1.0.0",
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      assert :ok = Schema.validate(schema)
    end

    test "returns error for invalid __schema_version__" do
      schema = %{
        __schema_version__: 1.0,
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      assert {:error, errors} = Schema.validate(schema)
      assert Enum.any?(errors, &String.contains?(&1, "__schema_version__"))
    end
  end

  describe "valid_type_spec?/1" do
    test "validates primitive types" do
      assert Schema.valid_type_spec?(:string)
      assert Schema.valid_type_spec?(:integer)
      assert Schema.valid_type_spec?(:float)
      assert Schema.valid_type_spec?(:boolean)
      assert Schema.valid_type_spec?(:atom)
      assert Schema.valid_type_spec?(:map)
      assert Schema.valid_type_spec?(:list)
      assert Schema.valid_type_spec?(:module)
      assert Schema.valid_type_spec?(:any)
    end

    test "validates struct type" do
      assert Schema.valid_type_spec?({:struct, SomeModule})
      refute Schema.valid_type_spec?({:struct, "not_a_module"})
    end

    test "validates enum type" do
      assert Schema.valid_type_spec?({:enum, [:a, :b, :c]})
      refute Schema.valid_type_spec?({:enum, :not_a_list})
    end

    test "validates list type" do
      assert Schema.valid_type_spec?({:list, :string})
      assert Schema.valid_type_spec?({:list, {:enum, [:a, :b]}})
      refute Schema.valid_type_spec?({:list, :invalid_type})
    end

    test "validates map type" do
      assert Schema.valid_type_spec?({:map, :atom, :string})
      assert Schema.valid_type_spec?({:map, :string, {:list, :integer}})
      refute Schema.valid_type_spec?({:map, :invalid, :string})
    end

    test "validates function type" do
      assert Schema.valid_type_spec?({:function, 0})
      assert Schema.valid_type_spec?({:function, 1})
      assert Schema.valid_type_spec?({:function, 2})
      refute Schema.valid_type_spec?({:function, -1})
      refute Schema.valid_type_spec?({:function, "not_an_integer"})
    end

    test "validates union type" do
      assert Schema.valid_type_spec?({:union, [:string, :integer]})
      assert Schema.valid_type_spec?({:union, [:float, {:enum, [:auto]}]})
      refute Schema.valid_type_spec?({:union, [:invalid_type]})
    end

    test "validates tuple type" do
      assert Schema.valid_type_spec?({:tuple, [:atom, :string]})
      assert Schema.valid_type_spec?({:tuple, [{:enum, [:ok, :error]}, :any]})
      refute Schema.valid_type_spec?({:tuple, [:invalid_type]})
    end

    test "rejects invalid types" do
      refute Schema.valid_type_spec?(:invalid_type)
      refute Schema.valid_type_spec?({:unknown, :something})
      refute Schema.valid_type_spec?("string")
      refute Schema.valid_type_spec?(123)
    end
  end
end
