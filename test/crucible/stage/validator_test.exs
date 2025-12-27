defmodule Crucible.Stage.ValidatorTest do
  use ExUnit.Case, async: true

  alias Crucible.Stage.Validator

  describe "validate/2" do
    test "returns :ok for valid options matching schema" do
      schema = %{
        name: :test,
        description: "Test",
        required: [:input],
        optional: [:verbose],
        types: %{input: :string, verbose: :boolean}
      }

      opts = %{input: "hello", verbose: true}

      assert :ok = Validator.validate(opts, schema)
    end

    test "returns :ok for empty options when no required fields" do
      schema = %{
        name: :test,
        description: "Test",
        required: [],
        optional: [:verbose],
        types: %{verbose: :boolean}
      }

      assert :ok = Validator.validate(%{}, schema)
    end

    test "returns :ok for nil options when no required fields" do
      schema = %{
        name: :test,
        description: "Test",
        required: [],
        optional: [],
        types: %{}
      }

      assert :ok = Validator.validate(nil, schema)
    end

    test "returns :ok for keyword list options" do
      schema = %{
        name: :test,
        description: "Test",
        required: [:input],
        optional: [],
        types: %{input: :string}
      }

      opts = [input: "hello"]

      assert :ok = Validator.validate(opts, schema)
    end

    test "returns error for missing required option" do
      schema = %{
        name: :test,
        description: "Test",
        required: [:input, :output],
        optional: [],
        types: %{input: :string, output: :string}
      }

      opts = %{input: "hello"}

      assert {:error, errors} = Validator.validate(opts, schema)
      assert Enum.any?(errors, &String.contains?(&1, "output"))
    end

    test "returns error for multiple missing required options" do
      schema = %{
        name: :test,
        description: "Test",
        required: [:input, :output],
        optional: [],
        types: %{input: :string, output: :string}
      }

      opts = %{}

      assert {:error, errors} = Validator.validate(opts, schema)
      assert Enum.any?(errors, &String.contains?(&1, "input"))
      assert Enum.any?(errors, &String.contains?(&1, "output"))
    end
  end

  describe "primitive type validation" do
    test "validates :string type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :string}
      }

      assert :ok = Validator.validate(%{x: "hello"}, schema)
      assert {:error, _} = Validator.validate(%{x: 123}, schema)
      assert {:error, _} = Validator.validate(%{x: :atom}, schema)
    end

    test "validates :integer type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :integer}
      }

      assert :ok = Validator.validate(%{x: 42}, schema)
      assert :ok = Validator.validate(%{x: -1}, schema)
      assert {:error, _} = Validator.validate(%{x: 3.14}, schema)
      assert {:error, _} = Validator.validate(%{x: "42"}, schema)
    end

    test "validates :float type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :float}
      }

      assert :ok = Validator.validate(%{x: 3.14}, schema)
      assert :ok = Validator.validate(%{x: -1.5}, schema)
      assert {:error, _} = Validator.validate(%{x: 42}, schema)
      assert {:error, _} = Validator.validate(%{x: "3.14"}, schema)
    end

    test "validates :boolean type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :boolean}
      }

      assert :ok = Validator.validate(%{x: true}, schema)
      assert :ok = Validator.validate(%{x: false}, schema)
      assert {:error, _} = Validator.validate(%{x: "true"}, schema)
      assert {:error, _} = Validator.validate(%{x: 1}, schema)
    end

    test "validates :atom type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :atom}
      }

      assert :ok = Validator.validate(%{x: :foo}, schema)
      assert :ok = Validator.validate(%{x: nil}, schema)
      assert {:error, _} = Validator.validate(%{x: "foo"}, schema)
    end

    test "validates :map type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :map}
      }

      assert :ok = Validator.validate(%{x: %{}}, schema)
      assert :ok = Validator.validate(%{x: %{a: 1}}, schema)
      assert {:error, _} = Validator.validate(%{x: []}, schema)
    end

    test "validates :list type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :list}
      }

      assert :ok = Validator.validate(%{x: []}, schema)
      assert :ok = Validator.validate(%{x: [1, 2, 3]}, schema)
      assert {:error, _} = Validator.validate(%{x: %{}}, schema)
    end

    test "validates :module type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :module}
      }

      assert :ok = Validator.validate(%{x: String}, schema)
      assert :ok = Validator.validate(%{x: Enum}, schema)
      # Atoms that are not loaded modules still pass (we don't verify loadability)
      assert :ok = Validator.validate(%{x: SomeUnknownModule}, schema)
      assert {:error, _} = Validator.validate(%{x: "String"}, schema)
    end

    test "validates :any type accepts anything" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: :any}
      }

      assert :ok = Validator.validate(%{x: "string"}, schema)
      assert :ok = Validator.validate(%{x: 123}, schema)
      assert :ok = Validator.validate(%{x: :atom}, schema)
      assert :ok = Validator.validate(%{x: %{}}, schema)
      assert :ok = Validator.validate(%{x: fn -> :ok end}, schema)
    end
  end

  describe "composite type validation" do
    test "validates {:struct, Module} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:struct, Range}}
      }

      assert :ok = Validator.validate(%{x: 1..10}, schema)
      assert {:error, _} = Validator.validate(%{x: %{first: 1, last: 10}}, schema)
    end

    test "validates {:enum, values} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:enum, [:a, :b, :c]}}
      }

      assert :ok = Validator.validate(%{x: :a}, schema)
      assert :ok = Validator.validate(%{x: :b}, schema)
      assert :ok = Validator.validate(%{x: :c}, schema)
      assert {:error, errors} = Validator.validate(%{x: :d}, schema)
      assert Enum.any?(errors, &String.contains?(&1, "one of"))
    end

    test "validates {:list, inner_type} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:list, :integer}}
      }

      assert :ok = Validator.validate(%{x: []}, schema)
      assert :ok = Validator.validate(%{x: [1, 2, 3]}, schema)
      assert {:error, _} = Validator.validate(%{x: [1, "two", 3]}, schema)
      assert {:error, _} = Validator.validate(%{x: "not a list"}, schema)
    end

    test "validates nested list type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:list, {:enum, [:a, :b]}}}
      }

      assert :ok = Validator.validate(%{x: [:a, :b, :a]}, schema)
      assert {:error, _} = Validator.validate(%{x: [:a, :c]}, schema)
    end

    test "validates {:function, arity} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:function, 1}}
      }

      assert :ok = Validator.validate(%{x: fn x -> x end}, schema)
      assert :ok = Validator.validate(%{x: &String.upcase/1}, schema)
      assert {:error, _} = Validator.validate(%{x: fn x, y -> x + y end}, schema)
      assert {:error, _} = Validator.validate(%{x: fn -> :ok end}, schema)
    end

    test "validates {:function, 0} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:function, 0}}
      }

      assert :ok = Validator.validate(%{x: fn -> :ok end}, schema)
      assert {:error, _} = Validator.validate(%{x: fn x -> x end}, schema)
    end

    test "validates {:union, types} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:union, [:string, :integer]}}
      }

      assert :ok = Validator.validate(%{x: "hello"}, schema)
      assert :ok = Validator.validate(%{x: 42}, schema)
      assert {:error, _} = Validator.validate(%{x: 3.14}, schema)
      assert {:error, _} = Validator.validate(%{x: :atom}, schema)
    end

    test "validates complex union type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:union, [:float, {:enum, [:auto]}]}}
      }

      assert :ok = Validator.validate(%{x: 0.5}, schema)
      assert :ok = Validator.validate(%{x: :auto}, schema)
      assert {:error, _} = Validator.validate(%{x: "auto"}, schema)
    end

    test "validates {:map, key_type, value_type} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:map, :atom, :integer}}
      }

      assert :ok = Validator.validate(%{x: %{}}, schema)
      assert :ok = Validator.validate(%{x: %{a: 1, b: 2}}, schema)
      assert {:error, _} = Validator.validate(%{x: %{a: "one"}}, schema)
      assert {:error, _} = Validator.validate(%{x: %{"a" => 1}}, schema)
    end

    test "validates {:tuple, types} type" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:x],
        types: %{x: {:tuple, [:atom, :string]}}
      }

      assert :ok = Validator.validate(%{x: {:ok, "result"}}, schema)
      assert {:error, _} = Validator.validate(%{x: {"not_atom", "result"}}, schema)
      assert {:error, _} = Validator.validate(%{x: {:ok, 123}}, schema)
      assert {:error, _} = Validator.validate(%{x: {:ok}}, schema)
    end
  end

  describe "unknown keys handling" do
    test "allows unknown keys by default" do
      schema = %{
        name: :t,
        description: "",
        required: [],
        optional: [:known],
        types: %{known: :string}
      }

      assert :ok = Validator.validate(%{known: "hello", unknown: "ignored"}, schema)
    end
  end

  describe "schema normalization" do
    test "normalizes legacy schema format before validation" do
      # Schema with :stage key instead of :name
      schema = %{
        stage: :legacy_stage,
        description: "Legacy format",
        inputs: [:a, :b],
        outputs: [:c]
      }

      opts = %{some_opt: "value"}

      # Should not crash - normalizer handles missing required/optional/types
      assert :ok = Validator.validate(opts, schema)
    end
  end
end
