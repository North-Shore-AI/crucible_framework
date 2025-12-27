defmodule Crucible.Stage.Schema.NormalizerTest do
  use ExUnit.Case, async: true

  alias Crucible.Stage.Schema.Normalizer

  describe "normalize/1" do
    test "returns schema unchanged if already canonical" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [:verbose],
        types: %{verbose: :boolean}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :test_stage
      assert normalized.required == []
      assert normalized.optional == [:verbose]
      assert normalized.types == %{verbose: :boolean}
    end

    test "converts string name to atom" do
      schema = %{
        name: "my_stage",
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :my_stage
    end

    test "converts :stage key to :name" do
      schema = %{
        stage: :my_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :my_stage
      refute Map.has_key?(normalized, :stage)
    end

    test "converts :stage key with string value to :name atom" do
      schema = %{
        stage: "my_stage",
        description: "A test stage"
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :my_stage
      refute Map.has_key?(normalized, :stage)
    end

    test "adds missing required field as empty list" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        optional: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.required == []
    end

    test "adds missing optional field as empty list" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.optional == []
    end

    test "adds missing types field as empty map" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: []
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.types == %{}
    end

    test "moves non-core fields to __extensions__" do
      schema = %{
        name: :ensemble_voting,
        description: "A voting stage",
        required: [],
        optional: [],
        types: %{},
        strategies: [:majority, :weighted],
        execution_modes: [:parallel, :sequential],
        custom_field: "value"
      }

      normalized = Normalizer.normalize(schema)

      refute Map.has_key?(normalized, :strategies)
      refute Map.has_key?(normalized, :execution_modes)
      refute Map.has_key?(normalized, :custom_field)

      assert normalized.__extensions__.strategies == [:majority, :weighted]
      assert normalized.__extensions__.execution_modes == [:parallel, :sequential]
      assert normalized.__extensions__.custom_field == "value"
    end

    test "merges with existing __extensions__" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{},
        custom_field: "value",
        __extensions__: %{
          existing: :data
        }
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.__extensions__.existing == :data
      assert normalized.__extensions__.custom_field == "value"
    end

    test "preserves __schema_version__ in core" do
      schema = %{
        __schema_version__: "1.0.0",
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.__schema_version__ == "1.0.0"
      refute Map.has_key?(Map.get(normalized, :__extensions__, %{}), :__schema_version__)
    end

    test "preserves defaults in core" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [:verbose],
        types: %{verbose: :boolean},
        defaults: %{verbose: false}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.defaults == %{verbose: false}
    end

    test "preserves version in core" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{},
        version: "0.1.0"
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.version == "0.1.0"
    end

    test "handles legacy CrucibleEnsemble format" do
      schema = %{
        name: "ensemble_voting",
        description: "Multi-model ensemble voting",
        version: "0.4.0",
        behaviour: Crucible.Stage,
        inputs: [{:context, :outputs, "List of model responses"}],
        outputs: [{:artifact, :ensemble_result, "Voting result"}],
        config_type: CrucibleIR.Reliability.Ensemble,
        strategies: [:majority, :weighted],
        execution_modes: [:parallel]
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :ensemble_voting
      assert normalized.description == "Multi-model ensemble voting"
      assert normalized.required == []
      assert normalized.optional == []
      assert normalized.types == %{}
      assert normalized.version == "0.4.0"

      assert normalized.__extensions__.behaviour == Crucible.Stage
      assert normalized.__extensions__.inputs == [{:context, :outputs, "List of model responses"}]
      assert normalized.__extensions__.strategies == [:majority, :weighted]
    end

    test "handles legacy CrucibleHedging format with :stage key" do
      schema = %{
        stage: :hedging,
        description: "Request hedging stage",
        inputs: [{:context, :request_fn}],
        outputs: [{:assign, :result}],
        config: %{strategy: :fixed}
      }

      normalized = Normalizer.normalize(schema)

      assert normalized.name == :hedging
      assert normalized.description == "Request hedging stage"
      assert normalized.required == []
      assert normalized.optional == []
      assert normalized.types == %{}

      assert normalized.__extensions__.inputs == [{:context, :request_fn}]
      assert normalized.__extensions__.outputs == [{:assign, :result}]
      assert normalized.__extensions__.config == %{strategy: :fixed}
    end

    test "does not add __extensions__ if no extra fields" do
      schema = %{
        name: :test_stage,
        description: "A test stage",
        required: [],
        optional: [],
        types: %{}
      }

      normalized = Normalizer.normalize(schema)

      refute Map.has_key?(normalized, :__extensions__)
    end
  end
end
