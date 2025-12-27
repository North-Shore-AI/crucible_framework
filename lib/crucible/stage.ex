defmodule Crucible.Stage do
  @moduledoc """
  Behaviour for a single experiment pipeline stage.

  Stages transform a `%Crucible.Context{}` and may enrich metrics, outputs,
  or orchestrate I/O with backends and external systems.

  ## Runner Location

  The pipeline runner lives in `crucible_framework`:

  - `Crucible.Pipeline.Runner` - executes experiment pipelines stage-by-stage
  - `CrucibleFramework.run/2` - public entrypoint

  `crucible_ir` does **not** execute anything; it only defines specs. All execution
  logic is owned by `crucible_framework`.

  ## Required Callback: `run/2`

  Every stage must implement `run/2`:

      @impl true
      def run(%Crucible.Context{} = ctx, opts) do
        # Transform context, perform work
        {:ok, updated_ctx}
      end

  - Returns `{:ok, %Crucible.Context{}}` on success
  - Returns `{:error, reason}` on failure
  - Must not mutate global state or bypass persistence helpers
  - Should be network-mockable and testable in isolation

  ## Policy Callback: `describe/1`

  While optional at the behaviour level, `describe/1` is **required by policy**
  for all stage implementations. It provides a discoverable schema for stage options:

      @impl true
      def describe(_opts) do
        %{
          name: :my_stage,
          description: "Human-readable description of what this stage does",
          required: [:model_name, :dataset_path],
          optional: [:batch_size, :seed, :log_level],
          types: %{
            model_name: :string,
            dataset_path: :string,
            batch_size: :integer,
            seed: :integer,
            log_level: {:enum, [:debug, :info, :warn, :error]}
          }
        }
      end

  ### Schema Keys

  - `:name` - Stage identifier (atom)
  - `:description` - Human-readable description (string)
  - `:required` - List of required option keys (list of atoms)
  - `:optional` - List of optional option keys (list of atoms)
  - `:types` - Map of key to type specification

  ### Type Specifications

  - `:string` - String value
  - `:integer` - Integer value
  - `:float` - Float value
  - `:boolean` - Boolean value
  - `:atom` - Atom value
  - `:map` - Map value
  - `:list` - List value
  - `{:struct, Module}` - Struct of the given module
  - `{:enum, [values]}` - One of the enumerated values
  - `{:function, arity}` - Function with given arity

  ## Options Handling

  - `CrucibleIR.StageDef.options` is an opaque map owned by each stage
  - Stages own their own options schema and validation
  - Stages may accept typed configs (e.g., `%CrucibleIR.Training.Config{}`)
    but must normalize internally
  """

  alias Crucible.Context

  @type opts :: map()

  @doc """
  Executes the stage logic on the given context.

  ## Parameters

  - `context` - The `%Crucible.Context{}` struct threaded through the pipeline
  - `opts` - Stage-specific options (from `StageDef.options`)

  ## Returns

  - `{:ok, %Crucible.Context{}}` - Updated context on success
  - `{:error, reason}` - Error tuple on failure
  """
  @callback run(context :: Context.t(), opts :: opts()) ::
              {:ok, Context.t()} | {:error, term()}

  @doc """
  Returns a schema describing the stage's purpose and options.

  This callback is optional at the behaviour level but **required by policy**
  for all stage implementations. See module documentation for the expected schema.
  """
  @callback describe(opts :: opts()) :: map()
  @optional_callbacks describe: 1
end
