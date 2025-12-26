defmodule Crucible.Stage.Validate do
  @moduledoc """
  Pre-flight validation of experiment pipeline stages.

  This stage validates that all pipeline stages can be resolved and implement
  the `Crucible.Stage` behaviour. Domain-specific validation (backends, datasets,
  ensemble configuration, etc.) should be done by domain-specific stages.

  ## Validation Checks

  1. All stage names resolve to modules (via registry or explicit module)
  2. Stage modules can be loaded
  3. Stage modules implement `run/2`

  ## Configuration

  Validation can be configured via stage options:

      %StageDef{
        name: :validate,
        options: %{
          strict: true  # Fail on warnings (default: false)
        }
      }

  ## Examples

      # Basic validation
      pipeline: [%StageDef{name: :validate}, ...]

      # Strict validation (warnings are errors)
      pipeline: [%StageDef{name: :validate, options: %{strict: true}}, ...]
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.{Context, Registry}

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    Logger.info("Validating experiment: #{experiment.id}")

    strict = Map.get(opts, :strict, false)

    case validate_stages(experiment.pipeline) do
      {:ok, _} ->
        Logger.info("Experiment validation passed")
        ctx = Context.put_metric(ctx, :validation, %{status: :passed})
        {:ok, ctx}

      {:error, errors} when strict ->
        Logger.error("Experiment validation failed: #{inspect(errors)}")
        {:error, {:validation_failed, errors}}

      {:error, warnings} ->
        Logger.warning("Experiment validation warnings: #{inspect(warnings)}")
        ctx = Context.put_metric(ctx, :validation, %{status: :warnings, warnings: warnings})
        {:ok, ctx}
    end
  end

  @impl true
  def describe(_opts) do
    %{
      stage: :validate,
      description: "Pre-flight validation of pipeline stages"
    }
  end

  # ============================================================================
  # Stage Validation
  # ============================================================================

  defp validate_stages(pipeline) when is_list(pipeline) do
    errors =
      pipeline
      |> Enum.map(&validate_single_stage/1)
      |> Enum.reject(&is_nil/1)

    if errors == [], do: {:ok, []}, else: {:error, errors}
  end

  defp validate_stages(_) do
    {:error, ["Pipeline is not a list"]}
  end

  defp validate_single_stage(%{module: mod}) when not is_nil(mod) do
    cond do
      not Code.ensure_loaded?(mod) ->
        "Stage module #{inspect(mod)} cannot be loaded"

      not function_exported?(mod, :run, 2) ->
        "Stage module #{inspect(mod)} does not implement run/2"

      true ->
        nil
    end
  end

  defp validate_single_stage(%{name: name}) do
    case Registry.stage_module(name) do
      {:ok, mod} ->
        cond do
          not Code.ensure_loaded?(mod) ->
            "Stage :#{name} resolves to #{inspect(mod)} which cannot be loaded"

          not function_exported?(mod, :run, 2) ->
            "Stage :#{name} resolves to #{inspect(mod)} which does not implement run/2"

          true ->
            nil
        end

      {:error, {:unknown_stage, ^name}} ->
        "Stage :#{name} is not registered"

      {:error, :no_stage_registry} ->
        "No stage registry configured"
    end
  end

  defp validate_single_stage(_) do
    "Stage definition missing both :name and :module"
  end
end
