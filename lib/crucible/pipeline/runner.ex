defmodule Crucible.Pipeline.Runner do
  @moduledoc """
  Executes experiment pipelines stage-by-stage.

  ## Location and Ownership

  This is the **authoritative** pipeline runner for the Crucible ecosystem.
  It lives in `crucible_framework` and is the only component that executes
  experiment pipelines. `crucible_ir` defines specs only; it does not execute.

  ## Public Entrypoint

  Users should call `CrucibleFramework.run/2` rather than this module directly:

      {:ok, ctx} = CrucibleFramework.run(experiment)

  ## Pipeline Execution

  The runner:

  1. Initializes a `%Crucible.Context{}` from the experiment
  2. Optionally persists run state to the database
  3. Executes each `%CrucibleIR.StageDef{}` in sequence
  4. Resolves stage modules via `Crucible.Registry` or explicit `:module` field
  5. Optionally validates stage options against `describe/1` schema
  6. Calls `stage_module.run(context, opts)` for each stage
  7. Marks stages complete and emits trace events
  8. Finalizes the run with success or failure status

  ## Stage Resolution

  Stages are resolved in order:

  1. If `StageDef.module` is set, use that module directly
  2. Otherwise, look up `StageDef.name` in `Crucible.Registry`

  ## Options Validation

  The runner supports opt-in validation of stage options against the schema
  returned by each stage's `describe/1` callback:

      CrucibleFramework.run(experiment, validate_options: :error)

  - `:off` (default) - No validation
  - `:warn` - Log warnings for invalid options but continue execution
  - `:error` - Fail immediately on invalid options

  ## Trace Integration

  When `:enable_trace` is passed, the runner emits stage lifecycle events
  via `Crucible.TraceIntegration` for observability and debugging.
  """

  require Logger

  alias Crucible.{Context, Registry, TraceIntegration}
  alias Crucible.Stage.Validator
  alias CrucibleFramework.Persistence
  alias CrucibleIR.{Experiment, StageDef}

  @doc """
  Runs an experiment, optionally persisting run state.

  ## Options

  - `:run_id` - Custom run ID (defaults to UUID)
  - `:persist` - Whether to persist run state (default: true)
  - `:enable_trace` - Enable trace integration (default: false)
  - `:assigns` - Initial context assigns (default: %{})
  - `:validate_options` - Options validation mode:
    - `:off` (default) - No validation
    - `:warn` - Log warnings but continue
    - `:error` - Fail on validation errors
  """
  @spec run(Experiment.t(), keyword()) :: {:ok, Context.t()} | {:error, term()}
  def run(%Experiment{} = experiment, opts \\ []) do
    run_id = Keyword.get(opts, :run_id, Ecto.UUID.generate())
    persist? = Keyword.get(opts, :persist, true)
    validate_mode = Keyword.get(opts, :validate_options, :off)

    {run_record, ctx} =
      case persist? do
        true ->
          case Persistence.start_run(experiment, metadata: %{run_id: run_id}) do
            {:ok, run} -> {run, build_context(experiment, run_id, opts) |> put_run(run)}
            {:error, _} -> {nil, build_context(experiment, run_id, opts)}
          end

        false ->
          {nil, build_context(experiment, run_id, opts)}
      end

    result =
      Enum.reduce_while(experiment.pipeline, {:ok, ctx}, fn %StageDef{} = stage_def,
                                                            {:ok, ctx_acc} ->
        run_stage(stage_def, ctx_acc, validate_mode)
      end)

    finalize(result, run_record)
  end

  defp run_stage(%StageDef{} = stage_def, ctx_acc, validate_mode) do
    stage_def = normalize_options(stage_def)

    case resolve_stage(stage_def) do
      {:ok, mod} ->
        log_stage(stage_def.name)
        ctx_acc = TraceIntegration.emit_stage_start(ctx_acc, stage_def.name, stage_def.options)

        case validate_stage_options(mod, stage_def, validate_mode) do
          :ok ->
            execute_stage(mod, stage_def, ctx_acc)

          {:error, errors} ->
            ctx_acc =
              TraceIntegration.emit_stage_failed(
                ctx_acc,
                stage_def.name,
                {:invalid_options, errors}
              )

            {:halt, {:error, {stage_def.name, {:invalid_options, errors}}, ctx_acc}}
        end

      {:error, reason} ->
        ctx_acc = TraceIntegration.emit_stage_failed(ctx_acc, stage_def.name, reason)
        {:halt, {:error, {stage_def.name, reason}, ctx_acc}}
    end
  end

  defp execute_stage(mod, stage_def, ctx_acc) do
    case mod.run(ctx_acc, stage_def.options) do
      {:ok, new_ctx} ->
        new_ctx = Context.mark_stage_complete(new_ctx, stage_def.name)
        new_ctx = TraceIntegration.emit_stage_complete(new_ctx, stage_def.name, new_ctx.metrics)
        {:cont, {:ok, new_ctx}}

      {:error, reason} ->
        ctx_acc = TraceIntegration.emit_stage_failed(ctx_acc, stage_def.name, reason)
        {:halt, {:error, {stage_def.name, reason}, ctx_acc}}
    end
  end

  defp build_context(experiment, run_id, opts) do
    ctx = %Context{
      experiment_id: experiment.id,
      run_id: run_id,
      experiment: experiment,
      assigns: Keyword.get(opts, :assigns, %{})
    }

    # Initialize tracing if enabled
    if Keyword.get(opts, :enable_trace, false) do
      TraceIntegration.init_trace(ctx, experiment.id)
    else
      ctx
    end
  end

  defp validate_stage_options(_mod, _stage_def, :off), do: :ok

  defp validate_stage_options(mod, stage_def, mode) when mode in [:warn, :error] do
    if function_exported?(mod, :describe, 1) do
      schema = mod.describe(stage_def.options || %{})

      case Validator.validate(stage_def.options, schema) do
        :ok ->
          :ok

        {:error, errors} when mode == :warn ->
          Logger.warning(
            "Stage #{stage_def.name} options validation warnings: #{Enum.join(errors, ", ")}"
          )

          :ok

        {:error, errors} ->
          {:error, errors}
      end
    else
      :ok
    end
  end

  defp resolve_stage(%StageDef{module: nil, name: name}) do
    Registry.stage_module(name)
  end

  defp resolve_stage(%StageDef{module: mod}), do: {:ok, mod}

  defp log_stage(name) do
    Logger.info("Running stage #{name}")
  end

  defp finalize({:ok, ctx}, nil), do: {:ok, ctx}

  defp finalize({:ok, ctx}, run_record) do
    Persistence.finish_run(run_record, "completed", %{
      metrics: ctx.metrics,
      outputs: ctx.outputs,
      metadata: Map.put(run_record.metadata || %{}, "assigns", ctx.assigns)
    })

    {:ok, ctx}
  end

  defp finalize({:error, {stage, reason}, ctx}, nil), do: {:error, {stage, reason, ctx}}

  defp finalize({:error, {stage, reason}, ctx}, run_record) do
    failure = %{"stage" => to_string(stage), "reason" => inspect(reason)}

    Persistence.finish_run(run_record, "failed", %{
      metrics: ctx.metrics,
      metadata: Map.merge(run_record.metadata || %{}, %{"failure" => failure})
    })

    {:error, {stage, reason, ctx}}
  end

  defp put_run(ctx, run_record) do
    %Context{ctx | assigns: Map.put(ctx.assigns, :run_record, run_record)}
  end

  defp normalize_options(%StageDef{options: nil} = stage_def),
    do: %StageDef{stage_def | options: %{}}

  defp normalize_options(%StageDef{} = stage_def), do: stage_def
end
