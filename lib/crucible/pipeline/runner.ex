defmodule Crucible.Pipeline.Runner do
  @moduledoc """
  Executes experiment pipelines stage-by-stage.
  """

  require Logger

  alias Crucible.{Context, Registry, TraceIntegration}
  alias CrucibleFramework.Persistence
  alias CrucibleIR.{Experiment, StageDef}

  @doc """
  Runs an experiment, optionally persisting run state.
  """
  @spec run(Experiment.t(), keyword()) :: {:ok, Context.t()} | {:error, term()}
  def run(%Experiment{} = experiment, opts \\ []) do
    run_id = Keyword.get(opts, :run_id, Ecto.UUID.generate())
    persist? = Keyword.get(opts, :persist, true)

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
        run_stage(stage_def, ctx_acc)
      end)

    finalize(result, run_record)
  end

  defp run_stage(%StageDef{} = stage_def, ctx_acc) do
    case resolve_stage(stage_def) do
      {:ok, mod} ->
        log_stage(stage_def.name)
        ctx_acc = TraceIntegration.emit_stage_start(ctx_acc, stage_def.name, stage_def.options)
        execute_stage(mod, stage_def, ctx_acc)

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
end
