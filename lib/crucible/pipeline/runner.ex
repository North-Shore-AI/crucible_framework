defmodule Crucible.Pipeline.Runner do
  @moduledoc """
  Executes experiment pipelines stage-by-stage.
  """

  alias Crucible.{Context, Registry}
  alias Crucible.IR.{Experiment, StageDef}
  alias CrucibleFramework.Persistence

  @doc """
  Runs an experiment, optionally persisting run state.
  """
  @spec run(Experiment.t(), keyword()) :: {:ok, Context.t()} | {:error, {atom(), term()}}
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
        case resolve_stage(stage_def) do
          {:ok, mod} ->
            log_stage(stage_def.name)

            case mod.run(ctx_acc, stage_def.options) do
              {:ok, new_ctx} -> {:cont, {:ok, new_ctx}}
              {:error, reason} -> {:halt, {:error, {stage_def.name, reason}, ctx_acc}}
            end

          {:error, reason} ->
            {:halt, {:error, {stage_def.name, reason}, ctx_acc}}
        end
      end)

    finalize(result, run_record)
  end

  defp build_context(experiment, run_id, opts) do
    %Context{
      experiment_id: experiment.id,
      run_id: run_id,
      experiment: experiment,
      assigns: Keyword.get(opts, :assigns, %{})
    }
  end

  defp resolve_stage(%StageDef{module: nil, name: name}) do
    Registry.stage_module(name)
  end

  defp resolve_stage(%StageDef{module: mod}), do: {:ok, mod}

  defp log_stage(name) do
    IO.puts("→ Running stage #{name}")
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
