defmodule Crucible.Stage.BackendCall do
  @moduledoc """
  Executes training or sampling against a configured backend.
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.{BackendManager, Context}
  alias Crucible.IR.BackendRef

  @impl true
  def run(%Context{experiment: %{backend: %BackendRef{} = ref}} = ctx, opts) do
    with {:ok, mod, state, backend_state} <- BackendManager.ensure_state(ctx.backend_state, ref),
         {:ok, session, backend_sessions} <-
           BackendManager.ensure_session(ctx.backend_sessions, mod, state, ref, ctx.experiment) do
      mode = Map.get(opts, :mode, :train)
      telemetry_start(mode, ref)

      result =
        case mode do
          :train -> do_train(ctx, mod, session, backend_state, backend_sessions, opts)
          :sample -> do_sample(ctx, mod, session, backend_state, backend_sessions, opts)
          _ -> {:error, {:unknown_mode, mode}}
        end

      telemetry_stop(mode, ref, result)
      result
    end
  end

  def run(_ctx, _opts), do: {:error, :missing_backend}

  defp do_train(ctx, mod, session, backend_state, backend_sessions, opts) do
    batches = ctx.batches || []

    train_results =
      batches
      |> Enum.with_index(1)
      |> Enum.reduce_while({[], 0}, fn {batch, idx}, {acc, total} ->
        case mod.train_step(session, batch) do
          {:ok, step} ->
            Logger.info("Trained batch #{idx}/#{length(batches)} (loss=#{step.loss})")
            {:cont, {[step | acc], total + step.batch_size}}

          {:error, reason} ->
            {:halt, {:error, {:train_failed, idx, reason}}}
        end
      end)

    case train_results do
      {:error, reason} ->
        {:error, reason}

      {steps, total} ->
        metrics = aggregate_train(Enum.reverse(steps), total)

        checkpoint? = Map.get(opts, :checkpoint?, true)
        sampler? = Map.get(opts, :create_sampler?, checkpoint?)

        {checkpoint_ref, sampler} =
          maybe_checkpoint(mod, session, steps, checkpoint?, sampler?)

        eval_prompts = Map.get(opts, :sample_prompts, [])

        {samples, sample_errors} =
          if sampler && eval_prompts != [] do
            run_samples(mod, sampler, eval_prompts, Map.get(opts, :sample_opts, %{}))
          else
            {[], []}
          end

        metrics =
          metrics
          |> Map.put(:checkpoint, checkpoint_ref)
          |> Map.put(:samples, length(samples))
          |> Map.put(:sample_errors, length(sample_errors))

        new_ctx =
          %Context{
            ctx
            | backend_state: backend_state,
              backend_sessions: backend_sessions,
              metrics: Map.put(ctx.metrics, :backend, metrics),
              outputs: ctx.outputs ++ samples,
              assigns:
                ctx.assigns
                |> Map.put(:checkpoint_ref, checkpoint_ref)
                |> Map.put(:sampler, sampler)
          }

        {:ok, new_ctx}
    end
  end

  defp do_sample(ctx, mod, session, backend_state, backend_sessions, opts) do
    prompts = Map.get(opts, :prompts, [])

    sampler =
      case Map.fetch(ctx.assigns, :sampler) do
        {:ok, sampler} -> sampler
        :error -> Map.get(opts, :sampler)
      end

    with {:ok, sampler} <-
           ensure_sampler(mod, session, sampler, ctx.assigns[:checkpoint_ref], opts) do
      {samples, sample_errors} =
        run_samples(mod, sampler, prompts, Map.get(opts, :sample_opts, %{}))

      metrics = %{
        samples: length(samples),
        sample_errors: length(sample_errors)
      }

      new_ctx =
        %Context{
          ctx
          | backend_state: backend_state,
            backend_sessions: backend_sessions,
            outputs: ctx.outputs ++ samples,
            metrics: Map.put(ctx.metrics, :backend, metrics),
            assigns: Map.put(ctx.assigns, :sampler, sampler)
        }

      {:ok, new_ctx}
    end
  end

  defp aggregate_train(steps, total_batch_size) do
    losses = Enum.map(steps, & &1.loss)
    mean_loss = if losses == [], do: 0.0, else: Enum.sum(losses) / length(losses)

    %{
      total_steps: length(steps),
      total_examples: total_batch_size,
      mean_loss: mean_loss,
      raw_steps: steps
    }
  end

  defp maybe_checkpoint(mod, session, steps, checkpoint?, sampler?) do
    if checkpoint? and steps != [] do
      case mod.save_checkpoint(session, length(steps)) do
        {:ok, ref} ->
          sampler =
            if sampler? do
              case mod.create_sampler(session, ref) do
                {:ok, sampler} -> sampler
                _ -> nil
              end
            else
              nil
            end

          {ref, sampler}

        {:error, _} ->
          {nil, nil}
      end
    else
      {nil, nil}
    end
  end

  defp run_samples(mod, sampler, prompts, sample_opts) do
    Enum.reduce(prompts, {[], []}, fn prompt, {ok, err} ->
      case mod.sample(sampler, prompt, sample_opts) do
        {:ok, outputs} -> {[%{prompt: prompt, responses: outputs} | ok], err}
        {:error, reason} -> {ok, [%{prompt: prompt, error: reason} | err]}
      end
    end)
    |> then(fn {ok, err} -> {Enum.reverse(ok), Enum.reverse(err)} end)
  end

  defp ensure_sampler(_mod, _session, sampler, _checkpoint, _opts) when not is_nil(sampler),
    do: {:ok, sampler}

  defp ensure_sampler(mod, session, _sampler, checkpoint, opts) do
    cond do
      checkpoint ->
        mod.create_sampler(session, checkpoint)

      opts[:create_new_sampler?] ->
        mod.save_checkpoint(session, 0)
        |> case do
          {:ok, ref} -> mod.create_sampler(session, ref)
          other -> other
        end

      true ->
        {:error, :no_sampler}
    end
  end

  defp telemetry_start(mode, %BackendRef{id: id}) do
    :telemetry.execute([:crucible, :backend, :start], %{mode: mode}, %{backend: id})
  end

  defp telemetry_stop(mode, %BackendRef{id: id}, result) do
    status = if match?({:ok, _}, result), do: :ok, else: :error

    :telemetry.execute([:crucible, :backend, :stop], %{mode: mode, status: status}, %{backend: id})
  end
end
