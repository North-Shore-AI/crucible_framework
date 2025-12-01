defmodule Crucible.Stage.BackendCall do
  @moduledoc """
  Canonical backend call stage with ensemble voting and request hedging support.

  This module extends the basic BackendCall functionality to support:
  - Ensemble voting across multiple backend models
  - Request hedging for tail latency reduction
  - Trace integration for decision transparency

  ## Ensemble Support

  When `experiment.reliability.ensemble.strategy != :none`, this stage will:
  1. Query multiple backend members in parallel
  2. Apply voting strategy (majority, weighted, best_confidence)
  3. Track costs and consensus metrics

  ## Hedging Support

  When `experiment.reliability.hedging.strategy != :off`, this stage will:
  1. Wrap requests with hedging for latency reduction
  2. Apply configured hedging strategy (fixed, percentile, adaptive)
  3. Track hedge effectiveness metrics

  ## Configuration Examples

  Ensemble configuration:

      %EnsembleConfig{
        strategy: :majority_vote,
        members: [
          %BackendRef{id: :gpt4, options: %{}},
          %BackendRef{id: :claude, options: %{}},
          %BackendRef{id: :gemini, options: %{}}
        ],
        options: %{min_consensus: 0.66}
      }

  Hedging configuration:

      %HedgingConfig{
        strategy: :percentile,
        percentile: 95,
        max_extra_requests: 1,
        options: %{}
      }
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.{BackendManager, Context}
  alias CrucibleIR.BackendRef
  alias CrucibleIR.Reliability.{Ensemble, Hedging}
  alias CrucibleEnsemble
  alias CrucibleHedging
  alias CrucibleTrace

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    # Check if backend is configured
    if experiment.backend == nil do
      {:error, :missing_backend}
    else
      ensemble_config = experiment.reliability.ensemble
      hedging_config = experiment.reliability.hedging

      # Check if we need ensemble processing
      if should_use_ensemble?(ensemble_config) do
        run_with_ensemble(ctx, opts, ensemble_config, hedging_config)
      else
        # Fall back to regular backend call with optional hedging
        run_single_backend(ctx, opts, hedging_config)
      end
    end
  end

  def run(_ctx, _opts), do: {:error, :missing_backend}

  # Check if ensemble should be used
  defp should_use_ensemble?(%Ensemble{strategy: :none}), do: false
  defp should_use_ensemble?(%Ensemble{strategy: _, models: []}), do: false
  defp should_use_ensemble?(%Ensemble{strategy: _, models: nil}), do: false

  defp should_use_ensemble?(%Ensemble{strategy: _, models: models}) when is_list(models),
    do: true

  defp should_use_ensemble?(_), do: false

  # Run with ensemble voting
  defp run_with_ensemble(ctx, opts, ensemble_config, hedging_config) do
    mode = Map.get(opts, :mode, :sample)

    if mode != :sample do
      Logger.warning("Ensemble mode only supports sampling, not training")
      {:error, :ensemble_requires_sample_mode}
    else
      # Initialize all backend members
      with {:ok, backend_states} <- init_ensemble_backends(ctx, ensemble_config.models) do
        prompts = Map.get(opts, :prompts, [])

        # Process each prompt with ensemble
        results =
          Enum.map(prompts, fn prompt ->
            run_ensemble_prompt(prompt, backend_states, ensemble_config, hedging_config, ctx)
          end)

        # Aggregate results
        {successes, failures} = Enum.split_with(results, fn r -> match?({:ok, _}, r) end)

        samples = Enum.map(successes, fn {:ok, sample} -> sample end)
        errors = Enum.map(failures, fn {:error, error} -> error end)

        # Calculate ensemble metrics
        ensemble_metrics = %{
          strategy: ensemble_config.strategy,
          members_count: length(ensemble_config.models),
          samples: length(samples),
          errors: length(errors),
          average_consensus: calculate_average_consensus(samples),
          total_cost: calculate_total_cost(samples)
        }

        new_ctx = %Context{
          ctx
          | outputs: ctx.outputs ++ samples,
            metrics:
              ctx.metrics
              |> Map.put(:backend, Map.get(ctx.metrics, :backend, %{}))
              |> Map.put(:ensemble, ensemble_metrics)
        }

        # Add trace event if tracing is enabled
        new_ctx = maybe_add_trace_event(new_ctx, :ensemble_completed, ensemble_metrics)

        {:ok, new_ctx}
      end
    end
  end

  # Run single backend with optional hedging
  defp run_single_backend(
         %Context{experiment: %{backend: %BackendRef{} = ref}} = ctx,
         opts,
         hedging_config
       ) do
    with {:ok, mod, state, backend_state} <- BackendManager.ensure_state(ctx.backend_state, ref),
         {:ok, session, backend_sessions} <-
           BackendManager.ensure_session(ctx.backend_sessions, mod, state, ref, ctx.experiment) do
      mode = Map.get(opts, :mode, :train)
      telemetry_start(mode, ref)

      # Wrap in hedging if configured
      result =
        if should_use_hedging?(hedging_config) and mode == :sample do
          run_with_hedging(
            ctx,
            mod,
            session,
            backend_state,
            backend_sessions,
            opts,
            hedging_config
          )
        else
          # Original implementation
          case mode do
            :train -> do_train(ctx, mod, session, backend_state, backend_sessions, opts)
            :sample -> do_sample(ctx, mod, session, backend_state, backend_sessions, opts)
            _ -> {:error, {:unknown_mode, mode}}
          end
        end

      telemetry_stop(mode, ref, result)
      result
    end
  end

  # Check if hedging should be used
  defp should_use_hedging?(%Hedging{strategy: :off}), do: false
  defp should_use_hedging?(%Hedging{strategy: _}), do: true
  defp should_use_hedging?(_), do: false

  # Initialize all ensemble backend members
  defp init_ensemble_backends(ctx, members) do
    results =
      Enum.map(members, fn %BackendRef{} = ref ->
        case BackendManager.ensure_state(ctx.backend_state, ref) do
          {:ok, mod, state, _} ->
            case BackendManager.ensure_session(
                   ctx.backend_sessions,
                   mod,
                   state,
                   ref,
                   ctx.experiment
                 ) do
              {:ok, session, _} -> {:ok, {ref, mod, session}}
              error -> error
            end

          error ->
            error
        end
      end)

    # Check if all succeeded
    errors = Enum.filter(results, fn r -> not match?({:ok, _}, r) end)

    if errors == [] do
      {:ok, Enum.map(results, fn {:ok, backend} -> backend end)}
    else
      {:error, {:ensemble_init_failed, errors}}
    end
  end

  # Run a single prompt through the ensemble
  defp run_ensemble_prompt(prompt, backend_states, ensemble_config, hedging_config, ctx) do
    start_time = System.monotonic_time(:millisecond)

    # Create a function for each backend call
    backend_fns =
      Enum.map(backend_states, fn {ref, mod, session} ->
        fn ->
          # Create sampler if needed
          case ensure_sampler_for_backend(mod, session, ctx.assigns[:checkpoint_ref], %{}) do
            {:ok, sampler} ->
              # Optionally wrap in hedging
              if should_use_hedging?(hedging_config) do
                run_hedged_sample(mod, sampler, prompt, ref, hedging_config)
              else
                run_regular_sample(mod, sampler, prompt, ref)
              end

            error ->
              error
          end
        end
      end)

    # Execute all backends in parallel
    responses =
      backend_fns
      |> Task.async_stream(& &1.(), timeout: 30_000, on_timeout: :kill_task)
      |> Enum.map(fn
        {:ok, {:ok, response}} -> response
        {:ok, {:error, reason}} -> %{error: reason}
        {:exit, _} -> %{error: :timeout}
      end)
      |> Enum.filter(fn r -> not Map.has_key?(r, :error) end)

    if responses == [] do
      {:error, %{prompt: prompt, error: "All ensemble members failed"}}
    else
      # Apply voting strategy - map to CrucibleEnsemble's expected names
      strategy =
        case ensemble_config.strategy do
          :majority_vote -> :majority
          :weighted_vote -> :weighted
          :best_confidence -> :best_confidence
          :unanimous -> :unanimous
          other -> other
        end

      vote_opts = normalize_options(ensemble_config.options)

      case CrucibleEnsemble.Vote.apply_strategy(responses, strategy, vote_opts) do
        {:ok, vote_result} ->
          end_time = System.monotonic_time(:millisecond)

          {:ok,
           %{
             prompt: prompt,
             response: vote_result.answer,
             ensemble_metadata: %{
               strategy: ensemble_config.strategy,
               consensus: vote_result.consensus,
               votes: Map.get(vote_result, :votes, %{}),
               members_responded: length(responses),
               latency_ms: end_time - start_time,
               costs: Enum.map(responses, fn r -> Map.get(r, :cost, 0) end) |> Enum.sum()
             }
           }}

        {:error, reason} ->
          {:error, %{prompt: prompt, error: reason}}
      end
    end
  end

  # Run sample with hedging
  defp run_with_hedging(ctx, mod, session, backend_state, backend_sessions, opts, hedging_config) do
    prompts = Map.get(opts, :prompts, [])

    sampler =
      case Map.fetch(ctx.assigns, :sampler) do
        {:ok, sampler} -> sampler
        :error -> Map.get(opts, :sampler)
      end

    with {:ok, sampler} <-
           ensure_sampler(mod, session, sampler, ctx.assigns[:checkpoint_ref], opts) do
      # Run samples with hedging
      {samples, sample_errors, hedge_metrics} =
        run_hedged_samples(mod, sampler, prompts, hedging_config, opts)

      metrics = %{
        samples: length(samples),
        sample_errors: length(sample_errors),
        hedging: hedge_metrics
      }

      new_ctx = %Context{
        ctx
        | backend_state: backend_state,
          backend_sessions: backend_sessions,
          outputs: ctx.outputs ++ samples,
          metrics: Map.put(ctx.metrics, :backend, metrics),
          assigns: Map.put(ctx.assigns, :sampler, sampler)
      }

      # Add trace event
      new_ctx = maybe_add_trace_event(new_ctx, :hedging_completed, hedge_metrics)

      {:ok, new_ctx}
    end
  end

  # Run samples with hedging
  defp run_hedged_samples(mod, sampler, prompts, hedging_config, opts) do
    sample_opts = Map.get(opts, :sample_opts, %{})

    results =
      Enum.map(prompts, fn prompt ->
        hedging_opts = build_hedging_opts(hedging_config)

        case CrucibleHedging.request(
               fn -> mod.sample(sampler, prompt, sample_opts) end,
               hedging_opts
             ) do
          {:ok, outputs, metadata} ->
            {:ok, %{prompt: prompt, responses: outputs, hedge_metadata: metadata}}

          {:error, reason} ->
            {:error, %{prompt: prompt, error: reason}}
        end
      end)

    {successes, failures} = Enum.split_with(results, fn r -> match?({:ok, _}, r) end)

    samples = Enum.map(successes, fn {:ok, sample} -> sample end)
    errors = Enum.map(failures, fn {:error, error} -> error end)

    # Aggregate hedging metrics
    hedge_metrics = aggregate_hedge_metrics(samples)

    {samples, errors, hedge_metrics}
  end

  # Build hedging options from config
  defp build_hedging_opts(%Hedging{} = config, strategy_name \\ nil) do
    strategy =
      case config.strategy do
        :fixed_delay -> :fixed
        other -> other
      end

    base_opts = [
      strategy: strategy,
      max_hedges: config.max_hedges
    ]

    opts =
      case strategy do
        :fixed -> Keyword.put(base_opts, :delay_ms, config.delay_ms)
        :percentile -> Keyword.put(base_opts, :percentile, config.percentile)
        _ -> base_opts
      end

    opts =
      opts
      |> Keyword.merge(Map.to_list(config.options))

    if strategy_name && !Keyword.has_key?(opts, :strategy_name) do
      Keyword.put(opts, :strategy_name, strategy_name)
    else
      opts
    end
  end

  defp hedging_strategy_name(%BackendRef{id: id}, %Hedging{} = config) do
    Map.get(config.options, :strategy_name) || :"hedging_#{id}"
  end

  # Run a hedged sample for ensemble
  defp run_hedged_sample(mod, sampler, prompt, ref, hedging_config) do
    hedging_opts = build_hedging_opts(hedging_config, hedging_strategy_name(ref, hedging_config))

    case CrucibleHedging.request(
           fn -> mod.sample(sampler, prompt, %{}) end,
           hedging_opts
         ) do
      {:ok, outputs, metadata} ->
        cleaned_outputs =
          case outputs do
            {:ok, inner} -> inner
            other -> other
          end

        {:ok,
         %{
           model: ref.id,
           response: cleaned_outputs,
           latency_ms: Map.get(metadata, :total_latency, 0),
           cost: estimate_cost(ref, cleaned_outputs),
           hedged: Map.get(metadata, :hedged, false)
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Run a regular sample for ensemble
  defp run_regular_sample(mod, sampler, prompt, ref) do
    start_time = System.monotonic_time(:millisecond)

    case mod.sample(sampler, prompt, %{}) do
      {:ok, outputs} ->
        end_time = System.monotonic_time(:millisecond)

        {:ok,
         %{
           model: ref.id,
           response: outputs,
           latency_ms: end_time - start_time,
           cost: estimate_cost(ref, outputs)
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Helper to ensure sampler for backend
  defp ensure_sampler_for_backend(mod, session, checkpoint, opts) do
    cond do
      checkpoint ->
        mod.create_sampler(session, checkpoint)

      opts[:create_new_sampler?] ->
        case mod.save_checkpoint(session, 0) do
          {:ok, ref} -> mod.create_sampler(session, ref)
          other -> other
        end

      true ->
        # Try to create a default sampler
        case mod.save_checkpoint(session, 0) do
          {:ok, ref} -> mod.create_sampler(session, ref)
          _ -> {:error, :no_sampler}
        end
    end
  end

  # Estimate cost for a backend response
  defp estimate_cost(%BackendRef{id: backend_id}, outputs) when is_list(outputs) do
    # Rough estimation based on output length
    # This should be replaced with actual cost calculation
    text_length = outputs |> Enum.join(" ") |> String.length()
    # Rough token estimate
    tokens = div(text_length, 4)

    # Placeholder costs per 1K tokens
    cost_per_1k =
      case backend_id do
        :gpt4 -> 0.03
        :claude -> 0.025
        :gemini -> 0.02
        _ -> 0.01
      end

    tokens * cost_per_1k / 1000
  end

  defp estimate_cost(_, _), do: 0.0

  # Calculate average consensus from samples
  defp calculate_average_consensus(samples) do
    consensuses =
      samples
      |> Enum.map(fn s -> get_in(s, [:ensemble_metadata, :consensus]) || 0 end)
      |> Enum.filter(&(&1 > 0))

    if consensuses == [] do
      0.0
    else
      Enum.sum(consensuses) / length(consensuses)
    end
  end

  # Calculate total cost from samples
  defp calculate_total_cost(samples) do
    samples
    |> Enum.map(fn s -> get_in(s, [:ensemble_metadata, :costs]) || 0 end)
    |> Enum.sum()
  end

  # Aggregate hedging metrics from samples
  defp aggregate_hedge_metrics(samples) do
    hedge_data =
      samples
      |> Enum.map(fn s -> Map.get(s, :hedge_metadata, %{}) end)
      |> Enum.filter(fn m -> m != %{} end)

    if hedge_data == [] do
      %{enabled: false}
    else
      total = length(hedge_data)
      hedged = Enum.count(hedge_data, fn m -> Map.get(m, :hedged, false) end)
      hedge_won = Enum.count(hedge_data, fn m -> Map.get(m, :hedge_won, false) end)

      avg_latency =
        hedge_data
        |> Enum.map(fn m -> Map.get(m, :total_latency, 0) end)
        |> Enum.sum()
        |> Kernel./(total)

      %{
        enabled: true,
        total_requests: total,
        hedged_requests: hedged,
        hedge_wins: hedge_won,
        hedge_rate: hedged / total,
        hedge_win_rate: if(hedged > 0, do: hedge_won / hedged, else: 0),
        average_latency_ms: avg_latency
      }
    end
  end

  # Add trace event if tracing is enabled
  defp maybe_add_trace_event(%Context{trace: nil} = ctx, _type, _data), do: ctx

  defp maybe_add_trace_event(%Context{trace: chain} = ctx, type, data) when not is_nil(chain) do
    event =
      CrucibleTrace.create_event(
        type,
        "#{type}",
        "Stage completed: #{type}",
        metadata: data
      )

    new_chain = CrucibleTrace.add_event(chain, event)
    %Context{ctx | trace: new_chain}
  end

  defp maybe_add_trace_event(ctx, _, _), do: ctx

  # Original methods from BackendCall (preserved for compatibility)

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
        mod.create_sampler(session, checkpoint)
    end
  end

  defp telemetry_start(mode, %BackendRef{id: id}) do
    :telemetry.execute([:crucible, :backend, :start], %{mode: mode}, %{backend: id})
  end

  defp telemetry_stop(mode, %BackendRef{id: id}, result) do
    status = if match?({:ok, _}, result), do: :ok, else: :error

    :telemetry.execute([:crucible, :backend, :stop], %{mode: mode, status: status}, %{backend: id})
  end

  defp normalize_options(opts) when is_map(opts), do: Map.to_list(opts)
  defp normalize_options(opts) when is_list(opts), do: opts
  defp normalize_options(_), do: []
end
