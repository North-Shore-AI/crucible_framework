defmodule Crucible.Stage.PlanStep do
  @moduledoc """
  Adapter stage for executing plan steps produced by `Crucible.PlanAdapter`.

  By default, this stage executes an action using `Jido.Exec` when available.
  If `Jido.Exec` is not loaded, it falls back to calling `action.run/2` directly.

  Results are stored in `context.assigns[:plan_results]` keyed by `:result_key`
  (defaults to the plan step name), and the most recent result is stored in
  `context.assigns[:last_result]`.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  @compile {:no_warn_undefined, Jido.Exec}

  @impl true
  def run(%Context{} = ctx, opts) do
    with {:ok, instruction} <- normalize_instruction(opts[:instruction]),
         {:ok, action} <- resolve_action(opts[:action] || instruction.action),
         {:ok, params} <- resolve_map(opts[:params] || instruction.params, :params),
         {:ok, context} <- resolve_map(opts[:context] || instruction.context, :context),
         {:ok, exec_opts} <- resolve_opts(opts[:exec_opts] || instruction.opts),
         {:ok, action_context} <- build_action_context(ctx, opts, context) do
      case execute_action(action, params, action_context, exec_opts) do
        {:ok, result} ->
          {:ok, record_result(ctx, opts, result)}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @impl true
  def describe(_opts) do
    %{
      name: :plan_step,
      description: "Executes a Jido-style instruction inside a Crucible pipeline",
      required: [:action],
      optional: [
        :params,
        :context,
        :exec_opts,
        :instruction,
        :plan_id,
        :step_id,
        :step_name,
        :depends_on,
        :plan_context,
        :result_key
      ],
      types: %{
        action: :module,
        params: :map,
        context: :map,
        exec_opts: :list,
        instruction: :any,
        plan_id: :any,
        step_id: :any,
        step_name: :atom,
        depends_on: {:list, :atom},
        plan_context: :map,
        result_key: :atom
      }
    }
  end

  defp resolve_action(action) when is_atom(action), do: {:ok, action}
  defp resolve_action(_action), do: {:error, :missing_action}

  defp build_action_context(%Context{} = ctx, opts, context) do
    plan_context = resolve_plan_context(opts[:plan_context])
    shared_context = Map.merge(plan_context, context)
    result_context = Map.get(ctx.assigns, :plan_results, %{})

    action_context =
      shared_context
      |> Map.put_new(:plan_id, opts[:plan_id])
      |> Map.put_new(:step_id, opts[:step_id])
      |> Map.put_new(:step_name, opts[:step_name])
      |> Map.put_new(:run_id, ctx.run_id)
      |> Map.put_new(:experiment_id, ctx.experiment_id)
      |> Map.put_new(:trace_id, ctx.telemetry_context[:trace_id])
      |> Map.put_new(:depends_on, opts[:depends_on])
      |> Map.put_new(:plan_results, result_context)

    {:ok, action_context}
  end

  defp resolve_plan_context(nil), do: %{}
  defp resolve_plan_context(map) when is_map(map), do: map
  defp resolve_plan_context(list) when is_list(list), do: Map.new(list)
  defp resolve_plan_context(_), do: %{}

  defp execute_action(action, params, context, exec_opts) do
    if Code.ensure_loaded?(Jido.Exec) and function_exported?(Jido.Exec, :run, 4) do
      execute_with_jido(action, params, context, exec_opts)
    else
      execute_direct(action, params, context)
    end
  end

  defp execute_with_jido(action, params, context, exec_opts) do
    # credo:disable-for-next-line Credo.Check.Refactor.Apply
    case apply(Jido.Exec, :run, [action, params, context, exec_opts]) do
      {:ok, result} -> {:ok, result}
      {:ok, result, _meta} -> {:ok, result}
      {:error, reason} -> {:error, reason}
      {:error, reason, _meta} -> {:error, reason}
      other -> {:error, {:invalid_action_result, other}}
    end
  end

  defp execute_direct(action, params, context) do
    if function_exported?(action, :run, 2) do
      case action.run(params, context) do
        {:ok, result} -> {:ok, result}
        {:error, reason} -> {:error, reason}
        other -> {:error, {:invalid_action_result, other}}
      end
    else
      {:error, {:missing_action, action}}
    end
  end

  defp record_result(%Context{} = ctx, opts, result) do
    result_key = Map.get(opts, :result_key, opts[:step_name] || :last_result)
    plan_results = Map.get(ctx.assigns, :plan_results, %{})

    ctx
    |> Context.assign(:plan_results, Map.put(plan_results, result_key, result))
    |> Context.assign(:last_result, result)
  end

  defp resolve_map(nil, _label), do: {:ok, %{}}
  defp resolve_map(map, _label) when is_map(map), do: {:ok, map}
  defp resolve_map(list, _label) when is_list(list), do: {:ok, Map.new(list)}
  defp resolve_map(value, label), do: {:error, {:invalid_map, label, value}}

  defp resolve_opts(nil), do: {:ok, []}
  defp resolve_opts(opts) when is_list(opts), do: {:ok, opts}
  defp resolve_opts(value), do: {:error, {:invalid_opts, value}}

  defp normalize_instruction(nil), do: {:ok, %{action: nil, params: %{}, context: %{}, opts: []}}

  defp normalize_instruction(%{action: action} = instruction) when is_atom(action) do
    params = Map.get(instruction, :params, %{})
    context = Map.get(instruction, :context, %{})
    opts = Map.get(instruction, :opts, [])

    with {:ok, params} <- resolve_map(params, :params),
         {:ok, context} <- resolve_map(context, :context),
         {:ok, opts} <- resolve_opts(opts) do
      {:ok, %{action: action, params: params, context: context, opts: opts}}
    end
  end

  defp normalize_instruction(action) when is_atom(action) do
    {:ok, %{action: action, params: %{}, context: %{}, opts: []}}
  end

  defp normalize_instruction({action, params}) when is_atom(action) do
    with {:ok, params} <- resolve_map(params, :params) do
      {:ok, %{action: action, params: params, context: %{}, opts: []}}
    end
  end

  defp normalize_instruction({action, params, context}) when is_atom(action) do
    with {:ok, params} <- resolve_map(params, :params),
         {:ok, context} <- resolve_map(context, :context) do
      {:ok, %{action: action, params: params, context: context, opts: []}}
    end
  end

  defp normalize_instruction({action, params, context, opts}) when is_atom(action) do
    with {:ok, params} <- resolve_map(params, :params),
         {:ok, context} <- resolve_map(context, :context),
         {:ok, opts} <- resolve_opts(opts) do
      {:ok, %{action: action, params: params, context: context, opts: opts}}
    end
  end

  defp normalize_instruction(_instruction), do: {:error, :invalid_instruction}
end
