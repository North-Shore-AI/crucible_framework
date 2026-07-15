defmodule Crucible.PlanAdapter do
  @moduledoc """
  Compiles Jido.Plan-style DAGs into ordered `CrucibleIR.StageDef` lists.

  This adapter intentionally works with the structural shape of `Jido.Plan`
  (maps containing `:steps`, `:context`, and `:id`) to avoid a hard dependency
  on `jido_action`. If `Jido.Plan` is available, pass it directly.

  Each plan step becomes a `StageDef` that defaults to the
  `Crucible.Stage.PlanStep` module, with normalized instruction metadata
  stored in `StageDef.options`.
  """

  alias CrucibleIR.StageDef

  @type plan :: map()

  @type plan_step :: %{
          name: atom(),
          id: String.t() | nil,
          instruction: term(),
          depends_on: [atom()],
          opts: keyword()
        }

  @doc """
  Converts a plan into a list of `StageDef` entries in topological order.

  Options:
  - `:stage_module` - Stage module to execute each plan step
  - `:include_plan_context` - Whether to include plan context in options (default: true)
  """
  @spec to_stage_defs(plan(), keyword()) :: {:ok, [StageDef.t()]} | {:error, term()}
  def to_stage_defs(plan, opts \\ [])

  def to_stage_defs(plan, opts) when is_map(plan) do
    with {:ok, {plan_id, plan_context, steps}} <- normalize_plan(plan),
         {:ok, ordered_steps} <- topo_sort(steps) do
      build_stage_defs(ordered_steps, plan_id, plan_context, opts)
    end
  end

  def to_stage_defs(_plan, _opts), do: {:error, :invalid_plan}

  @doc """
  Same as `to_stage_defs/2` but raises on error.
  """
  @spec to_stage_defs!(plan(), keyword()) :: [StageDef.t()] | no_return()
  def to_stage_defs!(plan, opts \\ []) do
    case to_stage_defs(plan, opts) do
      {:ok, stage_defs} ->
        stage_defs

      {:error, reason} ->
        raise ArgumentError, "invalid plan: #{inspect(reason)}"
    end
  end

  defp normalize_plan(plan) do
    steps = Map.get(plan, :steps)

    if is_map(steps) do
      plan_id = Map.get(plan, :id)
      plan_context = Map.get(plan, :context, %{})

      with {:ok, normalized_steps} <- normalize_steps(steps) do
        {:ok, {plan_id, plan_context, normalized_steps}}
      end
    else
      {:error, :missing_steps}
    end
  end

  defp normalize_steps(steps) do
    steps
    |> Enum.reduce_while({:ok, []}, fn {step_name, step}, {:ok, acc} ->
      case normalize_step(step_name, step) do
        {:ok, normalized} -> {:cont, {:ok, [normalized | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, normalized} -> {:ok, Enum.reverse(normalized)}
      {:error, _} = error -> error
    end
  end

  defp normalize_step(step_name, step) when is_map(step) do
    name = Map.get(step, :name, step_name)
    instruction = Map.get(step, :instruction)
    depends_on = step |> Map.get(:depends_on, []) |> List.wrap()

    cond do
      not is_atom(name) ->
        {:error, {:invalid_step_name, step_name}}

      is_nil(instruction) ->
        {:error, {:missing_instruction, name}}

      not Enum.all?(depends_on, &is_atom/1) ->
        {:error, {:invalid_dependency, name}}

      true ->
        {:ok,
         %{
           name: name,
           id: Map.get(step, :id),
           instruction: instruction,
           depends_on: depends_on,
           opts: Map.get(step, :opts, [])
         }}
    end
  end

  defp normalize_step(step_name, _step), do: {:error, {:invalid_step, step_name}}

  defp build_stage_defs(steps, plan_id, plan_context, opts) do
    stage_module = Keyword.get(opts, :stage_module, Crucible.Stage.PlanStep)
    include_plan_context = Keyword.get(opts, :include_plan_context, true)

    steps
    |> Enum.reduce_while({:ok, []}, fn step, {:ok, acc} ->
      case build_stage_def(step, plan_id, plan_context, stage_module, include_plan_context) do
        {:ok, stage_def} -> {:cont, {:ok, [stage_def | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, stage_defs} -> {:ok, Enum.reverse(stage_defs)}
      {:error, _} = error -> error
    end
  end

  defp build_stage_def(step, plan_id, plan_context, stage_module, include_plan_context) do
    with {:ok, normalized_instruction} <- normalize_instruction(step.instruction) do
      options = %{
        instruction: step.instruction,
        action: normalized_instruction.action,
        params: normalized_instruction.params,
        context: normalized_instruction.context,
        exec_opts: normalized_instruction.opts,
        plan_id: plan_id,
        step_id: step.id,
        step_name: step.name,
        depends_on: step.depends_on
      }

      options =
        if include_plan_context do
          Map.put(options, :plan_context, plan_context)
        else
          options
        end

      {:ok, %StageDef{name: step.name, module: stage_module, options: options}}
    end
  end

  defp normalize_instruction(%{action: action} = instruction) when is_atom(action) do
    params = Map.get(instruction, :params, %{})
    context = Map.get(instruction, :context, %{})
    opts = Map.get(instruction, :opts, [])

    with {:ok, params} <- normalize_map(params, :params),
         {:ok, context} <- normalize_map(context, :context),
         {:ok, opts} <- normalize_opts(opts) do
      {:ok, %{action: action, params: params, context: context, opts: opts}}
    end
  end

  defp normalize_instruction(action) when is_atom(action) do
    {:ok, %{action: action, params: %{}, context: %{}, opts: []}}
  end

  defp normalize_instruction({action, params}) when is_atom(action) do
    with {:ok, params} <- normalize_map(params, :params) do
      {:ok, %{action: action, params: params, context: %{}, opts: []}}
    end
  end

  defp normalize_instruction({action, params, context}) when is_atom(action) do
    with {:ok, params} <- normalize_map(params, :params),
         {:ok, context} <- normalize_map(context, :context) do
      {:ok, %{action: action, params: params, context: context, opts: []}}
    end
  end

  defp normalize_instruction({action, params, context, opts}) when is_atom(action) do
    with {:ok, params} <- normalize_map(params, :params),
         {:ok, context} <- normalize_map(context, :context),
         {:ok, opts} <- normalize_opts(opts) do
      {:ok, %{action: action, params: params, context: context, opts: opts}}
    end
  end

  defp normalize_instruction(invalid), do: {:error, {:invalid_instruction, invalid}}

  defp normalize_map(nil, _label), do: {:ok, %{}}
  defp normalize_map(map, _label) when is_map(map), do: {:ok, map}
  defp normalize_map(list, _label) when is_list(list), do: {:ok, Map.new(list)}
  defp normalize_map(value, label), do: {:error, {:invalid_map, label, value}}

  defp normalize_opts(nil), do: {:ok, []}
  defp normalize_opts(opts) when is_list(opts), do: {:ok, opts}
  defp normalize_opts(value), do: {:error, {:invalid_opts, value}}

  defp topo_sort(steps) do
    steps_by_name = Map.new(steps, &{&1.name, &1})

    case find_missing_dependency(steps_by_name) do
      nil -> perform_topo_sort(steps_by_name)
      {:error, _} = error -> error
    end
  end

  defp perform_topo_sort(steps_by_name) do
    in_degree =
      steps_by_name
      |> Enum.map(fn {name, step} -> {name, length(step.depends_on)} end)
      |> Enum.into(%{})

    dependents = build_dependents(steps_by_name)
    queue = in_degree |> Enum.filter(&match?({_name, 0}, &1)) |> Enum.map(&elem(&1, 0))
    {order, final_in_degree} = drain_queue(Enum.sort(queue), in_degree, dependents, [])

    if length(order) == map_size(steps_by_name) do
      ordered_steps =
        order
        |> Enum.reverse()
        |> Enum.map(&Map.fetch!(steps_by_name, &1))

      {:ok, ordered_steps}
    else
      {:error, {:cycle, cycle_nodes(final_in_degree)}}
    end
  end

  defp cycle_nodes(in_degree) do
    in_degree
    |> Enum.filter(fn {_name, degree} -> degree > 0 end)
    |> Enum.map(&elem(&1, 0))
  end

  defp find_missing_dependency(steps_by_name) do
    Enum.reduce_while(steps_by_name, nil, fn {name, step}, _acc ->
      handle_missing_dependency(name, step, steps_by_name)
    end)
  end

  defp handle_missing_dependency(name, step, steps_by_name) do
    case missing_dependency(step, steps_by_name) do
      nil -> {:cont, nil}
      missing -> {:halt, {:error, {:missing_dependency, missing, name}}}
    end
  end

  defp missing_dependency(step, steps_by_name) do
    Enum.find(step.depends_on, fn dep -> not Map.has_key?(steps_by_name, dep) end)
  end

  defp build_dependents(steps_by_name) do
    Enum.reduce(steps_by_name, %{}, fn {name, step}, acc ->
      Enum.reduce(step.depends_on, acc, fn dep, acc_inner ->
        Map.update(acc_inner, dep, [name], fn existing -> [name | existing] end)
      end)
    end)
  end

  defp drain_queue([], in_degree, _dependents, order), do: {order, in_degree}

  defp drain_queue([current | rest], in_degree, dependents, order) do
    {updated_degree, updated_queue} =
      dependents
      |> Map.get(current, [])
      |> Enum.reduce({in_degree, rest}, fn dependent, {degree_acc, queue_acc} ->
        new_degree = Map.update!(degree_acc, dependent, &(&1 - 1))

        if new_degree[dependent] == 0 do
          {new_degree, insert_sorted(queue_acc, dependent)}
        else
          {new_degree, queue_acc}
        end
      end)

    drain_queue(updated_queue, updated_degree, dependents, [current | order])
  end

  defp insert_sorted(queue, item) do
    (queue ++ [item]) |> Enum.sort()
  end
end
