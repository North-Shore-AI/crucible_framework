defmodule Crucible.Stage.Report do
  @moduledoc """
  Emits reports for experiment outputs and records artifacts.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias CrucibleFramework.Persistence
  alias CrucibleIR.OutputSpec
  require Logger

  @impl true
  def run(%Context{experiment: experiment} = ctx, _opts) do
    specs = experiment.outputs || []

    {new_outputs, artifacts} =
      Enum.map_reduce(specs, [], fn %OutputSpec{} = spec, acc ->
        rendered = render(spec, ctx)
        persisted = maybe_persist(spec, rendered, ctx)
        {%{name: spec.name, rendered: rendered}, persisted ++ acc}
      end)

    artifacts
    |> Enum.reject(&is_nil/1)
    |> Enum.each(fn
      {:ok, _} -> :ok
      {:error, reason} -> Logger.warning("Failed to persist artifact: #{inspect(reason)}")
      other -> Logger.debug("Artifact write result: #{inspect(other)}")
    end)

    {:ok, %Context{ctx | outputs: ctx.outputs ++ new_outputs}}
  end

  defp render(%OutputSpec{formats: formats} = spec, %Context{} = ctx) do
    formats
    |> Enum.map(fn format ->
      case format do
        :markdown ->
          %{format: :markdown, body: to_markdown(ctx)}

        :json ->
          %{format: :json, body: Jason.encode!(%{metrics: ctx.metrics, outputs: ctx.outputs})}

        other ->
          %{format: other, body: "Unsupported format #{other}"}
      end
    end)
    |> tap(&maybe_print_to_stdout(&1, spec))
  end

  defp maybe_print_to_stdout(entries, %OutputSpec{sink: :stdout} = spec) do
    Enum.each(entries, fn %{format: fmt, body: body} ->
      IO.puts("== #{spec.name} (#{fmt}) ==")
      IO.puts(body)
    end)
  end

  defp maybe_print_to_stdout(_entries, _spec), do: :ok

  defp maybe_persist(%OutputSpec{sink: :file} = spec, rendered, %Context{} = ctx) do
    for %{format: fmt, body: body} <- rendered do
      path = output_path(spec, ctx, fmt)
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, body)

      if run = ctx.assigns[:run_record] do
        Persistence.record_artifact(run, %{
          name: spec.name,
          type: Atom.to_string(spec.sink),
          format: Atom.to_string(fmt),
          location: path
        })
      end
    end
  end

  defp maybe_persist(_spec, _rendered, _ctx), do: []

  defp output_path(
         %OutputSpec{options: opts} = spec,
         %Context{experiment_id: exp_id, run_id: run_id},
         fmt
       ) do
    case opts[:path] do
      nil ->
        "reports/#{exp_id}-#{run_id}-#{spec.name}.#{fmt}"

      path ->
        path
    end
  end

  defp to_markdown(%Context{} = ctx) do
    metrics_json = ctx.metrics |> Jason.encode!() |> Jason.Formatter.pretty_print()
    outputs_json = ctx.outputs |> Jason.encode!() |> Jason.Formatter.pretty_print()

    """
    # Experiment #{ctx.experiment_id}

    ## Metrics
    ```json
    #{metrics_json}
    ```

    ## Outputs
    ```json
    #{outputs_json}
    ```
    """
  end
end
