defmodule Crucible.Stage.Analysis.SurrogateValidation do
  @moduledoc """
  Runs surrogate topology/logic validation via a configured adapter.

  Results are attached to:
    * `context.assigns[:analysis_surrogates]` - per-output surrogate metrics
    * `context.metrics[:analysis_surrogates]` - aggregate summary
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.Analysis.SurrogateNoop

  @impl true
  def run(%Context{} = ctx, opts) do
    opts = normalize_opts(opts)
    adapter = resolve_adapter(opts)
    examples = ctx.examples || []
    outputs = ctx.outputs || []

    case adapter.compute_surrogates(examples, outputs, opts) do
      {:ok, %{results: results, summary: summary} = payload} ->
        new_assigns =
          (ctx.assigns || %{})
          |> Map.put(:analysis_surrogates, results)
          |> Map.put(:analysis_surrogates_raw, payload)

        new_metrics =
          (ctx.metrics || %{})
          |> Map.put(:analysis_surrogates, summary)

        {:ok, %Context{ctx | assigns: new_assigns, metrics: new_metrics}}

      {:error, reason} ->
        {:error, {:analysis_surrogates_failed, reason}}
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: "AnalysisSurrogateValidation",
      description: "Compute surrogate metrics via adapter",
      adapter: opts[:adapter] || Application.get_env(:crucible_framework, :analysis_surrogate_adapter)
    }
  end

  defp resolve_adapter(opts),
    do: Map.get(opts, :adapter) || adapter_default()

  defp adapter_default,
    do: Application.get_env(:crucible_framework, :analysis_surrogate_adapter, SurrogateNoop)

  defp normalize_opts(nil), do: %{}
  defp normalize_opts(opts) when is_map(opts), do: opts
  defp normalize_opts(opts) when is_list(opts), do: Map.new(opts)
  defp normalize_opts(_), do: %{}
end
