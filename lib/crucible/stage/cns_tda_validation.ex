defmodule Crucible.Stage.CNSTDAValidation do
  @moduledoc """
  Runs topological data analysis (TDA) on CNS reasoning structures via a configured adapter.

  Results are attached to:
    * `context.assigns[:cns_tda_results]` - per-SNO metrics
    * `context.metrics[:cns_tda]` - aggregate summary
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.CNS.TDANoop

  @impl true
  def run(%Context{} = ctx, opts) do
    opts = normalize_opts(opts)
    adapter = resolve_adapter(opts)
    examples = ctx.examples || []
    outputs = ctx.outputs || []

    case adapter.compute_tda(examples, outputs, opts) do
      {:ok, %{results: results, summary: summary} = payload} ->
        new_assigns =
          (ctx.assigns || %{})
          |> Map.put(:cns_tda_results, results)
          |> Map.put(:cns_tda_raw, payload)

        new_metrics =
          (ctx.metrics || %{})
          |> Map.put(:cns_tda, summary)

        {:ok, %Context{ctx | assigns: new_assigns, metrics: new_metrics}}

      {:error, reason} ->
        {:error, {:cns_tda_failed, reason}}
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: "CNSTDAValidation",
      description: "Compute CNS TDA metrics via adapter",
      adapter: opts[:adapter] || Application.get_env(:crucible_framework, :cns_tda_adapter)
    }
  end

  defp resolve_adapter(opts) do
    mod =
      case opts do
        %{} -> Map.get(opts, :adapter)
        kw when is_list(kw) -> Keyword.get(kw, :adapter)
        _ -> nil
      end

    mod || Application.get_env(:crucible_framework, :cns_tda_adapter, TDANoop)
  end

  defp normalize_opts(nil), do: %{}
  defp normalize_opts(opts) when is_map(opts), do: opts
  defp normalize_opts(opts) when is_list(opts), do: Map.new(opts)
  defp normalize_opts(_), do: %{}
end
