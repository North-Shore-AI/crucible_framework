defmodule Crucible.Stage.CNSMetrics do
  @moduledoc """
  Integrates CNS evaluation results into the pipeline.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.CNS.Adapter
  alias Crucible.CNS.Noop

  @impl true
  def run(%Context{} = ctx, opts) do
    opts = normalize_opts(opts)
    adapter = opts[:adapter] || Application.get_env(:crucible_framework, :cns_adapter, Noop)

    with true <- implements?(adapter),
         {:ok, metrics} <- adapter.evaluate(ctx.examples || [], ctx.outputs, opts) do
      {:ok, %Context{ctx | metrics: Map.put(ctx.metrics, :cns, metrics)}}
    else
      false -> {:error, {:invalid_cns_adapter, adapter}}
      {:error, reason} -> {:error, reason}
    end
  end

  defp normalize_opts(nil), do: %{}
  defp normalize_opts(opts) when is_map(opts), do: opts
  defp normalize_opts(opts) when is_list(opts), do: Map.new(opts)
  defp normalize_opts(_), do: %{}

  defp implements?(adapter) do
    function_exported?(adapter, :evaluate, 3) and
      Adapter in (adapter.module_info(:attributes)[:behaviour] || [])
  end
end
