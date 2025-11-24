defmodule Crucible.Stage.DataLoad do
  @moduledoc """
  Loads dataset(s) for an experiment and prepares batches.
  """

  @behaviour Crucible.Stage

  alias Crucible.Context
  alias Crucible.Data.InMemory
  alias Crucible.IR.DatasetRef

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    dataset_refs =
      experiment.dataset
      |> List.wrap()
      |> Enum.reject(&is_nil/1)

    case dataset_refs do
      [] ->
        {:ok, ctx}

      [ref | _rest] ->
        with {:ok, provider} <- provider_module(ref, opts),
             {:ok, examples} <- provider.load(ref, opts) do
          limited =
            case limit(opts, ref) do
              :infinity -> Enum.to_list(examples)
              n -> examples |> Enum.take(n) |> Enum.to_list()
            end

          shaped = shape_examples(limited, opts)

          batch_size = batch_size(opts, ref)
          batches = Enum.chunk_every(shaped, batch_size, batch_size, [])

          {:ok,
           %Context{
             ctx
             | dataset: shaped,
               examples: shaped,
               batches: batches,
               metrics:
                 Map.put(ctx.metrics, :data_load, %{total: length(shaped), batch_size: batch_size})
           }}
        end
    end
  end

  defp batch_size(opts, %DatasetRef{options: ref_opts}) do
    opts[:batch_size] || ref_opts[:batch_size] || 4
  end

  defp limit(opts, %DatasetRef{options: ref_opts}) do
    opts[:limit] || ref_opts[:limit] || :infinity
  end

  defp provider_module(%DatasetRef{provider: provider}, opts) do
    cond do
      provider && Code.ensure_loaded?(provider) -> {:ok, provider}
      mod = opts[:provider] -> {:ok, mod}
      true -> {:ok, InMemory}
    end
  end

  defp shape_examples(examples, opts) do
    case opts do
      %{map_fn: mapper} when is_function(mapper, 1) ->
        Enum.map(examples, mapper)

      %{input_key: input_key} ->
        output_key = opts[:output_key] || :completion

        Enum.map(examples, fn example ->
          %{
            input: fetch(example, input_key),
            output: fetch(example, output_key)
          }
        end)

      _ ->
        examples
    end
  end

  defp fetch(example, key) do
    Map.get(example, key) || Map.get(example, to_string(key))
  end
end
