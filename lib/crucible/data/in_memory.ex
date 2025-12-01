defmodule Crucible.Data.InMemory do
  @moduledoc """
  Minimal dataset provider for in-memory lists or JSONL files.
  """

  @behaviour Crucible.Data.Provider

  alias CrucibleIR.DatasetRef

  @impl true
  def load(%DatasetRef{options: opts} = dataset_ref, _opts) do
    cond do
      is_list(opts[:data]) ->
        {:ok, opts[:data]}

      path = opts[:path] ->
        load_path(dataset_ref, path)

      true ->
        {:error, :no_dataset_supplied}
    end
  end

  defp load_path(_dataset_ref, path) do
    case File.exists?(path) do
      true ->
        {:ok, stream_jsonl(path)}

      false ->
        {:error, {:missing_file, path}}
    end
  end

  defp stream_jsonl(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(fn line ->
      case Jason.decode(line) do
        {:ok, data} -> data
        {:error, _} -> %{raw: line}
      end
    end)
  end
end
