defmodule Crucible.Registry do
  @moduledoc """
  Resolves configured backends and stages.
  """

  @spec backend_module(atom()) :: {:ok, module()} | {:error, term()}
  def backend_module(id) when is_atom(id) do
    case Application.fetch_env(:crucible_framework, :backends) do
      {:ok, map} ->
        case Map.fetch(map, id) do
          {:ok, mod} -> {:ok, mod}
          :error -> {:error, {:unknown_backend, id}}
        end

      :error ->
        {:error, :no_backends_configured}
    end
  end

  @spec stage_module(atom()) :: {:ok, module()} | {:error, term()}
  def stage_module(name) when is_atom(name) do
    case Application.fetch_env(:crucible_framework, :stage_registry) do
      {:ok, map} ->
        case Map.fetch(map, name) do
          {:ok, mod} -> {:ok, mod}
          :error -> {:error, {:unknown_stage, name}}
        end

      :error ->
        {:error, :no_stage_registry}
    end
  end
end
