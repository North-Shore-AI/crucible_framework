defmodule Crucible.BackendManager do
  @moduledoc """
  Helper utilities for resolving and initializing backends.
  """

  alias Crucible.IR.BackendRef
  alias Crucible.Registry

  @spec resolve_module(BackendRef.t()) :: {:ok, module()} | {:error, term()}
  def resolve_module(%BackendRef{id: id}) do
    Registry.backend_module(id)
  end

  @spec ensure_state(map(), BackendRef.t()) ::
          {:ok, module(), term(), map()} | {:error, term()}
  def ensure_state(backend_state_map, %BackendRef{} = ref) do
    case Map.fetch(backend_state_map, ref.id) do
      {:ok, %{module: mod, state: state}} ->
        {:ok, mod, state, backend_state_map}

      :error ->
        with {:ok, mod} <- resolve_module(ref),
             {:ok, state} <- mod.init(ref.id, ref.options) do
          {:ok, mod, state, Map.put(backend_state_map, ref.id, %{module: mod, state: state})}
        end
    end
  end

  @spec ensure_session(map(), module(), term(), BackendRef.t(), term()) ::
          {:ok, term(), map()} | {:error, term()}
  def ensure_session(session_map, mod, state, %BackendRef{} = ref, experiment) do
    key = {ref.id, experiment.id}

    case Map.fetch(session_map, key) do
      {:ok, session} ->
        {:ok, session, session_map}

      :error ->
        case mod.start_session(state, experiment) do
          {:ok, session} -> {:ok, session, Map.put(session_map, key, session)}
          {:error, reason} -> {:error, reason}
        end
    end
  end
end
