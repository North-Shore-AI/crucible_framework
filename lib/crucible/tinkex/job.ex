defmodule Crucible.Tinkex.Job do
  @moduledoc """
  Job manifest representing a training or evaluation request submitted through
  the Crucible Tinkex API surface.

  This struct deliberately excludes any Tinkex credentials; all credential
  handling remains internal to the `crucible_tinkex` application to satisfy
  the CNS single-owner rule.
  """

  alias Crucible.Tinkex

  @type status :: :pending | :queued | :running | :completed | :failed | :canceled

  @enforce_keys [:id, :status, :inserted_at, :artifacts_path, :spec]
  defstruct [
    :id,
    :name,
    :status,
    :inserted_at,
    :updated_at,
    :artifacts_path,
    :spec,
    :error
  ]

  @type t :: %__MODULE__{
          id: String.t(),
          name: String.t() | nil,
          status: status(),
          inserted_at: DateTime.t(),
          updated_at: DateTime.t(),
          artifacts_path: String.t(),
          spec: map(),
          error: String.t() | nil
        }

  @doc """
  Builds a new job manifest from request parameters.

  The caller may supply a `:name`, `:dataset_manifest`, and `:hyperparams`.
  """
  @spec new(map()) :: {:ok, t()} | {:error, :invalid_request}
  def new(params) when is_map(params) do
    now = DateTime.utc_now()
    id = Map.get(params, "job_id") || Tinkex.generate_id()
    dataset_manifest = Map.get(params, "dataset_manifest")

    if is_nil(dataset_manifest) or dataset_manifest == "" do
      {:error, :invalid_request}
    else
      job = %__MODULE__{
        id: id,
        name: Map.get(params, "name"),
        status: :pending,
        inserted_at: now,
        updated_at: now,
        artifacts_path: artifacts_root(id),
        spec: %{
          dataset_manifest: dataset_manifest,
          hyperparams: Map.get(params, "hyperparams") || %{},
          metadata: Map.get(params, "metadata") || %{},
          type: Map.get(params, "type", "training")
        }
      }

      {:ok, job}
    end
  end

  @doc """
  Updates the job status while refreshing timestamps.
  """
  @spec with_status(t(), status(), keyword()) :: t()
  def with_status(%__MODULE__{} = job, status, opts \\ []) do
    %__MODULE__{job | status: status, updated_at: DateTime.utc_now(), error: opts[:error]}
  end

  defp artifacts_root(job_id), do: Path.join(["artifacts", "crucible", job_id])
end
