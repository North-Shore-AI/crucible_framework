defmodule CrucibleFramework.Persistence do
  @moduledoc """
  Persistence helpers for recording experiments, runs, and artifacts.
  """

  alias CrucibleIR.Experiment
  alias CrucibleFramework.Repo
  alias CrucibleFramework.Persistence.{ArtifactRecord, ExperimentRecord, RunRecord}

  @doc """
  Inserts or updates an experiment definition.
  """
  @spec upsert_experiment(Experiment.t()) ::
          {:ok, ExperimentRecord.t()} | {:error, Ecto.Changeset.t()}
  def upsert_experiment(%Experiment{} = experiment) do
    attrs = %{
      id: experiment.id,
      definition: Map.from_struct(experiment),
      owner: experiment.owner,
      tags: experiment.tags || [],
      metadata: experiment.metadata || %{}
    }

    changeset = ExperimentRecord.changeset(%ExperimentRecord{}, attrs)

    Repo.insert(changeset,
      on_conflict: {:replace, [:definition, :owner, :tags, :metadata, :updated_at]},
      conflict_target: :id,
      returning: true
    )
  end

  @doc """
  Creates a run record marked as `running`.
  """
  @spec start_run(Experiment.t(), keyword()) ::
          {:ok, RunRecord.t()} | {:error, Ecto.Changeset.t()}
  def start_run(%Experiment{} = experiment, opts \\ []) do
    with {:ok, _exp} <- upsert_experiment(experiment) do
      attrs = %{
        experiment_id: experiment.id,
        status: "running",
        metadata: Keyword.get(opts, :metadata, %{})
      }

      %RunRecord{}
      |> RunRecord.changeset(attrs)
      |> Repo.insert()
    end
  end

  @doc """
  Updates a run's status/metrics/outputs.
  """
  @spec finish_run(RunRecord.t(), String.t(), map()) ::
          {:ok, RunRecord.t()} | {:error, Ecto.Changeset.t()}
  def finish_run(%RunRecord{} = run, status, attrs \\ %{}) do
    metrics = attrs |> Map.get(:metrics, run.metrics) |> sanitize() |> stringify_keys()
    outputs = attrs |> Map.get(:outputs, run.outputs) |> sanitize()
    metadata = attrs |> Map.get(:metadata, run.metadata) |> sanitize() |> stringify_keys()

    run
    |> RunRecord.changeset(%{
      status: status,
      metrics: metrics,
      outputs: outputs,
      metadata: metadata
    })
    |> Repo.update()
  end

  @doc """
  Records an artifact generated during a run.
  """
  @spec record_artifact(RunRecord.t(), map()) ::
          {:ok, ArtifactRecord.t()} | {:error, Ecto.Changeset.t()}
  def record_artifact(%RunRecord{id: run_id}, attrs) do
    attrs =
      attrs
      |> Map.put_new(:run_id, run_id)
      |> Map.update(:name, nil, &to_string/1)

    %ArtifactRecord{}
    |> ArtifactRecord.changeset(attrs)
    |> Repo.insert()
  end

  defp stringify_keys(map) when is_map(map) do
    map
    |> Enum.map(fn {k, v} -> {to_string(k), v} end)
    |> Enum.into(%{})
  end

  defp stringify_keys(other), do: other

  defp sanitize(%_{} = struct) do
    struct
    |> Map.from_struct()
    |> sanitize()
  end

  defp sanitize(map) when is_map(map) do
    map
    |> Enum.map(fn {k, v} -> {sanitize_key(k), sanitize(v)} end)
    |> Enum.into(%{})
  end

  defp sanitize(list) when is_list(list), do: Enum.map(list, &sanitize/1)

  defp sanitize(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> sanitize()
  end

  defp sanitize(value)
       when is_pid(value) or is_reference(value) or is_function(value) or is_port(value),
       do: inspect(value)

  defp sanitize(value), do: value

  defp sanitize_key(key) when is_atom(key), do: Atom.to_string(key)
  defp sanitize_key(key), do: key
end
