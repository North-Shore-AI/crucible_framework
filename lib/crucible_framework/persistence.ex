defmodule CrucibleFramework.Persistence do
  @moduledoc """
  Persistence helpers for recording experiments, runs, and artifacts.
  """

  alias Crucible.IR.Experiment
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
    metrics = attrs |> Map.get(:metrics, run.metrics) |> stringify_keys()
    outputs = attrs |> Map.get(:outputs, run.outputs)
    metadata = attrs |> Map.get(:metadata, run.metadata) |> stringify_keys()

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
end
