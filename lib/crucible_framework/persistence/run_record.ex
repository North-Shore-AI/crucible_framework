defmodule CrucibleFramework.Persistence.RunRecord do
  @moduledoc """
  Ecto schema for experiment runs.
  """

  use Ecto.Schema
  import Ecto.Changeset

  alias CrucibleFramework.Persistence.ExperimentRecord

  @primary_key {:id, :binary_id, autogenerate: true}
  schema "runs" do
    belongs_to(:experiment, ExperimentRecord, references: :id, type: :string)

    field(:status, :string, default: "pending")
    field(:context, :map, default: %{})
    field(:metrics, :map, default: %{})
    field(:outputs, :map, default: %{})
    field(:metadata, :map, default: %{})

    timestamps(type: :utc_datetime)
  end

  def changeset(schema, attrs) do
    schema
    |> cast(attrs, [:experiment_id, :status, :context, :metrics, :outputs, :metadata])
    |> validate_required([:experiment_id, :status])
    |> foreign_key_constraint(:experiment_id)
  end
end
