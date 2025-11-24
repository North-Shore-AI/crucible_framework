defmodule CrucibleFramework.Persistence.ExperimentRecord do
  @moduledoc """
  Ecto schema for persisted experiment definitions.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{
          id: String.t(),
          definition: map(),
          owner: String.t() | nil,
          tags: [String.t()],
          metadata: map(),
          inserted_at: DateTime.t() | nil,
          updated_at: DateTime.t() | nil
        }

  @primary_key {:id, :string, autogenerate: false}
  schema "experiments" do
    field(:definition, :map)
    field(:owner, :string)
    field(:tags, {:array, :string}, default: [])
    field(:metadata, :map, default: %{})

    timestamps(type: :utc_datetime)
  end

  def changeset(schema, attrs) do
    schema
    |> cast(attrs, [:id, :definition, :owner, :tags, :metadata])
    |> validate_required([:id, :definition])
  end
end
