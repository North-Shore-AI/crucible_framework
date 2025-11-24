defmodule CrucibleFramework.Repo.Migrations.CreateCrucibleCoreTables do
  use Ecto.Migration

  def change do
    create table(:experiments, primary_key: false) do
      add(:id, :string, primary_key: true)
      add(:definition, :map, null: false)
      add(:owner, :string)
      add(:tags, {:array, :string}, default: [])
      add(:metadata, :map, default: %{})

      timestamps(type: :utc_datetime)
    end

    create table(:runs, primary_key: false) do
      add(:id, :binary_id, primary_key: true)

      add(:experiment_id, references(:experiments, type: :string, on_delete: :nothing),
        null: false
      )

      add(:status, :string, null: false)
      add(:context, :map, default: %{})
      add(:metrics, :map, default: %{})
      add(:outputs, :map, default: %{})
      add(:metadata, :map, default: %{})

      timestamps(type: :utc_datetime)
    end

    create(index(:runs, [:experiment_id]))
    create(index(:runs, [:status]))

    create table(:artifacts, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:run_id, references(:runs, type: :binary_id, on_delete: :delete_all), null: false)
      add(:name, :string, null: false)
      add(:type, :string, null: false)
      add(:location, :string)
      add(:format, :string)
      add(:metadata, :map, default: %{})

      timestamps(type: :utc_datetime)
    end

    create(index(:artifacts, [:run_id]))
    create(index(:artifacts, [:type]))
  end
end
