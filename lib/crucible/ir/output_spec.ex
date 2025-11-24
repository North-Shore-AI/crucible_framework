defmodule Crucible.IR.OutputSpec do
  @derive {Jason.Encoder, only: [:name, :description, :formats, :sink, :options]}
  @moduledoc """
  Describes experiment outputs and where to send them.
  """

  @type format :: :markdown | :json | :html | :latex | :csv | :parquet
  @type sink :: :file | :stdout | :liveview | :s3 | :custom

  @type t :: %__MODULE__{
          name: atom(),
          description: String.t() | nil,
          formats: [format()],
          sink: sink(),
          options: map()
        }

  @enforce_keys [:name]
  defstruct [
    :name,
    :description,
    formats: [],
    sink: :file,
    options: %{}
  ]
end
