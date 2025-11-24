defmodule Crucible.Stage do
  @moduledoc """
  Behaviour for a single experiment pipeline stage.

  Stages transform a `%Crucible.Context{}` and may enrich metrics, outputs,
  or orchestrate I/O with backends and external systems.
  """

  alias Crucible.Context

  @type opts :: map()

  @callback run(context :: Context.t(), opts :: opts()) ::
              {:ok, Context.t()} | {:error, term()}

  @callback describe(opts :: opts()) :: map()
  @optional_callbacks describe: 1
end
