defmodule CrucibleFramework do
  @moduledoc """
  Public entrypoints for running Crucible experiments.
  """

  alias Crucible.Pipeline.Runner
  alias CrucibleIR.Experiment

  @spec run(Experiment.t(), keyword()) :: {:ok, Crucible.Context.t()} | {:error, term()}
  def run(%Experiment{} = experiment, opts \\ []) do
    Runner.run(experiment, opts)
  end
end
